# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmcls.apis import init_model, show_result_pyplot

import os
import json
from tqdm import tqdm
import cv2
import tabulate
import numpy as np
import torch
from mmcls.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
from mmcls.models.utils.augment import BatchMixupLayer

from demo.test_time_mixup import test_time_mixup

def inference_model_for_softmax(model, imgs, source_train_mixed_img, mixup_lambda=1.0):
    """Inference image(s) with the classifier.

    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    # prepare for mixup 

    # build the data pipeline
    assert len(imgs) > 0, f"Unexcepted empty imgs."

    if cfg.data.test.pipeline[0]['type'] == 'LoadImageFromFile':
        cfg.data.test.pipeline.pop(0)
    if cfg.data.test.pipeline[0]['type'] != 'LoadImageFromMixupFile':
        cfg.data.test.pipeline.insert(0, dict(type='LoadImageFromMixupFile'))

    test_pipeline = Compose(cfg.data.test.pipeline)
    test_input_data = [
        test_pipeline(
            dict(img_info=dict(filename=img, source_train_mixed_img=source_train_mixed_img), img_prefix=None, mixup_info=dict(lam=mixup_lambda))
        )
        for img in imgs
    ]
    data = collate(test_input_data, samples_per_gpu=len(test_input_data))
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # forward the model
    with torch.no_grad():
        scores = model(return_loss=False, **data)
        pred_score = np.max(scores, axis=1)
        pred_label = np.argmax(scores, axis=1)
        result = {'pred_label': pred_label, 'pred_score': pred_score, 'scores': scores}
    
    # result['pred_class'] = model.CLASSES[result['pred_label']]
    return result


class ForwardResult:
    gt_label_int: int
    pred_label_int: int
    file_path: str
    pred_score: float
    pred_scores: np.ndarray
    recovered_scores: np.ndarray


    def __init__(self, file_path, gt_label_int, pred_label_int, pred_score, pred_scores, recovered_scores=[]):
        self.file_path = file_path
        self.gt_label_int = gt_label_int
        self.pred_label_int = pred_label_int
        self.pred_score = pred_score
        self.pred_scores = pred_scores
        self.recovered_scores = recovered_scores

    @property
    def recovered_label_int(self):
        return np.argmax(self.recovered_scores, axis=0)

    @property
    def kl_div_between_pred_scores_and_recovered_scores(self):
        return test_time_mixup.calculate_kl_div(test_time_mixup.softmax(self.pred_scores), test_time_mixup.softmax(self.recovered_scores))

    def calculate_compositive_label_int(self, alpha=1.0):
        compositive_scores = self.calculate_compositive_scores(alpha)
        return np.argmax(compositive_scores, axis=0)

    def calculate_compositive_scores(self, alpha=1.0):
        return test_time_mixup.calculate_compositive_scores(self.pred_scores, self.recovered_scores, lambda_value=alpha)

    def __repr__(self):
        table_data = [
            ['file_path', self.file_path],
            ['gt_label_int', self.gt_label_int],
            ['pred_label_int', self.pred_label_int],
            ['pred_score', self.pred_score],
            ['pred_scores', [round(v, 6) for v in self.pred_scores]],
            ['recovered_scores', [round(v, 6) for v in self.recovered_scores]],
            ['softmax(recovered_scores)', test_time_mixup.softmax(self.recovered_scores)],
            ['kl_div_between_p_and_r', self.kl_div_between_pred_scores_and_recovered_scores],
        ]
        return tabulate.tabulate(table_data)
        # return f"{self.file_path}, {self.gt_label_int}, {self.pred_label_int}, {self.pred_score}, {self.pred_scores}, {self.recovered_scores}"


def build_source_train_mixed_img():
    source_train_image_paths = [
        'data/cifar10/train_split/0/10008_airplane.png',
        # 'data/cifar10/train_split/1/10000_automobile.png',
        # 'data/cifar10/train_split/2/10018_bird.png',
        # 'data/cifar10/train_split/3/10005_cat.png',
        # 'data/cifar10/train_split/4/10006_deer.png',
        # 'data/cifar10/train_split/5/10014_dog.png',
        # 'data/cifar10/train_split/6/0_frog.png',
        # 'data/cifar10/train_split/7/10028_horse.png',
        # 'data/cifar10/train_split/8/10003_ship.png',
        # 'data/cifar10/train_split/9/1000_truck.png',
    ]
    source_train_imgs = [cv2.imread(p) for p in source_train_image_paths]
    source_train_mixed_img = 0.0 * source_train_imgs[0]
    for source_train_img in source_train_imgs:
        source_train_mixed_img += 1 / len(source_train_imgs) * source_train_img
    # cv2.imwrite("temp.png", source_train_mixed_img)
    # exit()

    soft_labels = test_time_mixup.generate_soft_labels([0], 10)
    
    return source_train_mixed_img, soft_labels


def evaluate_for_dataset(model, dataset_root: str, max_count=999999, mixup_lambda=1.0):
    # read source_train_image
    source_train_mixed_img, train_soft_label = build_source_train_mixed_img()
    # cv2.imwrite("source_train_mixed_img.jpg", source_train_mixed_img)
    # print(f"train_soft_label: {train_soft_label}")

    fr_list = []
    cls_names = os.listdir(dataset_root)
    forward_result_list_to_perform = []
    for cls_name in cls_names:
        cls_dir_path = os.path.join(dataset_root, cls_name)
        cls_dir_file_names = os.listdir(cls_dir_path)
        for cls_dir_file_name in tqdm(cls_dir_file_names):
            img_path = os.path.join(cls_dir_path, cls_dir_file_name)
            gt_label_int = int(cls_name)
            fr = ForwardResult(img_path, gt_label_int, 0, 0, [])
            forward_result_list_to_perform.append(fr)

    print(f"Collect image paths to forward finished.")

    BATCH_SIZE = 32
    for idx in tqdm(range(0, len(forward_result_list_to_perform), BATCH_SIZE)):
        batch_data = forward_result_list_to_perform[idx: idx+BATCH_SIZE]
        # TODO: add img_target correctly
        img_path_list = [fr.file_path for fr in batch_data]
        gt_label_int_list = [fr.gt_label_int for fr in batch_data]

        pred_results_dict = inference_model_for_softmax(model, img_path_list, source_train_mixed_img, mixup_lambda=mixup_lambda)

        pred_label_int_list = [int(p) for p in pred_results_dict['pred_label']]
        pred_score_list = [float(p) for p in pred_results_dict['pred_score']]
        pred_scores_list = [p for p in pred_results_dict['scores']]
        recovered_scores_list = [test_time_mixup.calculate_recovered_scores(train_soft_label, p, source_ratio=1-mixup_lambda) for p in pred_results_dict['scores']]

        cur_fr_list = [ForwardResult(*x) for x in zip(img_path_list, gt_label_int_list, pred_label_int_list, pred_score_list, pred_scores_list, recovered_scores_list)]
        fr_list.extend(cur_fr_list)

    # calculate
    total_count = len(fr_list)
    correct_count = len(list(filter(lambda x: x.gt_label_int == x.pred_label_int, fr_list)))
    correct_count_by_recovered_scores = len(list(filter(lambda x: x.gt_label_int == x.recovered_label_int, fr_list)))
    accuracy = correct_count / total_count
    accuracy_by_recovered_scores = correct_count_by_recovered_scores / total_count
    
    correct_count_by_compositive_scores_list = [
        len(list(filter(lambda x: x.gt_label_int == x.calculate_compositive_label_int(alpha), fr_list)))
        for alpha in np.arange(0.1, 1.0, 0.1)
    ]
    accuracy_by_compositive_scores_list = [
        (count / total_count) for count in correct_count_by_compositive_scores_list
    ]
    kl_div_list = [
        fr.kl_div_between_pred_scores_and_recovered_scores for fr in fr_list
    ]
    print(fr_list)
    average_kl_dev = sum(kl_div_list) / len(kl_div_list)
    
    return {
        'accuracy': accuracy, 
        'accuracy_by_recovered_scores': accuracy_by_recovered_scores, 
        'accuracy_by_compositive_scores_list': accuracy_by_compositive_scores_list,
        'average_kl_dev': average_kl_dev
    }


def draw_plot_lines(evaluated_result_list):
    import matplotlib.pyplot as plt
    x = [v['mixup_lambda'] for v in evaluated_result_list]
    y = [v['accuracy'] for v in evaluated_result_list]
    y_recovered = [v['accuracy_by_recovered_scores'] for v in evaluated_result_list]
    plt.plot(x, y, linewidth=1, color='orange', marker="o", label="Accuracy")
    plt.plot(x, y_recovered, linewidth=1, color='green', marker="d", label="Accuracy by Recorvered")
    plt.xticks(x)
    plt.grid()
    plt.savefig("temp.png")



def evaluate_for_different_mixup_lambda(model, args):
    evaluated_result_list = []
    # for mixup_lambda in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    for mixup_lambda in [0.8, 0.9, 1.0]:
        eval_result = evaluate_for_dataset(model, args.dataset_root, mixup_lambda=mixup_lambda)
        evaluated_result_list.append({
            'mixup_lambda': mixup_lambda,
            'accuracy': eval_result['accuracy'],
            'accuracy_by_recovered_scores': eval_result['accuracy_by_recovered_scores'],
        })
    print(evaluated_result_list)
    draw_plot_lines(evaluated_result_list)


def main():
    parser = ArgumentParser()
    parser.add_argument('dataset_root', help='Dataset Root')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--mixup_lambda', help='mixup_lambda', default=1.0)
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    # evaluate once
    eval_res = evaluate_for_dataset(model, args.dataset_root, mixup_lambda=float(args.mixup_lambda))
    print(eval_res)

    # evaluate many times
    # evaluate_for_different_mixup_lambda(model, args)
    



if __name__ == '__main__':
    main()
