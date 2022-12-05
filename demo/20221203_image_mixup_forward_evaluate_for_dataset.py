# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmcls.apis import init_model, show_result_pyplot

import os
import json
from tqdm import tqdm

import numpy as np
import torch
from mmcls.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
from mmcls.models.utils.augment import BatchMixupLayer


def inference_model_for_softmax(model, imgs, img_target):
    """Inference image(s) with the classifier.

    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    # prepare for mixup 

    # build the data pipeline
    assert len(imgs) > 0, f"Unexcepted empty imgs."

    
    if cfg.data.test.pipeline[0]['type'] != 'LoadImageFromMixupFile':
        cfg.data.test.pipeline.insert(0, dict(type='LoadImageFromMixupFile'))

    test_pipeline = Compose(cfg.data.test.pipeline)
    test_input_data = [
        test_pipeline(
            dict(img_info=dict(filename=img, filename_target=img_target), img_prefix=None, mixup_info=dict(lam=1.0))
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
    pred_scores: list

    def __init__(self, file_path, gt_label_int, pred_label_int, pred_score, pred_scores):
        self.file_path = file_path
        self.gt_label_int = gt_label_int
        self.pred_label_int = pred_label_int
        self.pred_score = pred_score
        self.pred_scores = pred_scores

    def __repr__(self):
        return f"{self.file_path}, {self.gt_label_int}, {self.pred_label_int}, {self.pred_score}, {self.pred_scores}"


def evaluate_for_dataset(model, dataset_root: str, img_target, max_count=999999):

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

    batch_size = 16
    for idx in tqdm(range(0, len(forward_result_list_to_perform), batch_size)):
        batch_data = forward_result_list_to_perform[idx: idx+batch_size]
        # TODO: add img_target correctly
        img_path_list = [fr.file_path for fr in batch_data]
        gt_label_int_list = [fr.gt_label_int for fr in batch_data]

        pred_results_dict = inference_model_for_softmax(model, img_path_list, img_path_list[0])

        pred_label_int_list = [int(p) for p in pred_results_dict['pred_label']]
        pred_score_list = [float(p) for p in pred_results_dict['pred_score']]
        pred_scores_list = [p.tolist() for p in pred_results_dict['scores']]

        cur_fr_list = [ForwardResult(*x) for x in zip(img_path_list, gt_label_int_list, pred_label_int_list, pred_score_list, pred_scores_list)]
        fr_list.extend(cur_fr_list)

    # calculate
    total_count = len(fr_list)
    correct_count = len(list(filter(lambda x: x.gt_label_int == x.pred_label_int, fr_list)))
    accuracy = correct_count / total_count
    
    print(f'Accuracy is: {accuracy}')




def main():
    parser = ArgumentParser()
    parser.add_argument('dataset_root', help='Dataset Root')
    parser.add_argument('img_target', help='Target Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    evaluate_for_dataset(model, args.dataset_root, args.img_target)


if __name__ == '__main__':
    main()
