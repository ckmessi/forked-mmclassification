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


def inference_model_for_softmax(model, img, img_target):
    """Inference image(s) with the classifier.

    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    # prepare for mixup 

    # build the data pipeline
    # if isinstance(img, str):
    #     if cfg.data.test.pipeline[0]['type'] != 'LoadImageFromMixupFile':
    #         cfg.data.test.pipeline.insert(0, dict(type='LoadImageFromMixupFile'))
    #     data = dict(img_info=dict(filename=img, filename_target=img_target), img_prefix=None, mixup_info=dict(lam=0.5))
    # else:
    #     raise ValueError(f"Unexcepted branch")
        
    if isinstance(img, str):
        if cfg.data.test.pipeline[0]['type'] != 'LoadImageFromFile':
            cfg.data.test.pipeline.insert(0, dict(type='LoadImageFromFile'))
        data = dict(img_info=dict(filename=img), img_prefix=None)
    else:
        if cfg.data.test.pipeline[0]['type'] == 'LoadImageFromFile':
            cfg.data.test.pipeline.pop(0)
        data = dict(img=img)

    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # forward the model
    with torch.no_grad():
        scores = model(return_loss=False, **data)
        pred_score = np.max(scores, axis=1)[0]
        pred_label = np.argmax(scores, axis=1)[0]
        result = {'pred_label': pred_label, 'pred_score': float(pred_score), 'scores': scores[0]}
    result['pred_class'] = model.CLASSES[result['pred_label']]
    return result


class ForwardResult:
    gt_label_int: int
    pred_label_int: int
    file_path: str
    pred_score: float
    pred_scores: float

    def __init__(self, file_path, gt_label_int, pred_label_int, pred_score, pred_scores):
        self.file_path = file_path
        self.gt_label_int = gt_label_int
        self.pred_label_int = pred_label_int
        self.pred_score = pred_score
        self.pred_scores = pred_scores


def evaluate_for_dataset(model, dataset_root: str, img_target, max_count=999999):

    fr_list = []
    cls_names = os.listdir(dataset_root)
    for cls_name in cls_names:
        cls_dir_path = os.path.join(dataset_root, cls_name)
        cls_dir_file_names = os.listdir(cls_dir_path)
        for cls_dir_file_name in tqdm(cls_dir_file_names):
            img_path = os.path.join(cls_dir_path, cls_dir_file_name)
            # TODO: add img_target correctly
            res = inference_model_for_softmax(model, img_path, img_path)
            # format the res
            pred_label_int = int(res['pred_label'])
            pred_score = float(res['pred_score'])
            pred_scores = res['scores'].tolist()
            fr = ForwardResult("", int(cls_name), pred_label_int, pred_score, pred_scores)
            fr_list.append(fr)
    
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
