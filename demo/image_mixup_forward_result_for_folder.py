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
    if isinstance(img, str):
        if cfg.data.test.pipeline[0]['type'] != 'LoadImageFromMixupFile':
            cfg.data.test.pipeline.insert(0, dict(type='LoadImageFromMixupFile'))
        data = dict(img_info=dict(filename=img, filename_target=img_target), img_prefix=None, mixup_info=dict(lam=0.5))
    else:
        raise ValueError(f"Unexcepted branch")

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

def inference_for_folder(model, img_dir: str, img_target, save_file_path: str, max_count=1000):
    img_names = os.listdir(img_dir)
    result_list = []
    for img_name in tqdm(img_names):
        img_path = os.path.join(img_dir, img_name)
        res = inference_model_for_softmax(model, img_path, img_target)
        res['file_path'] = img_path
        # format the res
        res['pred_label'] = int(res['pred_label'])
        res['pred_score'] = float(res['pred_score'])
        res['scores'] = res['scores'].tolist()
        result_list.append(res)
        if len(result_list) >= max_count:
            break
    
    with open(save_file_path, 'w') as f:
        json.dump(result_list, f)

def main():
    parser = ArgumentParser()
    parser.add_argument('img_dir', help='Image Dir')
    parser.add_argument('img_target', help='Target Image file')
    parser.add_argument('save_file_path', help='')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    # test a single image
    inference_for_folder(model, args.img_dir, args.img_target, args.save_file_path)


if __name__ == '__main__':
    main()
