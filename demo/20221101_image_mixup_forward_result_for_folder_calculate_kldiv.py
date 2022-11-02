# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmcls.apis import init_model, show_result_pyplot
import scipy
import os
import json
from tqdm import tqdm

import numpy as np
import torch
from mmcls.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
from mmcls.models.utils.augment import BatchMixupLayer

from collections import defaultdict

def inference_model_for_softmax(model, img, img_src, lam=0.5):
    """Inference image(s) with the classifier.

    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    # prepare for mixup 

    # build the data pipeline
    if isinstance(img, str):
        if cfg.data.test.pipeline[0]['type'] != 'LoadImageFromMixupFile':
            cfg.data.test.pipeline.insert(0, dict(type='LoadImageFromMixupFile'))
        data = dict(img_info=dict(filename=img, filename_target=img_src), img_prefix=None, mixup_info=dict(lam=lam))
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

def inference_for_folder(model, img_dir: str, img_src, save_file_path: str, max_count=1000):
    img_names = os.listdir(img_dir)
    dataset_kl_div_v_info_dict = defaultdict(list)
    
    for img_name in tqdm(img_names):
        img_path = os.path.join(img_dir, img_name)
        # 每一张图，算N个内容
        # 先算两个样本
        res_source = inference_model_for_softmax(model, img_path, img_src, lam=0.0)
        res_target = inference_model_for_softmax(model, img_path, img_src, lam=1.0)
        scores_source = res_source['scores']
        scores_target = res_target['scores']


        # 算中间的内容
        kl_div_v_max = 0
        kl_div_v_min = 99999
        kl_div_v_list = []
        for cur_lam in [0.1, 0.3, 0.5, 0.7, 0.9]:
            current_res = inference_model_for_softmax(model, img_path, img_src, lam=cur_lam)
            cur_scores_gt = scores_source * cur_lam + scores_target * (1-cur_lam)
            # print(f"cur_scores_gt: {cur_scores_gt}")
            # print(f"current_res: {current_res['scores']}")
            kl_div_v = float(scipy.stats.entropy(cur_scores_gt, current_res['scores']))
            # print(f"kl_div_v: {kl_div_v}")
            kl_div_v_max = max(kl_div_v_max, kl_div_v)
            kl_div_v_min = min(kl_div_v_min, kl_div_v)
            kl_div_v_list.append(kl_div_v)
        
        kl_div_v_avg = sum(kl_div_v_list) / len(kl_div_v_list)
        # print(f"max: {kl_div_v_max}")
        # print(f"min: {kl_div_v_min}")
        # print(f"average: {kl_div_v_avg}")
        dataset_kl_div_v_info_dict['min'].append(kl_div_v_min)
        dataset_kl_div_v_info_dict['max'].append(kl_div_v_max)
        dataset_kl_div_v_info_dict['avg'].append(kl_div_v_avg)
        if len(dataset_kl_div_v_info_dict['avg']) >= max_count:
            break

     
    print(f"max: {sum(dataset_kl_div_v_info_dict['max']) / len(dataset_kl_div_v_info_dict['max'])}")
    print(f"min: {sum(dataset_kl_div_v_info_dict['min']) / len(dataset_kl_div_v_info_dict['min'])}")
    print(f"average: {sum(dataset_kl_div_v_info_dict['avg']) / len(dataset_kl_div_v_info_dict['avg'])}")

    with open(save_file_path, 'w') as f:
        json.dump(dataset_kl_div_v_info_dict, f)

def main():
    parser = ArgumentParser()
    parser.add_argument('img_dir', help='Image Dir')
    parser.add_argument('img_src', help='Target Image file')
    parser.add_argument('save_file_path', help='')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    # test a single image
    inference_for_folder(model, args.img_dir, args.img_src, args.save_file_path)


if __name__ == '__main__':
    main()
