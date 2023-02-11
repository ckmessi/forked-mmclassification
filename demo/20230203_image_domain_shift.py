import os
import cv2
import argparse
import random
import torch
import loguru
import numpy as np
from tqdm import tqdm

from mmcls.apis import init_model

from mmcls.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter


from demo.test_time_mixup import test_time_mixup

from loguru import logger


def inference_model_for_softmax(model, imgs, source_train_mixed_img, target_lambda=1.0):
    """
    copy from `20221203_image_mixup_forward_evaluate_for_dataset.py`

    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    assert len(imgs) > 0, f"Unexcepted empty imgs."

    if cfg.data.test.pipeline[0]['type'] == 'LoadImageFromFile':
        cfg.data.test.pipeline.pop(0)
    if cfg.data.test.pipeline[0]['type'] != 'LoadImageFromMixupFile':
        cfg.data.test.pipeline.insert(0, dict(type='LoadImageFromMixupFile'))

    test_pipeline = Compose(cfg.data.test.pipeline)
    test_input_data = [
        test_pipeline(
            dict(img_info=dict(filename=img, source_train_mixed_img=source_train_mixed_img), img_prefix=None, mixup_info=dict(lam=target_lambda))
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
    def __init__(self, file_path, gt_label_int, pred_label_int, pred_score, pred_scores):
        self.file_path = file_path
        self.gt_label_int = gt_label_int
        self.pred_label_int = pred_label_int
        self.pred_score = pred_score
        self.pred_scores = pred_scores

class ForwardResultForMixup:
    def __init__(self, file_path, mixup_lambda, forward_result):
        self.file_path = file_path
        self.mixup_lambda = mixup_lambda
        self.forward_result = forward_result

    def __repr__(self):
        return f"{self.file_path}, {self.mixup_lambda} \n {self.forward_result}"



# 从数据集中读取路径
def read_image_path_list_for_cls_dataset_root(dataset_root: str):
    cls_names = os.listdir(dataset_root)
    image_path_list = []
    for cls_name in cls_names:
        cls_dir_path = os.path.join(dataset_root, cls_name)
        cls_dir_file_names = os.listdir(cls_dir_path)
        for cls_dir_file_name in cls_dir_file_names:
            img_path = os.path.join(cls_dir_path, cls_dir_file_name)
            image_path_list.append(img_path)
    return image_path_list


class ImageInfo:
    image_path: str
    category_int: int
    img: object
    img_loaded: bool
    one_hot_label: object
    dataset_num_classes: int

    def __init__(self, image_path: str, category_int: int, dataset_num_classes: int):
        self.image_path = image_path
        self.category_int = category_int
        self.img = None
        self.img_loaded = False
        self.dataset_num_classes = dataset_num_classes
        self._build_one_hot_label()
    
    def _build_one_hot_label(self):
        self.one_hot_label = test_time_mixup.generate_soft_labels([self.category_int], self.dataset_num_classes)


    def load_image_into_memory(self):
        self.img = cv2.imread(self.image_path)
        if self.img is not None:
            self.img_loaded = True
        else:
            raise ValueError(f"Could not load image from `{self.image_path}`")


# 从数据集中读取{路径+类别}
def read_image_info_list_for_cls_dataset_root(dataset_root: str):
    cls_names = os.listdir(dataset_root)
    dataset_num_classes = len(cls_names)
    image_info_list = []
    for (category_int, cls_name) in enumerate(cls_names):
        cls_dir_path = os.path.join(dataset_root, cls_name)
        cls_dir_file_names = os.listdir(cls_dir_path)
        for cls_dir_file_name in cls_dir_file_names:
            img_path = os.path.join(cls_dir_path, cls_dir_file_name)
            image_info_list.append(ImageInfo(img_path, category_int, dataset_num_classes))
    return image_info_list


# 针对一张图，给定 mixup_lambda 进行混合后，输出结果
def forward_for_images_in_specific_mixup_ratio(model, input_image_path_list: str, source_train_image_info: ImageInfo, target_lambda: float):
    
    source_train_image_info.load_image_into_memory()
    if not source_train_image_info.img_loaded:
        raise ValueError(f"Unexpected unloaded source_train_image")


    pred_results_dict = inference_model_for_softmax(model, input_image_path_list, source_train_image_info.img, target_lambda=target_lambda)

    pred_label_int_list = [int(p) for p in pred_results_dict['pred_label']]
    gt_label_int_list = [-1 for p in pred_results_dict['pred_label']]
    pred_score_list = [float(p) for p in pred_results_dict['pred_score']]
    pred_scores_list = [p for p in pred_results_dict['scores']]
    forward_result_list = [ForwardResult(*x) for x in zip(input_image_path_list, gt_label_int_list, pred_label_int_list, pred_score_list, pred_scores_list)]
    forward_result_mixup_list = [ForwardResultForMixup(input_image_path, target_lambda, fr) for (input_image_path, fr) in zip(input_image_path_list, forward_result_list)]
    return forward_result_mixup_list


def calculate_single_image_different_labmda_loss(forward_result_mixup_list: list, one_hot_label):
    # loss= 𝛼_2∗𝑟_(𝑡1𝑎1−𝛼_1 )−𝛼_1∗𝑟_(𝑡1𝑎1−𝛼_2 )−(𝛼_2−𝛼_1)[𝑒〖(𝑐_1 )]〗^T
    assert len(forward_result_mixup_list) >= 2
    # TODO: 暂时先取前两个
    frm_1 = forward_result_mixup_list[0]
    frm_2 = forward_result_mixup_list[1]

    alpha_1 = frm_1.mixup_lambda
    alpha_2 = frm_2.mixup_lambda

    vector_diff = alpha_2 * frm_1.forward_result.pred_scores \
        - alpha_1 * frm_2.forward_result.pred_scores \
        - (alpha_2 - alpha_1) * one_hot_label

    # 这里 vector_diff 是一个向量，暂时用 norm1 看它的损失值
    loss = torch.norm(torch.tensor(vector_diff), p=1, dim=0)
    
    # logger.debug(f"loss value is: {loss}")
    return loss      




# 针对一个数据集，计算每个样本的目标域损失值
# 2023年02月03日
def calculate_shift_value_for_dataset(model, source_dataset_root, target_dataset_root: str, output_file_path=None):
    
    # 读取所有图片
    logger.info(f'read image path list for target dataset...')
    target_image_path_list = read_image_path_list_for_cls_dataset_root(target_dataset_root)
    logger.info(f'read image path list for target dataset finished.')

    # 读取源域图片，先假定读一部分进来
    logger.info(f'read image info list for source dataset...')
    source_train_image_infos = read_image_info_list_for_cls_dataset_root(source_dataset_root)
    SOURCE_TRAIN_IMAGE_SAMPLE_COUNT = 1000
    source_train_image_infos = source_train_image_infos[0: SOURCE_TRAIN_IMAGE_SAMPLE_COUNT]
    logger.info(f'read image info list for source dataset finished.')



    # Loss1：针对两个 mixup_ratio 的一致性
    TARGET_LAMBDA_RATIO_1 = 0.9
    TARGET_LAMBDA_RATIO_2 = 0.8
    # 推理
    forward_result_mixup_dict = {}
    def add_forward_result_mixup_to_dict(forward_result_mixup_dict, forward_result_mixup_list):
        # 按照文件名组织字典
        for fr_for_mixup in forward_result_mixup_list:
            file_path = fr_for_mixup.file_path
            if file_path not in forward_result_mixup_dict:
                forward_result_mixup_dict[file_path] = {}
            forward_result_mixup_dict[file_path][fr_for_mixup.mixup_lambda] = fr_for_mixup
        return forward_result_mixup_dict

    BATCH_SIZE = 256
    for idx in tqdm(range(0, len(target_image_path_list), BATCH_SIZE)):
        current_batch = target_image_path_list[idx : idx+BATCH_SIZE]
        source_train_image_info = random.choice(source_train_image_infos)
        # 第一次
        forward_result_mixup_list_1 = forward_for_images_in_specific_mixup_ratio(model, current_batch, source_train_image_info, TARGET_LAMBDA_RATIO_1)
        # 第二次
        forward_result_mixup_list_2 = forward_for_images_in_specific_mixup_ratio(model, current_batch, source_train_image_info, TARGET_LAMBDA_RATIO_2)
        forward_result_mixup_dict = add_forward_result_mixup_to_dict(forward_result_mixup_dict, forward_result_mixup_list_1)
        forward_result_mixup_dict = add_forward_result_mixup_to_dict(forward_result_mixup_dict, forward_result_mixup_list_2)
    
    # 计算 loss
    loss_list = []
    for file_path, frm_d in forward_result_mixup_dict.items():
        frm_list = list(frm_d.values())
        loss = calculate_single_image_different_labmda_loss(frm_list, source_train_image_info.one_hot_label)
        loss_list.append(loss)
    loss_average = sum(loss_list) / len(loss_list) if len(loss_list) > 0 else 0
    

    logger.info(f"Average loss value is {loss_average}")

    # print(f"average avg_kl_div: {sum(avg_kl_div_list) / len(avg_kl_div_list)}")
    
    # 写文件
    # def save_avg_kl_div_list_to_json_file(avg_kl_div_list, output_file_path = 'tmp/avg_kl_div_list.json'):
    #     avg_kl_div_list_rounded = [round(v, 6) for v in avg_kl_div_list]
    #     with open(output_file_path, 'w') as f:
    #         json.dump({'avg_kl_div_list': avg_kl_div_list_rounded}, f)
    # if output_file_path:
    #     save_avg_kl_div_list_to_json_file(avg_kl_div_list, output_file_path)

    # return avg_kl_div_list



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('target_dataset_root', help='Target Dataset Root')
    parser.add_argument('source_dataset_root', help='Source Dataset Root')
    parser.add_argument('--command', help='', choices=['t20230203'])
    parser.add_argument('--output_file_path', default=None, type=str, help="the file to save")
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    if args.command == 't20230203':
        calculate_shift_value_for_dataset(model, args.source_dataset_root, args.target_dataset_root)
    else:
        raise ValueError(f"Unexpected command: {args.command}")

if __name__ == '__main__':
    main()
