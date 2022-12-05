from cmath import inf
from typing import List
import numpy as np
import pytest
import scipy
import math


class ImageInfo:
    img: object
    label: int
    num_classes: int

    def __init__(self, img, label: int, num_classes: int) -> None:
        self.img = img
        self.label = label
        self.num_classes = num_classes

def compare_array_nearly_equal(np_arr1, np_arr2):
    if np_arr1.shape[0] != np_arr2.shape[0]:
        return False
    for i in range(0, np_arr1.shape[0]):
        if not math.isclose(np_arr1[i], np_arr2[i], rel_tol=1e-6):
            return False
    return True


def test_compare_array_nearly_equal():
    np_arr1 = np.array([])
    np_arr2 = np.array([])
    assert compare_array_nearly_equal(np_arr1, np_arr2) == True

    np_arr1 = np.array([1])
    np_arr2 = np.array([])
    assert compare_array_nearly_equal(np_arr1, np_arr2) == False

    np_arr1 = np.array([1])
    np_arr2 = np.array([2])
    assert compare_array_nearly_equal(np_arr1, np_arr2) == False
    
    np_arr1 = np.array([1])
    np_arr2 = np.array([1])
    assert compare_array_nearly_equal(np_arr1, np_arr2) == True

    np_arr1 = np.array([1, 2])
    np_arr2 = np.array([1, 3])
    assert compare_array_nearly_equal(np_arr1, np_arr2) == False

    np_arr1 = np.array([1, 2])
    np_arr2 = np.array([1, 2])
    assert compare_array_nearly_equal(np_arr1, np_arr2) == True


def forward_for_scores(input_img):
    """
    给定图像，
    输出分类的分数值 scores
    """
    # TODO: add logic about model forward
    return [1.0, 0.0, 0.0]


def test_forward_for_scores(input_img):
    raise NotImplementedError(f'This test not implemented')


def generate_soft_labels(labels_list, num_classes):
    assert len(labels_list) > 0, f"Invalid length of labels_list, expected > 0, 0 got."
    soft_labels = np.zeros(num_classes)
    for label in labels_list:
        soft_labels[label] += 1.0
    soft_labels /= len(labels_list)
    return soft_labels


def test_generate_soft_labels():
    with pytest.raises(AssertionError, match="Invalid length"):
        generate_soft_labels([], 10)
    
    soft_labels = generate_soft_labels([1, 2, 3, 4], 10)
    assert (soft_labels == [0, 0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0, 0]).all()
    
    soft_labels = generate_soft_labels([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10)
    assert (soft_labels == [0.1] * 10).all()
    soft_labels = generate_soft_labels([0, 1, 2, 3], 4)
    assert (soft_labels == [0.25] * 4).all()


def mixup_image_list():
    pass

def mixup_two_images(img1, img2, img2_ratio=0.3):
    pass


def forward_for_mixup_scores(input_img, train_images_info: List[ImageInfo], source_ratio=0.3):
    """
    给定 {待测图像} 和 {训练图像+标签}，
    输出分类的分数值
    """
    # TODO:

    # 图像的混合
    
    # 推理



def calculate_recovered_scores(train_image_infos, scores1, num_classes=5, source_ratio=0.3):
    """
    给定 {待测图像} 和 {训练图像+标签}，以及获得的输出值
    返回正常的 scores0'（即 input_img 所隐含的成份） 

    input_img -> scores0 = {label_X, [0, 0, ..., 1, ..., 0]} = {[0, 0, ..., 1, ..., 0]（第i个label为1，其它为0）, [0, 0, ..., 1, ..., 0]（推理结果）}
    train_images_info -> scores_s = {{label_0, label_1, ..., label_i, ..., label_c}, scores_s}

    """
    if source_ratio == 0.0:
        return scores1
    
    if source_ratio >= 1.0:
        raise ValueError(f"source_ratio is larger than 1.0, expected 0.0 <= source_ratio <= 1.0, {source_ratio} got.")
    
    """
    (1 - source_ratio) * scores0' + source_ratio * train_soft_label = scores1
    ->
    scores0' = (scores1 - source_ratio * train_soft_label) / (1-source_ratio)
    """
    train_soft_label = generate_soft_labels_for_image_infos(train_image_infos, num_classes)
    recovered_scores = (scores1 - source_ratio * train_soft_label) / (1 - source_ratio)
    return recovered_scores


def test_calculate_recovered_scores():
    # for extereme info
    train_images_info = [ImageInfo(None, 1, 5)]
    scores1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    recovered_scores = calculate_recovered_scores(train_images_info, scores1, 5, 0.0)
    assert (recovered_scores == scores1).all()
    
    train_images_info = [ImageInfo(None, 1, 5)]
    scores1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match='source_ratio is larger than 1.0'):
        recovered_scores = calculate_recovered_scores(train_images_info, scores1, 5, 1.0)

    # normal scene
    train_images_info = [ImageInfo(None, 1, 5)]
    scores1 = np.array([0.5, 0.5, 0.0, 0.0, 0.0])
    recovered_scores = calculate_recovered_scores(train_images_info, scores1, 5, 0.5)
    assert (recovered_scores == np.array([1.0, 0.0, 0.0, 0.0, 0.0])).all()
    
    train_images_info = [ImageInfo(None, 1, 5)]
    scores1 = np.array([0.2, 0.8, 0.0, 0.0, 0.0])
    recovered_scores = calculate_recovered_scores(train_images_info, scores1, 5, 0.8)
    print(recovered_scores)
    assert compare_array_nearly_equal(recovered_scores, np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    
    train_images_info = [ImageInfo(None, 1, 5)]
    scores1 = np.array([0.8, 0.2, 0.0, 0.0, 0.0])
    recovered_scores = calculate_recovered_scores(train_images_info, scores1, 5, 0.2)
    assert compare_array_nearly_equal(recovered_scores, np.array([1.0, 0.0, 0.0, 0.0, 0.0]))


def generate_soft_labels_for_image_infos(train_image_infos, num_classes=1):
    image_count = len(train_image_infos)
    if image_count == 0:
        return np.zeros(num_classes)

    soft_labels = np.zeros(num_classes)
    for image_info in train_image_infos:
        soft_labels[image_info.label] += 1.0
    
    soft_labels /= image_count
    return soft_labels


def test_generate_soft_labels_for_image_infos():
    # empty scenario
    soft_labels = generate_soft_labels_for_image_infos([], 1)
    assert soft_labels == [0.0]
    soft_labels = generate_soft_labels_for_image_infos([], 3)
    assert (soft_labels == [0.0, 0.0, 0.0]).all()
    soft_labels = generate_soft_labels_for_image_infos([], 10)
    assert (soft_labels == np.zeros(10)).all()

    # normal
    train_image_infos = [
        ImageInfo(None, 1, 5),
    ]
    soft_labels = generate_soft_labels_for_image_infos(train_image_infos, 5)

    assert (soft_labels == [0.0, 1, 0.0, 0.0, 0.0]).all()
    train_image_infos = [
        ImageInfo(None, 1, 5),
        ImageInfo(None, 2, 5),
        ImageInfo(None, 3, 5),
        ImageInfo(None, 4, 5),
    ]
    soft_labels = generate_soft_labels_for_image_infos(train_image_infos, 5)
    assert (soft_labels == [0.0, 0.25, 0.25, 0.25, 0.25]).all()

    train_image_infos = [
        ImageInfo(None, 1, 10),
        ImageInfo(None, 1, 10),
        ImageInfo(None, 3, 10),
        ImageInfo(None, 4, 10),
        ImageInfo(None, 5, 10),
    ]
    soft_labels = generate_soft_labels_for_image_infos(train_image_infos, 10)
    assert (soft_labels == [0.0, 0.4, 0.0, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0]).all()
    
    train_image_infos = [
        ImageInfo(None, 1, 10),
        ImageInfo(None, 1, 10),
        ImageInfo(None, 1, 10),
        ImageInfo(None, 1, 10),
        ImageInfo(None, 1, 10),
    ]
    soft_labels = generate_soft_labels_for_image_infos(train_image_infos, 10)
    assert (soft_labels == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).all()




def calculate_kl_div(scores0, scores0_recovered):
    """
    计算 scores0 和 scores0_recovered 的一致程度
    """
    kl_div_value = float(scipy.stats.entropy(scores0, scores0_recovered))
    return kl_div_value


def test_calculate_kl_div():
    scores0 = np.array([1.0, 0.0, 0.0])
    scores0_recovered = np.array([1.0, 0.0, 0.0])
    assert calculate_kl_div(scores0, scores0_recovered) == 0.0

    scores0 = np.array([1.0, 0.0, 0.0])
    scores0_recovered = np.array([0.9, 0.1, 0.1])
    assert math.isclose(calculate_kl_div(scores0, scores0_recovered), 0.2006706, rel_tol=1e-6)

    scores0 = np.array([1.0, 0.0, 0.0])
    scores0_recovered = np.array([1.0, 0.0, 1.0])
    kl_div = calculate_kl_div(scores0, scores0_recovered)
    assert math.isclose(kl_div, 0.693147, rel_tol=1e-6)

    # totally different
    scores0 = np.array([1.0, 0.0, 0.0])
    scores0_recovered = np.array([0.0, 0.0, 1.0])
    kl_div = calculate_kl_div(scores0, scores0_recovered)
    assert math.isclose(kl_div, inf, rel_tol=1e-6)
    scores0 = np.array([1.0, 0.0, 0.0])
    scores0_recovered = np.array([0.0, 1.0, 0.0])
    kl_div = calculate_kl_div(scores0, scores0_recovered)
    assert math.isclose(kl_div, inf, rel_tol=1e-6)

    # more consistent
    scores0 = np.array([1.0, 0.0, 0.0])
    scores0_recovered = np.array([0.8, 0.1, 0.1])
    kl_div = calculate_kl_div(scores0, scores0_recovered)
    assert math.isclose(kl_div, 0.2231435, rel_tol=1e-6)
    scores0 = np.array([1.0, 0.0, 0.0])
    scores0_recovered = np.array([0.5, 0.25, 0.25])
    kl_div = calculate_kl_div(scores0, scores0_recovered)
    assert math.isclose(kl_div, 0.6931472, rel_tol=1e-6)


def calculate_compositive_scores(scores0, scores0_recovered, lambda_value):
    """
    根据原始的 scores0 和计算得到的 scores0_recovered，计算综合得分。
    当 `lambda_value` 为 `1` 时，退化为原精度

    s = lambda_value * scores0 + (1-lambda_value) * (1-kl_div) * scores0_recovered
    """
    kl_div = calculate_kl_div(scores0, scores0_recovered)
    compositive_scores = lambda_value * scores0 + (1 - lambda_value) * (1 - kl_div) * scores0_recovered
    return compositive_scores


def test_calculate_compositive_scores():
    scores0 = np.array([1.0, 0.0, 0.0])
    scores0_recovered = np.array([0.9, 0.1, 0.1])
    compositive_scores = calculate_compositive_scores(scores0, scores0_recovered, 1.0)
    assert compare_array_nearly_equal(compositive_scores, scores0) == True

