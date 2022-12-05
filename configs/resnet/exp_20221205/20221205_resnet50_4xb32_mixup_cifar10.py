_base_ = [
    '../../_base_/models/resnet50_cifar.py', '../../_base_/datasets/cifar10_bs16_plain.py',
    '../../_base_/schedules/cifar10_bs128.py', '../../_base_/default_runtime.py'
]

data = dict(
    samples_per_gpu=32,
)
