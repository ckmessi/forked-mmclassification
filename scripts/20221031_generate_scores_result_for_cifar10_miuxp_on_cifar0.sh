# 输出领域外数据集的推理结果
for CLASS_INDEX in 0 1 2 3 4 5 6 7 8 9
do
    echo "forward for class $CLASS_INDEX"
    python demo/image_mixup_forward_result_for_folder.py data/cifar10/test_split/$CLASS_INDEX data/cifar10/test_split/0/1001_airplane.png tmp/cifar10-test-mixup-0/cifar10-$CLASS_INDEX-test-mixup-0.json configs/resnet/exp_20221009/20221009_resnet50_8xb16-mixup_cifar10.py work_dirs/20221009_resnet50_8xb16-mixup_cifar10/epoch_200.pth
done

