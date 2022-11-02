# 输出领域外数据集的推理结果
for CLASS_INDEX in 0 1 2 3 4 5 6 7 8 9
do
    echo "forward for class $CLASS_INDEX"
    python demo/20221101_image_mixup_forward_result_for_folder_calculate_kldiv.py data/cifar10/test_split/$CLASS_INDEX data/cifar10/test_split/0/1001_airplane.png tmp/20221102_kldiv_cifar10/20221102_kldiv_cifar10_"$CLASS_INDEX"_target_0.json configs/resnet/exp_20221009/20221009_resnet50_8xb16-mixup_cifar10.py work_dirs/20221009_resnet50_8xb16-mixup_cifar10/epoch_200.pth
done

