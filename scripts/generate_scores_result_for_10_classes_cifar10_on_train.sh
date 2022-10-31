for CLASS_INDEX in 0 1 2 3 4 5 6 7 8 9
do
    echo "forward for class $CLASS_INDEX"
    python demo/image_forward_result_for_folder.py /data/chenkai/datasets/public/cifar10/train_split/$CLASS_INDEX tmp/cifar-10-on-train/cifar-test-"$CLASS_INDEX".json configs/resnet/exp_20221017/20221017_resnet50_8xb16-cifar10.py work_dirs/20221017_resnet50_8xb16-cifar10/epoch_200.pth
done

