CUDA_VISIBLE_DEVICES=1 python train_cifar.py \
        --model=MobileNetV2 \
        --resume \
        --resume_path=../checkpoints/mobilenetv2_cifar10_3_128_0.01_3_150_100_p8_sp0.5_all/model_masked_finetune.pth \
        --pretrain_epochs=300 \
        --epochs=1 \
        --iters=3 \
        --fintune_epochs=0 \
        --postfix=connect_test
