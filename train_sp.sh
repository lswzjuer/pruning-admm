CUDA_VISIBLE_DEVICES=1 python train_cifar.py \
        --resume \
        --resume_path=../checkpoints/resnet20_cifar10_3_128_0.01_1/model_trained_stage1.pth \
        --epochs=3 \
        --iters=50 \
        --fintune_epochs=100 \
        --pattern_num=8 \
        --sp=0.2 \
        --postfix=all 
CUDA_VISIBLE_DEVICES=1 python train_cifar.py \
        --resume \
        --resume_path=../checkpoints/resnet20_cifar10_3_128_0.01_1/model_trained_stage1.pth \
        --epochs=3 \
        --iters=50 \
        --fintune_epochs=100 \
        --pattern_num=8 \
        --sp=0.3 \
        --postfix=all 
CUDA_VISIBLE_DEVICES=1 python train_cifar.py \
        --resume \
        --resume_path=../checkpoints/resnet20_cifar10_3_128_0.01_1/model_trained_stage1.pth \
        --epochs=3 \
        --iters=50 \
        --fintune_epochs=100 \
        --pattern_num=8 \
        --sp=0.4 \
        --postfix=all 
CUDA_VISIBLE_DEVICES=1 python train_cifar.py \
        --resume \
        --resume_path=../checkpoints/resnet20_cifar10_3_128_0.01_1/model_trained_stage1.pth \
        --epochs=3 \
        --iters=50 \
        --fintune_epochs=100 \
        --pattern_num=8 \
        --sp=0.6 \
        --postfix=all 