CUDA_VISIBLE_DEVICES=0 python train_cifar_pipeline.py  \
        --resume \
        --resume_path=../checkpoints/resnet20_cifar10_3_128_0.01_1/model_trained_stage1.pth \
        --epochs=3 \
        --iters=50 \
        --fintune_epochs=100 \
        --postfix=connect_pattern_f_3_50_100_t3_p9