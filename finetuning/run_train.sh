#!/bin/bash
DATE=$(date "+%m%d%y_%H_%M_%S")

module load load cuda/11.7.0 cudnn/v8.8.0
conda activate miniconda3/envs/pyone_project/
nvidia-smi



# accelerate launch train.py \
#     --output_dir=checkpoints2/ \
#     --resume_from_checkpoint=/fs/nexus-scratch/kcobbina/Pyone_Project/checkpoints/epoch_2 \
#     --dataset=data/crawl-data-2022-0.csv \
#     --num_train_epochs=6



accelerate launch train.py \
    --output_dir=checkpoints3/ \
    --resume_from_checkpoint=checkpoints2/epoch_5 \
    --dataset=data/crawl-data-2023-0.csv \
    --num_train_epochs=9



# python final.py \
#     --dataset=data/crawl-data-2021-0.csv
# python train.py
# python train.py
# python new_train.py

# python accel.py

# # sbatch run.sh run_train.sh 
