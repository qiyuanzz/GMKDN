#!/bin/bash

lr=0.0002 
temp=4           
seeds=(1 2 3 4 5) 
gc=32
max_epochs=100
top=8192

for seed in "${seeds[@]}"; do     # 
    # 输出当前正在训练的参数组合
    echo "Training with learning_rate=$lr, gc=$gc, temperature=$temp,  max_epochs=$max_epochs, seed=$seed"

    # 执行训练阶段
    python train_kd.py --split_dir "ER" --lr $lr --gc $gc --temp $temp --stage "train" --max_epochs $max_epochs --seed $seed --weighted_sample --early_stopping 
    python train_kd.py --split_dir "ER" --lr $lr --gc $gc --temp $temp --stage "test" --max_epochs $max_epochs --seed $seed --weighted_sample --early_stopping -

done
