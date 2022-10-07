#!/bin/sh

for model in net5 net2 net3 net4 net1
do
for loss_type in mse mae
  do
    for fold in 0 1 2 3 4
    do
    CUDA_VISIBLE_DEVICES=2 python3 runner.py --model $model --epochs 50 --fold $fold --batch_size 64 --scheduler_patience 8 --lr 3e-4 --amsgrad --weight_decay 3e-5 --loss_type $loss_type
    done
  done
done