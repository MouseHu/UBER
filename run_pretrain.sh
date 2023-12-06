#!/bin/bash

# Script to run pretrain

train_type="r4"
comment="pretrain_metaworld_replay"
#envs=(
#	"halfcheetah-expert-v0"
#  "halfcheetah-medium-v0"
#  "halfcheetah-medium-replay-v0"
#  "halfcheetah-medium-expert-v0"
#)
envs=(
"push-v2"
"reach-v2"
"pick-place-v2"
)
num_seed=1
gpus=(0 6 7)
for ((j=0;j<${#envs[@]};j+=1))
do
  export CUDA_VISIBLE_DEVICES=${gpus[j]}
  env=${envs[j]}
  for ((i=0;i<${num_seed};i+=1))
  do
    nohup python pretrain_metaworld.py \
    --env "${env}" \
    --seed "${i}" \
    --save_model \
    --train_type "${train_type}" \
    --comment "${comment}" \
    --reward_dim 100 > nohup_logs/"${env}"_"${train_type}"_"${comment}"_"${i}".out &
  done
done