
#!/bin/bash

# Script to run finetune

train_type="r4"
comment="finetune_cup"
envs=(
	"walker2d-expert-v0"
  "walker2d-medium-v0"
  "walker2d-medium-expert-v0"
  "walker2d-medium-replay-v0"
)

num_seed=5
gpus=(4 5 6 7)
for ((j=0;j<${#envs[@]};j+=1))
do
  export CUDA_VISIBLE_DEVICES=${gpus[j]}
  env=${envs[j]}
  for ((i=0;i<${num_seed};i+=1))
  do
    filename="./TD3_BC_"${env}"_"${train_type}"_pretrain_0"
    nohup python cup.py \
    --env "${env}" \
    --seed "${i}" \
    --comment "${train_type}"_"${comment}" \
    --load_guidance "${filename}" \
    --reward_dim 256 \
    --seed $i > nohup_logs/"${env}"_"${train_type}"_"${comment}"_"${i}".out &
  done
done