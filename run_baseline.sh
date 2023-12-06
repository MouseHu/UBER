
#!/bin/bash

# Script to run finetune

comment="baseline"
envs=(
	"halfcheetah-expert-v0"
  "hopper-expert-v0"
  "walker2d-expert-v0"
  "ant-expert-v0"
)

num_seed=5
gpus=(0 1 2 3)
for ((j=0;j<${#envs[@]};j+=1))
do
  export CUDA_VISIBLE_DEVICES=${gpus[j]}
  env=${envs[j]}
  for ((i=0;i<${num_seed};i+=1))
  do
    nohup python cup.py \
    --env "${env}" \
    --seed "${i}" \
    --comment "${comment}" \
    --seed $i > nohup_logs/"${env}"_"${comment}"_"${i}".out &
  done
done