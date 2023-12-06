
#!/bin/bash

# Script to run finetune

train_type="avg"
comment="finetune_long"
envs=(
	"halfcheetah-expert-v0"
  "halfcheetah-medium-v0"
  "halfcheetah-medium-expert-v0"
  "halfcheetah-medium-replay-v0"
)

num_seed=5
gpus=(4 5 6 7)
for ((j=0;j<${#envs[@]};j+=1))
do
  export CUDA_VISIBLE_DEVICES=${gpus[j]}
  env=${envs[j]}
  for ((i=0;i<${num_seed};i+=1))
  do
    filename="./TD3_BC_"${env}"_"${train_type}"_pretrain_"${i}
    nohup python pex.py \
    --env "${env}" \
    --seed "${i}" \
    --comment "${train_type}"_"${comment}" \
    --load_guidance "${filename}" \
    --reward_dim 1 \
    --seed $i > nohup_logs/"${env}"_"${train_type}"_"${comment}"_"${i}".out &
  done
done