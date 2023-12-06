
#!/bin/bash

# Script to run finetune

train_type="r4"
comment="finetune_metaworld_replay"
envs=(
	"shelf-place-v2"
	"pick-place-wall-v2"
	"push-wall-v2"
	"push-back-v2"
)
#envs=(
#"hammer-v2"
#)
num_seed=5
gpus=(0 1 2 3)
#for ((j=0;j<${#envs[@]};j+=1))
#do
#  export CUDA_VISIBLE_DEVICES=${gpus[j]}
#  env=${envs[j]}
#  for ((i=0;i<${num_seed};i+=1))
#  do
#    filename1="./TD3_BC_reach-v2_"${train_type}"_pretrain_metaworld_0"
#    filename2="./TD3_BC_push-v2_"${train_type}"_pretrain_metaworld_0"
#    filename3="./TD3_BC_pick-place-v2_"${train_type}"_pretrain_metaworld_0"
#    nohup python pex_multitask.py \
#    --env "${env}" \
#    --seed "${i}" \
#    --comment "${train_type}"_"${comment}" \
#    --load_guidance "${filename1}" "${filename2}" "${filename3}"\
#    --reward_dim 100 \
#    --alpha 1 \
#    --seed $i > nohup_logs/"${env}"_"${train_type}"_"${comment}"_"${i}".out &
#  done
#done

for ((j=0;j<${#envs[@]};j+=1))
do
  export CUDA_VISIBLE_DEVICES=${gpus[j]}
  env=${envs[j]}
  for ((i=0;i<${num_seed};i+=1))
  do
#    filename1="./yyq/TD3_BC_reach-v2_"${train_type}"_r4_pretrain_0"
#    filename2="./yyq/TD3_BC_push-v2_"${train_type}"_r4_pretrain_0"
#    filename3="./yyq/TD3_BC_pick-place-v2_"${train_type}"_r4_pretrain_0"
    filename1="./TD3_BC_reach-v2_"${train_type}"_pretrain_metaworld_replay_0"
    filename2="./TD3_BC_push-v2_"${train_type}"_pretrain_metaworld_0"
    filename3="./TD3_BC_pick-place-v2_"${train_type}"_pretrain_metaworld_0"
    nohup python pex_multitask.py \
    --env "${env}" \
    --seed "${i}" \
    --comment "${train_type}"_"${comment}" \
    --reward_dim 100 \
    --load_guidance "${filename1}" "${filename2}" "${filename3}"\
    --seed $i > nohup_logs/"${env}"_"${train_type}"_"${comment}"_"${i}".out &
  done
done