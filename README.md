# Unsupervised Behavior Extraction via Random Intent Priors

Code for the paper [Unsupervised Behavior Extraction via Random Intent Priors](https://arxiv.org/abs/2106.01969). 

Paper accept at Thirty-seventh Conference on Neural Information Processing Systems (**NeurIPS**), 2023

Authors: [Hao Hu*](https://mousehu.github.io/), [Yiqin Yang*](https://openreview.net/profile?id=~Yiqin_Yang1), [Jianing Ye](https://openreview.net/profile?id=~Jianing_Ye1), [Ziqing Mai](https://openreview.net/profile?id=~Ziqing_Mai1),[Chongjie Zhang](https://engineering.wustl.edu/faculty/Chongjie-Zhang.html)

## Installation

```
git clone https://github.com/MouseHu/UBER.git
pip install -r requirements.txt
```


## Training

### Stage 1: Pretrain

```shell
python pretrain.py \
    --env "${env}" \
    --seed "${i}" \
    --save_model \
    --train_type r4 \
    --comment r4_pretrain
```

```shell
python pretrain_metaworld.py \
    --env "${env}" \
    --seed "${i}" \
    --save_model \
    --train_type r4 \
    --comment r4_pretrain
```

### Stage 2: Reuse

```shell
CUDA_VISIBLE_DEVICES=0 python cup.py --comment from_r4_random_reproduce --env halfcheetah-random-v0 --seed 0 --load_guidance ./TD3_BC_halfcheetah-random-v0_r4_reproduce_0
```

```shell
CUDA_VISIBLE_DEVICES=0 python pex.py --comment from_r4_medium_reproduce --env halfcheetah-medium-v0 --seed 0 --load_guidance ./TD3_BC_halfcheetah-medium-v0_r4_reproduce_0
```

```shell
CUDA_VISIBLE_DEVICES=0 python pex.py --comment from_r4_medium_reproduce --env halfcheetah-medium-v0 --seed 0 --load_guidance ./TD3_BC_halfcheetah-medium-v0_r4_reproduce_0
```

```shell
CUDA_VISIBLE_DEVICES=0 python pex_multitask.py --comment test --env hammer-v2 --seed 0 --load_guidance TD3_BC_pick-place-v2_r4_pretrain_metaworld_0 TD3_BC_reach-v2_r4_pretrain_metaworld_0 TD3_BC_push-v2_r4_pretrain_metaworld_0
```


### Evaluation:

```shell
CUDA_VISIBLE_DEVICES=1 python multihead_evaluation.py --load_actor TD3_BC_halfcheetah-medium-v0_r4_reproduce_0 --comment test_medium --env halfcheetah-medium-v0 --reward_dim 256

```



