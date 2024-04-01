# The Winning Solution for the NeurIPS 2023 Neural MMO Challenge

This solution is based on the [Neural MMO Baselines](https://github.com/NeuralMMO/baselines/tree/2.0?tab=readme-ov-file). For more information about the challenge, please refer to the [challenge homepage](https://www.aicrowd.com/challenges/neurips-2023-the-neural-mmo-challenge).

## How to Use

### Environment Installation
Use `docker/build.sh` to build image for training.
```shell
cd docker
bash build.sh
```

### Run training
Run inside the training container:
```shell
export WANDB_API_KEY=xxx  # Change it to yours
WANDB_PROJECT=xxx  # Change it to yours
WANDB_ENTITY=xxx  # Change it to yours

export WANDB_DISABLE_GIT=true
export WANDB_DISABLE_CODE=true

export OMP_NUM_THREADS=4

python train.py \
       --runs-dir runs \
       --use-ray-vecenv true \
       --wandb-project $WANDB_PROJECT \
       --wandb-entity $WANDB_ENTITY \
       --model ReducedModelV2 \
       --meander-bonus-weight 0.0 \
       --heal-bonus-weight 0.0 \
       --num-npcs 128 \
       --early-stop-agent-num 0 \
       --resilient-population 0.0 \
       --ppo-update-epochs 1 \
       --train-num-steps 40000000 \
       --num-maps 1280 \
```

### Evaluation

After training, copy the checkpoints into `policies` and run:
```shell
python evaluate.py -p policies
```
`policies/submission.pkl` is the trained model we submitted.
