# CAMA
Code for ICLR2023 submission: Consciousness-Aware Multi-Agent Reinforcement Learning.

This codebase is built on top of the [PyMARL](https://github.com/oxwhirl/pymarl) framework and the codebase of [REFIL](https://github.com/shariqiqbal2810/REFIL) algorithm. Thanks for [Shariq Iqbal](https://github.com/shariqiqbal2810) for sharing his code.

## Setup instructions

Please follow the instructions in [REFIL](https://github.com/shariqiqbal2810/REFIL) codebase. Note: If you want to run environment `sc2custom`, an empty map  needs to be copied to the SC2 directory. Note: The particle environment uses gym==0.10.5.

## Run an experiment 

Run an `ALGORITHM` from the folder `src/config/algs`
in an `ENVIRONMENT` from the folder `src/config/envs`

```shell
export CUDA_VISIBLE_DEVICES="0" && python src/main.py --env-config=<ENVIRONMENT> --config=<ALGORITHM> with <PARAMETERS>
```

Possible environments are:
- `particle`: Resource collection.
- `sc2custom`: StarCraft.
- `traffic_junction`: Traffic Junction.
- `gridworld`: The demo "Catch Apple" in the paper.

## Command examples

  Run CAMA with Resource collection:

```shell
export CUDA_VISIBLE_DEVICES="0" && python CAMA/main.py --env-config=particle --config=cama_qmix_atten with test_unseen=False
```

Run CAMA with sc2:

```shell
export CUDA_VISIBLE_DEVICES="0" && python CAMA/main.py --env-config=sc2custom --config=cama_refil with test_unseen=False scenario=3-8sz_symmetric
```

## Parameter Setting
The parameter names in the article correspond to the parameter settings in the code as follows:
- `Common`:
  - $\alpha$: rank_percent
  - $\beta$: beta
  - weight of $\mathcal{L}_{IM}$: ce_weight
  - weight of $\mathcal{L}_{MI}$: club_weight
- `Resource Collection`:
  - sight range: env_args.sight_range_kind (a dictionary of \{sight_range_kind:true sight range in maps\}: \{0:0.2, 1:0.5, 2:1.0, 3:0.8, 4:$\infty$, 5:1.5, 6:2.0\})
- `SC2`:
  - sight range: env_args.sight_range
- `Traffic Junction`:
  - sight range: env_args.vision (0 for 0, 1 for 3*3 grids)
  