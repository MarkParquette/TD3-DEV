#!/bin/bash

# Script to reproduce results

for ((i=0;i<1;i+=1))
do 
	python main.py \
	--dev \
	--policy "TD3" \
	--env "HalfCheetah-v5" \
	--seed $i

	python main.py \
	--dev \
	--policy "TD3" \
	--env "Hopper-v5" \
	--seed $i

	python main.py \
	--dev \
	--policy "TD3" \
	--env "Walker2d-v5" \
	--seed $i

	python main.py \
	--dev \
	--policy "TD3" \
	--env "Ant-v5" \
	--seed $i

	python main.py \
	--dev \
	--policy "TD3" \
	--env "Humanoid-v5" \
	--seed $i

	python main.py \
	--dev \
	--policy "TD3" \
	--env "InvertedPendulum-v5" \
	--seed $i \
	--start_timesteps 1000

	python main.py \
	--dev \
	--policy "TD3" \
	--env "InvertedDoublePendulum-v5" \
	--seed $i \
	--start_timesteps 1000

	python main.py \
	--dev \
	--policy "TD3" \
	--env "Reacher-v5" \
	--seed $i \
	--start_timesteps 1000
done
