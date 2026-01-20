models = TinyLlama-1.1B-Chat-v1.0 llama-2-7b-hf Meta-Llama-3-8B Llama-2-13b-hf opt-125m opt-1.3b opt-2.7b opt-6.7b opt-13b Qwen2.5-0.5B Qwen2.5-1.5B Qwen2.5-7B Qwen2.5-14B llama-7b-hf llama-13b-hf llama-30b-hf

default: weather

all: electricity
# all: weather traffic electricity etth1 etth2 ettm1 ettm2

# SHELL := /bin/bash
weather:
	@start=$$(date +%s); \
	seq_len=336; \
	model_name=PatchTST; \
	model_id_name=weather; \
	weather_fixed="--random_seed=2021 --is_training=0 --root_path=./dataset/ --data_path=$${model_id_name}.csv  --model=$$model_name --data=custom --features=M --seq_len=$${seq_len} --enc_in=21 --e_layers=3 --n_heads=16 --d_model=128 --d_ff=256 --dropout=0.2 --fc_dropout=0.2 --head_dropout=0 --patch_len=16 --stride=8 --des=Exp --train_epochs=100 --patience=20 --itr=1 --batch_size=128 --learning_rate=0.0001"; \
	electricity_fixed="--random_seed=2021 --is_training=0 --root_path=./dataset/ --data_path=$${model_id_name}.csv --model=$$model_name --data=custom --features=M --seq_len=$${seq_len} --enc_in=321 --e_layers=3 --n_heads=16 --d_model=128 --d_ff=256 --dropout=0.2 --fc_dropout=0.2 --head_dropout=0 --patch_len=16 --stride=8 --des=Exp --train_epochs=100 --patience=10 --itr=1 --batch_size=32 --learning_rate=0.0001"; \
	traffic_fixed="--random_seed=2021 --is_training=0 --root_path=./dataset/ --data_path=$${model_id_name}.csv --model=$$model_name --data=custom --features=M --seq_len=$${seq_len} --enc_in 862 --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 --dropout 0.2 --fc_dropout 0.2 --head_dropout 0 --patch_len 16 --stride 8 --des Exp --train_epochs 100 --patience 10 --lradj 'TST' --pct_start 0.2 --itr 1 --batch_size 64 --learning_rate 0.0001"; \
	fixed=$$weather_fixed; \
	pred_lens="96"; \
	for pred_len in $$pred_lens; do \
		for smooth_n_samples in 1; do \
			mkdir -p logs/$${model_id_name}_$${seq_len}_$${pred_len} ;\
			python -u mysmoothquant/run_smooth.py $${fixed} \
			--model_id=$${model_id_name}_$${seq_len}_$${pred_len}  --pred_len=$${pred_len} \
			--quant --separate --rotate \
			| tee logs/$${model_id_name}_$${seq_len}_$${pred_len}/123.log ; \
		done ;\
	done ;\
	end=$$(date +%s); \
	delta=$$((end - start)); \
	hours=$$((delta / 3600)); \
	minutes=$$(((delta % 3600) / 60)); \
	echo "Time elapsed: $${hours}h-$${minutes}m"
traffic:
	@start=$$(date +%s); \
	seq_len=512; \
	model_name=PatchTST; \
	model_id_name=traffic; \
	fixed="--random_seed=2021 --is_training=0 --root_path=./dataset/ --data_path=$${model_id_name}.csv --model=$$model_name --data=custom --features=M --seq_len=$${seq_len} --enc_in 862 --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 --dropout 0.2 --fc_dropout 0.2 --head_dropout 0 --patch_len 16 --stride 8 --des Exp --train_epochs 100 --patience 10 --lradj 'TST' --pct_start 0.2 --itr 1 --batch_size 64 --learning_rate 0.0001"; \
	pred_lens="96 192 336 720"; \
	for pred_len in $$pred_lens; do \
		for smooth_n_samples in 1; do \
			mkdir -p logs/$${model_id_name}_$${seq_len}_$${pred_len} ;\
			python -u mysmoothquant/run_smooth.py $${fixed} \
			--model_id=$${model_id_name}_$${seq_len}_$${pred_len}  --pred_len=$${pred_len} \
			--quant --search --load_cfg --powersmooth \
			| tee logs/$${model_id_name}_$${seq_len}_$${pred_len}/123.log ; \
		done ;\
	done ;\
	end=$$(date +%s); \
	delta=$$((end - start)); \
	hours=$$((delta / 3600)); \
	minutes=$$(((delta % 3600) / 60)); \
	echo "Time elapsed: $${hours}h-$${minutes}m"
electricity:
	@start=$$(date +%s); \
	seq_len=512; \
	model_name=PatchTST; \
	model_id_name=electricity; \
	fixed="--random_seed=2021 --is_training=0 --root_path=./dataset/ --data_path=$${model_id_name}.csv --model=$$model_name --data=custom --features=M --seq_len=$${seq_len} --enc_in=321 --e_layers=3 --n_heads=16 --d_model=128 --d_ff=256 --dropout=0.2 --fc_dropout=0.2 --head_dropout=0 --patch_len=16 --stride=8 --des=Exp --train_epochs=100 --patience=10 --itr=1 --batch_size=128 --learning_rate=0.0001"; \
	pred_lens="96 192 336"; \
	for pred_len in $$pred_lens; do \
		for smooth_n_samples in 1; do \
			mkdir -p logs/$${model_id_name}_$${seq_len}_$${pred_len} ;\
			python -u mysmoothquant/run_smooth.py $${fixed} \
			--model_id=$${model_id_name}_$${seq_len}_$${pred_len}  --pred_len=$${pred_len} \
			--quant --search --load_cfg --powersmooth \
			| tee logs/$${model_id_name}_$${seq_len}_$${pred_len}/123.log ; \
		done ;\
	done ;\
	end=$$(date +%s); \
	delta=$$((end - start)); \
	hours=$$((delta / 3600)); \
	minutes=$$(((delta % 3600) / 60)); \
	echo "Time elapsed: $${hours}h-$${minutes}m"

etth1:
	@start=$$(date +%s); \
	seq_len=512; \
	model_name=PatchTST; \
	model_id_name=ETTh1; \
	fixed="--random_seed=2021 --is_training 0 --root_path=./dataset/ --data_path=$${model_id_name}.csv --model=$$model_name --data=$${model_id_name} --features M --seq_len=$${seq_len} --enc_in 7 --e_layers 3 --n_heads 4 --d_model 16 --d_ff 128 --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 --patch_len 16 --stride 8 --des Exp --train_epochs 100 --itr 1 --batch_size 128 --learning_rate 0.0001"; \
	pred_lens="96 192 336 720"; \
	for pred_len in $$pred_lens; do \
		for smooth_n_samples in 1; do \
			mkdir -p logs/$${model_id_name}_$${seq_len}_$${pred_len} ;\
			python -u mysmoothquant/run_smooth.py $${fixed} \
			--model_id=$${model_id_name}_$${seq_len}_$${pred_len}  --pred_len=$${pred_len} \
			--quant --load_cfg --search --powersmooth \
			| tee logs/$${model_id_name}_$${seq_len}_$${pred_len}/123.log ; \
		done ;\
	done ;\
	end=$$(date +%s); \
	delta=$$((end - start)); \
	hours=$$((delta / 3600)); \
	minutes=$$(((delta % 3600) / 60)); \
	echo "Time elapsed: $${hours}h-$${minutes}m"

etth2:
	@start=$$(date +%s); \
	seq_len=512; \
	model_name=PatchTST; \
	model_id_name=ETTh2; \
	fixed="--random_seed=2021 --is_training 0 --root_path=./dataset/ --data_path=$${model_id_name}.csv --model=$$model_name --data=$${model_id_name} --features M --seq_len=$${seq_len} --enc_in 7 --e_layers 3 --n_heads 4 --d_model 16 --d_ff 128 --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 --patch_len 16 --stride 8 --des Exp --train_epochs 100 --itr 1 --batch_size 128 --learning_rate 0.0001"; \
	pred_lens="96 192 336 720"; \
	for pred_len in $$pred_lens; do \
		for smooth_n_samples in 1; do \
			mkdir -p logs/$${model_id_name}_$${seq_len}_$${pred_len} ;\
			python -u mysmoothquant/run_smooth.py $${fixed} \
			--model_id=$${model_id_name}_$${seq_len}_$${pred_len}  --pred_len=$${pred_len} \
			--quant --load_cfg --search --powersmooth \
			| tee logs/$${model_id_name}_$${seq_len}_$${pred_len}/123.log ; \
		done ;\
	done ;\
	end=$$(date +%s); \
	delta=$$((end - start)); \
	hours=$$((delta / 3600)); \
	minutes=$$(((delta % 3600) / 60)); \
	echo "Time elapsed: $${hours}h-$${minutes}m"

ettm1:
	@start=$$(date +%s); \
	seq_len=512; \
	model_name=PatchTST; \
	model_id_name=ETTm1; \
	fixed="--random_seed 2021 --is_training 1 --root_path=./dataset --data_path=$${model_id_name}.csv --model $$model_name --data $${model_id_name} --features M --seq_len $${seq_len} --enc_in 7 --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 --dropout 0.2 --fc_dropout 0.2 --head_dropout 0 --patch_len 16 --stride 8 --des Exp --train_epochs 100 --patience 20 --lradj 'TST' --pct_start 0.4 --itr 1 --batch_size 128 --learning_rate 0.0001"; \
	pred_lens="96 192 336 720"; \
	for pred_len in $$pred_lens; do \
		for smooth_n_samples in 1; do \
			mkdir -p logs/$${model_id_name}_$${seq_len}_$${pred_len} ;\
			python -u mysmoothquant/run_smooth.py $${fixed} \
			--model_id=$${model_id_name}_$${seq_len}_$${pred_len}  --pred_len=$${pred_len} \
			--quant --load_cfg --search --powersmooth \
			| tee logs/$${model_id_name}_$${seq_len}_$${pred_len}/123.log ; \
		done ;\
	done ;\
	end=$$(date +%s); \
	delta=$$((end - start)); \
	hours=$$((delta / 3600)); \
	minutes=$$(((delta % 3600) / 60)); \
	echo "Time elapsed: $${hours}h-$${minutes}m"

ettm2:
	@start=$$(date +%s); \
	seq_len=512; \
	model_name=PatchTST; \
	model_id_name=ETTm2; \
	fixed="--random_seed 2021 --is_training 1 --root_path=./dataset --data_path=$${model_id_name}.csv --model $$model_name --data $${model_id_name} --features M --seq_len $${seq_len} --enc_in 7 --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 --dropout 0.2 --fc_dropout 0.2 --head_dropout 0 --patch_len 16 --stride 8 --des Exp --train_epochs 100 --patience 20 --lradj 'TST' --pct_start 0.4 --itr 1 --batch_size 128 --learning_rate 0.0001"; \
	pred_lens="96 192 336 720"; \
	for pred_len in $$pred_lens; do \
		for smooth_n_samples in 1; do \
			mkdir -p logs/$${model_id_name}_$${seq_len}_$${pred_len} ;\
			python -u mysmoothquant/run_smooth.py $${fixed} \
			--model_id=$${model_id_name}_$${seq_len}_$${pred_len}  --pred_len=$${pred_len} \
			--quant --load_cfg --search --powersmooth \
			| tee logs/$${model_id_name}_$${seq_len}_$${pred_len}/123.log ; \
		done ;\
	done ;\
	end=$$(date +%s); \
	delta=$$((end - start)); \
	hours=$$((delta / 3600)); \
	minutes=$$(((delta % 3600) / 60)); \
	echo "Time elapsed: $${hours}h-$${minutes}m"

scale_patchTST:
	@start=$$(date +%s); \
	seq_len=512; \
	pred_len=336; \
	model_name=PatchTST; \
	model_id_name=traffic; \
	weather_fixed="--random_seed=2021 --is_training=0 --root_path=./dataset/ --data_path=$${model_id_name}.csv --model_id=$${model_id_name}_$${seq_len}_$${pred_len} --model=$$model_name --data=custom --features=M --seq_len=$${seq_len} --pred_len=$${pred_len} --enc_in=21 --e_layers=3 --n_heads=16 --d_model=128 --d_ff=256 --dropout=0.2 --fc_dropout=0.2 --head_dropout=0 --patch_len=16 --stride=8 --des=Exp --train_epochs=100 --patience=20 --itr=1 --batch_size=128 --learning_rate=0.0001"; \
	electricity_fixed="--random_seed=2021 --is_training=0 --root_path=./dataset/ --data_path=$${model_id_name}.csv --model_id=$${model_id_name}_$${seq_len}_$${pred_len} --model=$$model_name --data=custom --features=M --seq_len=$${seq_len} --pred_len=$${pred_len} --enc_in=321 --e_layers=3 --n_heads=16 --d_model=128 --d_ff=256 --dropout=0.2 --fc_dropout=0.2 --head_dropout=0 --patch_len=16 --stride=8 --des=Exp --train_epochs=100 --patience=10 --itr=1 --batch_size=32 --learning_rate=0.0001"; \
	traffic_fixed="--random_seed=2021 --is_training=0 --root_path=./dataset/ --data_path=$${model_id_name}.csv --model_id=$${model_id_name}_$${seq_len}_$${pred_len} --model=$$model_name --data=custom --features=M --seq_len=$${seq_len} --pred_len=$${pred_len} --enc_in 862 --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 --dropout 0.2 --fc_dropout 0.2 --head_dropout 0 --patch_len 16 --stride 8 --des Exp --train_epochs 100 --patience 10 --lradj 'TST' --pct_start 0.2 --itr 1 --batch_size 24 --learning_rate 0.0001"; \
	fixed=$$traffic_fixed; \
	for pred_len in 96; do \
		python -u mysmoothquant/get_act_scale_patchTST.py $${fixed} \
		; \
	done; \
	end=$$(date +%s); \
	delta=$$((end - start)); \
	hours=$$((delta / 3600)); \
	minutes=$$(((delta % 3600) / 60)); \
	echo "\e[36mTime elapsed: $${hours}h-$${minutes}m\e[0m"

scale_llm:
	@methods="--eval_clamp --eval_clamp_qwt --wgt_nbit=4 --act_nbit=8"; \
	methods="--eval_base --wgt_nbit=4 --act_nbit=8"; \
	models="TinyLlama-1.1B-Chat-v1.0 llama-2-7b-hf Qwen2.5-7B"; \
	models="opt-125m"; \
	tasks="wikitext"; \
	start=$$(date +%s); \
	for model in $$models; do \
		for task in $$tasks; do \
			CUDA_VISIBLE_DEVICES=0 python mysmoothquant/get_act_scale.py --model_name=$$model; \
		done; \
	done; \
	end=$$(date +%s); \
	delta=$$((end - start)); \
	hours=$$((delta / 3600)); \
	minutes=$$(((delta % 3600) / 60)); \
	echo "\e[36mTime elapsed: $${hours}h-$${minutes}m\e[0m"

train:
	@start=$$(date +%s); \
	seq_len=512; \
	model_name=PatchTST; \
	model_id_name=electricity; \
	pred_lens=96 96 192 336 720 336 720; \
	pred_lens=720; \
	for pred_len in $$pred_lens; do \
		python -u train.py --random_seed=2021 --is_training=1 --root_path=./dataset/ --data_path=$${model_id_name}.csv --model_id=$${model_id_name}_$${seq_len}_$$pred_len --model=$$model_name --data=custom --features=M --seq_len=$${seq_len} --pred_len=$$pred_len --enc_in=321 --e_layers=3 --n_heads=16 --d_model=128 --d_ff=256 --dropout=0.2 --fc_dropout=0.2 --head_dropout=0 --patch_len=16 --stride=8 --des='Exp' --train_epochs=100 --patience=10 --lradj=TST --pct_start=0.2 --itr=1 --batch_size=32 --learning_rate=0.0001; \
	done; \
	end=$$(date +%s); \
	delta=$$((end - start)); \
	hours=$$((delta / 3600)); \
	minutes=$$(((delta % 3600) / 60)); \
	echo "\e[36mTime elapsed: $${hours}h-$${minutes}m\e[0m"

test:
	python test.py

result:
	python mycode/result_process.py
fig:
	python mycode/profile_like_awq.py

mac:
	python mycode/explore_cuda_MAC.py


ppl:
	@methods="--eval_clamp --eval_clamp_qwt --wgt_nbit=4 --act_nbit=8"; \
	methods="--eval_base --wgt_nbit=4 --act_nbit=8"; \
	models="TinyLlama-1.1B-Chat-v1.0 llama-2-7b-hf Qwen2.5-0.5B"; \
	models="opt-125m"; \
	tasks="wikitext"; \
	start=$$(date +%s); \
	for model in $$models; do \
		for task in $$tasks; do \
			CUDA_VISIBLE_DEVICES=0 python llm_eval.py --model_name=$$model \
			--quant --search \
			; \
		done; \
	done; \
	end=$$(date +%s); \
	delta=$$((end - start)); \
	hours=$$((delta / 3600)); \
	minutes=$$(((delta % 3600) / 60)); \
	echo "\e[36mTime elapsed: $${hours}h-$${minutes}m\e[0m"