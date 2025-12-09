models = TinyLlama-1.1B-Chat-v1.0 llama-2-7b-hf Meta-Llama-3-8B Llama-2-13b-hf opt-125m opt-1.3b opt-2.7b opt-6.7b opt-13b Qwen2.5-0.5B Qwen2.5-1.5B Qwen2.5-7B Qwen2.5-14B llama-7b-hf llama-13b-hf llama-30b-hf

default: smooth_int

# for pred_len in 96 192 336 720; do
# block_size=0 means no quantization
debug:
	@start=$$(date +%s); \
	seq_len=336; \
	model_name=PatchTST; \
	model_id_name=weather; \
	for pred_len in 96; do \
		for block_size in 0; do \
			for elem_format in int4; do \
				for acc_bits in 0; do \
				python -u run.py --random_seed=2021 --is_training=0 --root_path=./dataset/ --data_path=weather.csv --model_id=$${model_id_name}_$${seq_len}_$$pred_len --model=$$model_name --data=custom --features=M --seq_len=$${seq_len} --pred_len=$$pred_len --enc_in=21 --e_layers=3 --n_heads=16 --d_model=128 --d_ff=256 --dropout=0.2 --fc_dropout=0.2 --head_dropout=0 --patch_len=16 --stride=8 --des='Exp' --train_epochs=100 --patience=20 --itr=1 --batch_size=128 --learning_rate=0.0001 \
				--block_size=$$block_size --w_elem_format=$$elem_format --a_elem_format=$$elem_format --acc_bits=$$acc_bits; \
				done; \
			done; \
		done; \
	done; \
	end=$$(date +%s); \
	delta=$$((end - start)); \
	hours=$$((delta / 3600)); \
	minutes=$$(((delta % 3600) / 60)); \
	echo "\e[36mTime elapsed: $${hours}h-$${minutes}m\e[0m"
smooth_mx:
	@start=$$(date +%s); \
	seq_len=336; \
	model_name=PatchTST; \
	model_id_name=weather; \
	for pred_len in 96; do \
		for alpha in 0.1 0.9; do \
			python -u mysmoothquant/run_smooth.py --random_seed=2021 --is_training=0 --root_path=./dataset/ --data_path=weather.csv --model_id=$${model_id_name}_$${seq_len}_$$pred_len --model=$$model_name --data=custom --features=M --seq_len=$${seq_len} --pred_len=$$pred_len --enc_in=21 --e_layers=3 --n_heads=16 --d_model=128 --d_ff=256 --dropout=0.2 --fc_dropout=0.2 --head_dropout=0 --patch_len=16 --stride=8 --des='Exp' --train_epochs=100 --patience=20 --itr=1 --batch_size=128 --learning_rate=0.0001 \
			--mxquant --block_size=32 \
			; \
		done; \
	done; \
	end=$$(date +%s); \
	delta=$$((end - start)); \
	hours=$$((delta / 3600)); \
	minutes=$$(((delta % 3600) / 60)); \
	echo "\e[36mTime elapsed: $${hours}h-$${minutes}m\e[0m"
smooth_int:
	@start=$$(date +%s); \
	seq_len=336; \
	model_name=PatchTST; \
	model_id_name=weather; \
	for pred_len in 96; do \
		for alpha in 0.9; do \
			python -u mysmoothquant/run_smooth.py --random_seed=2021 --is_training=0 --root_path=./dataset/ --data_path=weather.csv --model_id=$${model_id_name}_$${seq_len}_$$pred_len --model=$$model_name --data=custom --features=M --seq_len=$${seq_len} --pred_len=$$pred_len --enc_in=21 --e_layers=3 --n_heads=16 --d_model=128 --d_ff=256 --dropout=0.2 --fc_dropout=0.2 --head_dropout=0 --patch_len=16 --stride=8 --des='Exp' --train_epochs=100 --patience=20 --itr=1 --batch_size=128 --learning_rate=0.0001 \
			--intquant --n_bits=8 --smooth --alpha=$$alpha \
			; \
		done; \
	done; \
	end=$$(date +%s); \
	delta=$$((end - start)); \
	hours=$$((delta / 3600)); \
	minutes=$$(((delta % 3600) / 60)); \
	echo "\e[36mTime elapsed: $${hours}h-$${minutes}m\e[0m"

scale:
	@start=$$(date +%s); \
	seq_len=336; \
	model_name=PatchTST; \
	model_id_name=weather; \
	for pred_len in 96; do \
		python -u mysmoothquant/get_act_scale.py --random_seed=2021 --is_training=0 --root_path=./dataset/ --data_path=weather.csv --model_id=$${model_id_name}_$${seq_len}_$$pred_len --model=$$model_name --data=custom --features=M --seq_len=$${seq_len} --pred_len=$$pred_len --enc_in=21 --e_layers=3 --n_heads=16 --d_model=128 --d_ff=256 --dropout=0.2 --fc_dropout=0.2 --head_dropout=0 --patch_len=16 --stride=8 --des='Exp' --train_epochs=100 --patience=20 --itr=1 --batch_size=128 --learning_rate=0.0001 \
		; \
	done; \
	end=$$(date +%s); \
	delta=$$((end - start)); \
	hours=$$((delta / 3600)); \
	minutes=$$(((delta % 3600) / 60)); \
	echo "\e[36mTime elapsed: $${hours}h-$${minutes}m\e[0m"
weather:
	@start=$$(date +%s); \
	seq_len=336; \
	model_name=PatchTST; \
	model_id_name=weather; \
	for pred_len in 96 192 336 720; do \
		for block_size in 8 16 32 64; do \
			for elem_format in int3 int4 int5 int6 int7 int8; do \
			python -u run.py --random_seed=2021 --is_training=0 --root_path=./dataset/ --data_path=weather.csv --model_id=$${model_id_name}_$${seq_len}_$$pred_len --model=$$model_name --data=custom --features=M --seq_len=$${seq_len} --pred_len=$$pred_len --enc_in=21 --e_layers=3 --n_heads=16 --d_model=128 --d_ff=256 --dropout=0.2 --fc_dropout=0.2 --head_dropout=0 --patch_len=16 --stride=8 --des='Exp' --train_epochs=100 --patience=20 --itr=1 --batch_size=128 --learning_rate=0.0001 \
			--block_size=$$block_size --w_elem_format=$$elem_format --a_elem_format=$$elem_format; \
			done; \
		done; \
	done; \
	end=$$(date +%s); \
	delta=$$((end - start)); \
	hours=$$((delta / 3600)); \
	minutes=$$(((delta % 3600) / 60)); \
	echo "\e[36mTime elapsed: $${hours}h-$${minutes}m\e[0m"

electricity:
	@start=$$(date +%s); \
	seq_len=336; \
	model_name=PatchTST; \
	model_id_name=electricity; \
	for pred_len in 96 192 336 720; do \
		for block_size in 8; do \
			for elem_format in int3; do \
			python -u train.py --random_seed=2021 --is_training=1 --root_path=./dataset/ --data_path=$${model_id_name}.csv --model_id=$${model_id_name}_$${seq_len}_$$pred_len --model=$$model_name --data=custom --features=M --seq_len=$${seq_len} --pred_len=$$pred_len --enc_in=321 --e_layers=3 --n_heads=16 --d_model=128 --d_ff=256 --dropout=0.2 --fc_dropout=0.2 --head_dropout=0 --patch_len=16 --stride=8 --des='Exp' --train_epochs=100 --patience=10 --lradj=TST --pct_start=0.2 --itr=1 --batch_size=32 --learning_rate=0.0001 \
			--block_size=$$block_size --w_elem_format=$$elem_format --a_elem_format=$$elem_format; \
			done; \
		done; \
	done; \
	end=$$(date +%s); \
	delta=$$((end - start)); \
	hours=$$((delta / 3600)); \
	minutes=$$(((delta % 3600) / 60)); \
	echo "\e[36mTime elapsed: $${hours}h-$${minutes}m\e[0m"

test:
	python test.py
data:
	python mycode/data_process.py

mac:
	python mycode/explore_cuda_MAC.py
