models = TinyLlama-1.1B-Chat-v1.0 llama-2-7b-hf Meta-Llama-3-8B Llama-2-13b-hf opt-125m opt-1.3b opt-2.7b opt-6.7b opt-13b Qwen2.5-0.5B Qwen2.5-1.5B Qwen2.5-7B Qwen2.5-14B llama-7b-hf llama-13b-hf llama-30b-hf
methods = --eval_base --eval_quant --eval_clamp --eval_quant_qwt --eval_clamp_qwt

default: debug

org:
	@methods="--eval_clamp --wgt_nbit=4 --act_nbit=8"; \
	methods="--eval_quant --wgt_nbit=4 --act_nbit=8 --n_samples=1"; \
	models="Mistral-7B-v0.1 llama-2-7b-hf Meta-Llama-3-8B Qwen2.5-7B Qwen3-8B Llama-2-13b-hf Qwen3-14B"; \
	models="llama-2-7b-hf"; \
	tasks="wikitext"; \
	start=$$(date +%s); \
	for task in $$tasks; do \
		for model in $$models; do \
			echo "TOKENIZERS_PARALLELISM=false python main.py --model=/localssd/lbxj/$$model  --rotate --a_bits=4 --v_bits=4 --k_bits=4 --w_bits=4 --w_clip"; \
			TOKENIZERS_PARALLELISM=false python main.py --model=/localssd/lbxj/$$model --a_bits=8 --w_bits=8 --w_rtn --v_bits=16 --k_bits=16; \
			TOKENIZERS_PARALLELISM=false python main.py --model=/localssd/lbxj/$$model --a_bits=8 --w_bits=8 --rotate --w_rtn --v_bits=16 --k_bits=16; \
			TOKENIZERS_PARALLELISM=false python main.py --model=/localssd/lbxj/$$model --a_bits=8 --w_bits=8 --rotate --rotate_online --w_rtn --v_bits=16 --k_bits=16; \
		done; \
	done; \
	end=$$(date +%s); \
	delta=$$((end - start)); \
	hours=$$((delta / 3600)); \
	minutes=$$(((delta % 3600) / 60)); \
	echo "\e[36mTime elapsed: $${hours}h-$${minutes}m\e[0m"
debug:
	@methods="--eval_clamp --wgt_nbit=4 --act_nbit=8"; \
	methods="--eval_quant --wgt_nbit=4 --act_nbit=8 --n_samples=1"; \
	models="Mistral-7B-v0.1 llama-2-7b-hf Meta-Llama-3-8B Qwen2.5-7B Qwen3-8B Llama-2-13b-hf Qwen3-14B"; \
	models="llama-2-7b-hf"; \
	tasks="wikitext"; \
	start=$$(date +%s); \
	for task in $$tasks; do \
		for model in $$models; do \
			echo "TOKENIZERS_PARALLELISM=false python main.py --model=/localssd/lbxj/$$model --rotate --a_bits=4 --v_bits=4 --k_bits=4 --w_bits=4 --w_clip"; \
			TOKENIZERS_PARALLELISM=false python main.py --model=/localssd/lbxj/$$model --w_rtn --a_bits=16 --v_bits=16 --k_bits=16 --w_bits=16 --w_clip; \
		done; \
	done; \
	end=$$(date +%s); \
	delta=$$((end - start)); \
	hours=$$((delta / 3600)); \
	minutes=$$(((delta % 3600) / 60)); \
	echo "\e[36mTime elapsed: $${hours}h-$${minutes}m\e[0m"

test:
	python mycode/test.py

data:
	python mycode/data_process.py
gpu:
	python shared_code/use_gpu.py



