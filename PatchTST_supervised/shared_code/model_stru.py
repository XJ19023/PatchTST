from accelerate import init_empty_weights

from transformers import AutoModelForCausalLM, AutoConfig

config = AutoConfig.from_pretrained("/cephfs/shared/impact/LLM_models/gpt2-xl")

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

print(model)