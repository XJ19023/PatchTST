import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
activations_save = {}
weights_save = {}
hooks, layers, layers_name = [], [], []
forward_hook_counter = [0]
# 定义钩子函数
def create_hook(layer_name, layer):
    def layer_hook(module, inp, out):
        if layer_name == 'model.layers.0.self_attn.q_proj':
            forward_hook_counter[0] += 1
        layers_name.append(f'{layer_name}_case{forward_hook_counter[0]}')
        print(f'>>> name {layer_name}')
        # print(f'layer {layer}\n')
    return layer_hook


MODEL_ID = "/cephfs/juxin/models/opt-125m"
# MODEL_ID = "quantizedModel"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

regist_num = 0
for name, module in model.named_modules():
    # print(f'{regist_num}, {name}, {module}\n')
    hook = module.register_forward_hook(create_hook(name, module)) # 注册hook
    hooks.append(hook)
    regist_num += 1




model.to(device=torch.device("cuda"))
# 设置模型为评估模式
model.eval()

# 编写 Prompt（以 LLaMA2 风格为例）
prompt = "Q: What is the capital of China?\nA:"

# 编码输入
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 模型生成
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
    )

# 解码输出
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 打印结果
print("=== Output ===")
print(response)

for hook in hooks:
    hook.remove()

print(f'reg num {regist_num}')
print(f'forward_hook_counter {forward_hook_counter[0]}')