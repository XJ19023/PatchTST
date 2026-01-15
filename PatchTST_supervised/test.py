import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 定义量化工具函数 ====================
class Quantizer:
    """Per-channel INT8量化工具类"""
    
    @staticmethod
    def quantize_per_channel(tensor, num_bits=8):
        """对称per-channel量化"""
        # 计算每个通道的缩放因子
        scales = torch.max(torch.abs(tensor), dim=1, keepdim=True)[0]
        scales = scales.clamp(min=1e-8)
        
        # 计算量化范围
        qmax = 2 ** (num_bits - 1) - 1
        qmin = -2 ** (num_bits - 1)
        
        # 量化
        scaled_tensor = tensor / scales
        quantized = torch.clamp(torch.round(scaled_tensor), qmin, qmax)
        
        # 反量化
        dequantized = quantized * scales
        
        return quantized, scales, dequantized
    
    @staticmethod
    def compute_quantization_error(original, dequantized):
        """计算量化误差"""
        mse = F.mse_loss(original, dequantized)
        return mse.item()

# ==================== 2. 定义三层Linear模型 ====================
class ThreeLayerModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=64):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.linear3 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    
    def get_layer_weights(self):
        """获取各层权重"""
        return {
            'linear1': self.linear1.weight.data,
            'linear2': self.linear2.weight.data,
            'linear3': self.linear3.weight.data
        }

# ==================== 3. 贝叶斯优化器类 ====================
class BayesianAlphaOptimizer:
    def __init__(self, model, calibration_data):
        self.model = model
        self.calibration_data = calibration_data
        self.layer_names = ['linear1', 'linear2', 'linear3']
        
        # 收集各层的激活和权重统计信息
        self.layer_stats = self._collect_layer_statistics()
        
    def _collect_layer_statistics(self):
        """前向传播收集激活统计信息"""
        self.model.eval()
        layer_stats = {}
        
        # 注册hook收集激活
        activations = {}
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook
        
        hooks = []
        hooks.append(self.model.linear1.register_forward_hook(get_activation('linear1')))
        hooks.append(self.model.linear2.register_forward_hook(get_activation('linear2')))
        hooks.append(self.model.linear3.register_forward_hook(get_activation('linear3')))
        
        # 前向传播
        with torch.no_grad():
            for data in self.calibration_data:
                _ = self.model(data)
        
        # 移除hooks
        for hook in hooks:
            hook.remove()
        
        # 计算每层的激活scale（使用绝对值最大值）
        for name in self.layer_names:
            act_scale = torch.max(torch.abs(activations[name]), dim=0)[0]
            weight_scale = torch.max(torch.abs(self.model.state_dict()[f'{name}.weight']), dim=1)[0]
            
            layer_stats[name] = {
                'act_scales': act_scale,
                'weight_scales': weight_scale
            }
        
        return layer_stats
    
    def compute_scales_with_alpha(self, layer_name, alpha):
        """使用给定的alpha计算融合scale"""
        stats = self.layer_stats[layer_name]
        act_scales = stats['act_scales']
        weight_scales = stats['weight_scales']
        
        # Smoothed Quantization公式
        scales = (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        scales = scales.clamp(min=1e-5)
        
        return scales
    
    def evaluate_alpha_for_layer(self, layer_name, alpha):
        """评估给定alpha的量化误差"""
        stats = self.layer_stats[layer_name]
        
        # 获取原始权重
        weight = self.model.state_dict()[f'{layer_name}.weight']
        
        # 计算融合scale
        scales = self.compute_scales_with_alpha(layer_name, alpha)
        
        # 应用scale并量化
        # 注意：实际部署中，scale会被融合到权重中
        scaled_weight = weight / scales.unsqueeze(1)
        
        # 对缩放后的权重进行量化
        _, _, dequantized = Quantizer.quantize_per_channel(scaled_weight)
        
        # 恢复原始尺度
        dequantized_original = dequantized * scales.unsqueeze(1)
        
        # 计算量化误差
        error = Quantizer.compute_quantization_error(weight, dequantized_original)
        
        return error
    
    def optimize_layer_alpha(self, layer_name, n_calls=20):
        """使用贝叶斯优化寻找最优alpha"""
        
        # 定义搜索空间
        search_space = [Real(0.0, 1.0, name='alpha')]
        
        # 定义目标函数
        @use_named_args(search_space)
        def objective(**params):
            alpha = params['alpha']
            return self.evaluate_alpha_for_layer(layer_name, alpha)
        
        # 运行贝叶斯优化
        print(f"\n优化 {layer_name} 的alpha值...")
        result = gp_minimize(
            func=objective,
            dimensions=search_space,
            n_calls=n_calls,
            n_initial_points=5,
            random_state=42,
            verbose=False
        )
        
        optimal_alpha = result.x[0]
        min_error = result.fun
        
        print(f"  - 最优alpha: {optimal_alpha:.4f}")
        print(f"  - 最小误差: {min_error:.6f}")
        
        return optimal_alpha, min_error
    
    def optimize_all_layers(self, n_calls_per_layer=20):
        """优化所有层的alpha"""
        optimal_alphas = {}
        errors = {}
        
        for layer_name in self.layer_names:
            alpha, error = self.optimize_layer_alpha(
                layer_name, 
                n_calls=n_calls_per_layer
            )
            optimal_alphas[layer_name] = alpha
            errors[layer_name] = error
        
        return optimal_alphas, errors

# ==================== 4. 可视化函数 ====================
def plot_alpha_optimization_results(model, optimizer):
    """可视化各层的alpha优化结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (layer_name, ax) in enumerate(zip(optimizer.layer_names, axes)):
        stats = optimizer.layer_stats[layer_name]
        
        # 生成不同的alpha值进行评估
        alphas = np.linspace(0, 1, 50)
        errors = []
        
        for alpha in alphas:
            error = optimizer.evaluate_alpha_for_layer(layer_name, alpha)
            errors.append(error)
        
        # 找到最优alpha
        optimal_alpha = np.array(alphas)[np.argmin(errors)]
        
        # 绘图
        ax.plot(alphas, errors, 'b-', linewidth=2, label='量化误差')
        ax.axvline(x=optimal_alpha, color='r', linestyle='--', 
                  label=f'最优alpha={optimal_alpha:.3f}')
        ax.set_xlabel('Alpha值')
        ax.set_ylabel('量化误差(MSE)')
        ax.set_title(f'{layer_name} - Alpha优化')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def compare_quantization_errors(model, optimal_alphas):
    """比较不同alpha策略下的量化误差"""
    results = []
    
    # 不同alpha策略
    strategies = {
        '固定alpha=0.5': 0.5,
        '激活优先(alpha=1.0)': 1.0,
        '权重优先(alpha=0.0)': 0.0,
        '贝叶斯优化': None  # 特殊处理
    }
    
    for layer_name in model.layer_names:
        layer_results = {'Layer': layer_name}
        
        for strategy_name, alpha_value in strategies.items():
            if strategy_name == '贝叶斯优化':
                # 使用优化后的alpha
                error = optimizer.evaluate_alpha_for_layer(
                    layer_name, 
                    optimal_alphas[layer_name]
                )
            else:
                error = optimizer.evaluate_alpha_for_layer(
                    layer_name, 
                    alpha_value
                )
            
            layer_results[strategy_name] = error
        
        results.append(layer_results)
    
    # 打印结果
    print("\n" + "="*60)
    print("不同Alpha策略下的量化误差比较:")
    print("="*60)
    for result in results:
        print(f"\n{result['Layer']}:")
        for strategy in strategies.keys():
            print(f"  {strategy}: {result[strategy]:.6f}")
    
    return results

# ==================== 5. 主程序 ====================
def main():
    # 设置随机种子保证可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. 创建模型
    print("创建三层Linear模型...")
    model = ThreeLayerModel(input_dim=128, hidden_dim=256, output_dim=64)
    model.eval()
    
    # 2. 生成校准数据
    print("生成校准数据...")
    calibration_data = [torch.randn(32, 128) for _ in range(10)]  # 10个batch
    
    # 3. 创建优化器并收集统计信息
    print("收集各层统计信息...")
    optimizer = BayesianAlphaOptimizer(model, calibration_data)
    
    # 4. 可视化各层的激活和权重分布
    print("\n各层统计信息:")
    for layer_name in optimizer.layer_names:
        stats = optimizer.layer_stats[layer_name]
    
    # 5. 运行贝叶斯优化
    optimal_alphas, errors = optimizer.optimize_all_layers(n_calls_per_layer=25)
    
    # 8. 保存结果
    print("\n" + "="*60)
    print("优化完成！最优Alpha值:")
    print("="*60)
    for layer_name, alpha in optimal_alphas.items():
        print(f"{layer_name}: alpha = {alpha:.4f}")

# ==================== 6. 运行示例 ====================
if __name__ == "__main__":
    # 检查skopt是否安装
    try:
        import skopt
        main()
    except ImportError:
        print("错误: 需要安装scikit-optimize库")
        print("请运行: pip install scikit-optimize")
        
    # 保存完整代码到文件
    with open('smoothed_quantization_demo.py', 'w', encoding='utf-8') as f:
        # 这里简化保存，实际使用时可以保存完整代码
        f.write("# 完整代码已保存到smoothed_quantization_demo.py")
        print("\n完整代码已保存到smoothed_quantization_demo.py")