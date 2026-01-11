import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# 创建图形
fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis('off')

# 颜色定义
colors = {
    'start': '#2E86AB',        # 深蓝色
    'calibration': '#A23B72',   # 紫红色
    'mse': '#F18F01',          # 橙色
    'binary': '#C73E1D',       # 红色
    'bit4': '#6A4C93',         # 紫色
    'bit8': '#1982C4',         # 蓝色
    'smooth': '#3E8914',       # 绿色
    'end': '#404E4D',          # 深灰色
    'arrow': '#5D5D5D',        # 灰色
    'highlight': '#FFD166'     # 黄色高亮
}

# 1. 主流程标题
ax.text(8, 11.5, "Mixed Precision Quantization Search Flow", fontsize=18, fontweight='bold', 
        ha='center', color=colors['start'])
ax.text(8, 11, "Strategy: From Highest Precision to Lowest Cost", fontsize=12, 
        ha='center', color=colors['binary'])

# 2. 第一步：Calibration和MSE计算
ax.add_patch(patches.FancyBboxPatch((1, 9), 14, 1.5, 
                                    boxstyle="round,pad=0.1",
                                    facecolor=colors['calibration'], 
                                    alpha=0.9, edgecolor='black'))
ax.text(8, 9.75, "Step 1: Calibration and Difference Calculation", fontsize=14, 
        fontweight='bold', ha='center', color='white')

# 子步骤
steps = [
    (3, 9.3, "1. Compute layer-wise INT8 vs FP32 differences on calibration data"),
    (3, 9.0, "2. Use MSE or KL divergence (MSE works better)"),
    (13, 9.3, "3. Get quantization difficulty scores"),
    (13, 9.0, "4. Sort by difficulty ascending to get mse_list")
]

for x, y, text in steps:
    ax.text(x, y, f"• {text}", fontsize=10, ha='left', color='white')

# 3. 第二步：二分搜索框架
ax.add_patch(patches.FancyBboxPatch((1, 7), 14, 1.5, 
                                    boxstyle="round,pad=0.1",
                                    facecolor=colors['binary'], 
                                    alpha=0.9, edgecolor='black'))
ax.text(8, 7.75, "Step 2: Binary Search for Quantization Config", fontsize=14, 
        fontweight='bold', ha='center', color='white')

# 4. 二分搜索过程图示
# 左侧：4-bit量化分支
ax.add_patch(patches.FancyBboxPatch((2, 5), 4, 3, 
                                    boxstyle="round,pad=0.1",
                                    facecolor=colors['bit4'], 
                                    alpha=0.8, edgecolor='black'))
ax.text(4, 7.5, "4-bit Quantization Search", fontsize=12, 
        fontweight='bold', ha='center', color='white')

# 4-bit内部步骤
ax.text(4, 6.8, "Take layers mse_list[:mid]", fontsize=10, ha='center', color='white')
ax.text(4, 6.3, "Try replace with 4-bit", fontsize=10, ha='center', color='white')

# INT4 vs BFP4决策
ax.add_patch(patches.FancyBboxPatch((2.5, 5.5), 3, 0.8, 
                                    boxstyle="round,pad=0.1",
                                    facecolor='#8B4C94', 
                                    alpha=0.9, edgecolor='black'))
ax.text(4, 5.9, "INT4 vs BFP4 Selection", fontsize=10, 
        fontweight='bold', ha='center', color='white')

# 决策条件
ax.text(4, 5.3, "INT4 MSE × 1.2 < BFP4 MSE → Choose INT4", fontsize=9, 
        ha='center', color='white')
ax.text(4, 5.0, "Otherwise → Choose BFP4 (lower hardware cost)", fontsize=9, 
        ha='center', color='white')

# 右侧：8-bit量化分支
ax.add_patch(patches.FancyBboxPatch((10, 5), 4, 3, 
                                    boxstyle="round,pad=0.1",
                                    facecolor=colors['bit8'], 
                                    alpha=0.8, edgecolor='black'))
ax.text(12, 7.5, "8-bit Quantization Search", fontsize=12, 
        fontweight='bold', ha='center', color='white')

# 8-bit内部步骤
ax.text(12, 6.8, "Continue binary search on unfrozen layers", fontsize=10, ha='center', color='white')
ax.text(12, 6.3, "Replace with BFP8", fontsize=10, ha='center', color='white')

# 中间：阈值检查
ax.add_patch(patches.FancyBboxPatch((6.5, 4), 3, 1, 
                                    boxstyle="round,pad=0.1",
                                    facecolor=colors['highlight'], 
                                    alpha=0.9, edgecolor='black'))
ax.text(8, 4.5, "MSE Threshold Check", fontsize=11, 
        fontweight='bold', ha='center', color='black')
ax.text(8, 4.0, "Satisfy threshold → Freeze current config", fontsize=10, 
        ha='center', color='black')

# 5. 第三步：Smooth配置
ax.add_patch(patches.FancyBboxPatch((2, 1.5), 12, 2, 
                                    boxstyle="round,pad=0.1",
                                    facecolor=colors['smooth'], 
                                    alpha=0.9, edgecolor='black'))
ax.text(8, 2.5, "Step 3: Layer-wise Smooth Configuration", fontsize=14, 
        fontweight='bold', ha='center', color='white')

# Smooth步骤
smooth_steps = [
    (3, 2.0, "1. Try Smooth config for each layer"),
    (3, 1.7, "2. Compute output difference after Smooth"),
    (13, 2.0, "3. Difference < threshold → Enable Smooth"),
    (13, 1.7, "4. Otherwise → Keep original config")
]

for x, y, text in smooth_steps:
    ax.text(x, y, f"• {text}", fontsize=10, ha='left', color='white')

# 6. 最终结果
ax.add_patch(patches.FancyBboxPatch((6, 0), 4, 1, 
                                    boxstyle="round,pad=0.1",
                                    facecolor=colors['end'], 
                                    alpha=0.9, edgecolor='black'))
ax.text(8, 0.5, "Result: Mixed Precision Model", fontsize=12, 
        fontweight='bold', ha='center', color='white')

# 7. 添加流程箭头
# 从第一步到第二步
ax.annotate('', xy=(8, 9), xytext=(8, 8.5),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], 
                           lw=2, shrinkA=5, shrinkB=5))

# 从第二步到4-bit分支
ax.annotate('', xy=(5, 7), xytext=(5, 6.5),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], 
                           lw=2, shrinkA=5, shrinkB=5))

# 从第二步到8-bit分支
ax.annotate('', xy=(11, 7), xytext=(11, 6.5),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], 
                           lw=2, shrinkA=5, shrinkB=5))

# 从4-bit到阈值检查
ax.annotate('', xy=(5, 5), xytext=(8, 5),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], 
                           lw=2, shrinkA=5, shrinkB=5))

# 从8-bit到阈值检查
ax.annotate('', xy=(11, 5), xytext=(9.5, 5),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], 
                           lw=2, shrinkA=5, shrinkB=5))

# 从阈值检查到Smooth
ax.annotate('', xy=(8, 4), xytext=(8, 3.5),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], 
                           lw=2, shrinkA=5, shrinkB=5))

# 从Smooth到最终结果
ax.annotate('', xy=(8, 1.5), xytext=(8, 1),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], 
                           lw=2, shrinkA=5, shrinkB=5))

# 8. 添加循环箭头（表示二分搜索的迭代过程）
# 从阈值检查回到二分搜索
ax.annotate('', xy=(7, 4.5), xytext=(5, 5.5),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], 
                           lw=2, linestyle='--', alpha=0.7,
                           connectionstyle="arc3,rad=-0.3"))

# 添加"不满足阈值，调整mid"文字
ax.text(5.5, 5.8, "Threshold not met\nAdjust mid and continue", fontsize=8, 
        ha='center', color=colors['binary'], style='italic')

# 9. 添加图例
legend_elements = [
    patches.Patch(facecolor=colors['calibration'], alpha=0.9, label='Calibration & Diff Calculation'),
    patches.Patch(facecolor=colors['binary'], alpha=0.9, label='Binary Search Framework'),
    patches.Patch(facecolor=colors['bit4'], alpha=0.8, label='4-bit Quantization Branch'),
    patches.Patch(facecolor=colors['bit8'], alpha=0.8, label='8-bit Quantization Branch'),
    patches.Patch(facecolor=colors['smooth'], alpha=0.9, label='Smooth Configuration'),
    patches.Patch(facecolor=colors['end'], alpha=0.9, label='Final Output'),
    patches.Patch(facecolor=colors['highlight'], alpha=0.9, label='Threshold Checkpoint'),
]

ax.legend(handles=legend_elements, loc='upper center', 
          bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=9)

# 10. 添加关键说明
ax.text(8, -0.8, "Key Features: Layer-wise Processing | Auto Precision Selection | Hardware-Aware Optimization", 
        fontsize=10, ha='center', color=colors['binary'], style='italic')

plt.tight_layout()
plt.savefig("Mixed_Precision_Quantization_Search_Flow.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("Image saved as 'Mixed_Precision_Quantization_Search_Flow.png'")
