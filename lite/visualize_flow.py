#!/usr/bin/env python3
"""
图片传递流程可视化演示
展示从宿主机到Docker容器的完整数据流
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def create_data_flow_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    colors = {
        'host': '#E8F4FD',      # 浅蓝色 - 宿主机
        'docker': '#FFF2CC',    # 浅黄色 - Docker
        'process': '#E1D5E7',   # 浅紫色 - 处理过程
        'model': '#D5E8D4',     # 浅绿色 - 模型
        'result': '#FFE6CC'     # 浅橙色 - 结果
    }
    
    # 1. 宿主机Mac系统
    host_box = FancyBboxPatch((0.5, 8), 3, 2, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['host'], 
                              edgecolor='blue', linewidth=2)
    ax.add_patch(host_box)
    ax.text(2, 9, 'macOS 宿主机', ha='center', va='center', fontsize=12, weight='bold')
    ax.text(2, 8.5, '📁 /Users/ding/Desktop/NUS-proj/', ha='center', va='center', fontsize=10)
    
    # 2. 图片文件
    img_box = FancyBboxPatch((0.5, 6), 1.4, 1.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['host'], 
                             edgecolor='blue', linewidth=1)
    ax.add_patch(img_box)
    ax.text(1.25, 6.75, '🖼️ 图片文件', ha='center', va='center', fontsize=10, weight='bold')
    ax.text(1.25, 6.25, 'rd.jpg\nsing.jpg\nsp.jpg', ha='center', va='center', fontsize=9)
    
    # 3. 模型文件
    model_box = FancyBboxPatch((2.1, 6), 1.4, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['model'], 
                               edgecolor='green', linewidth=1)
    ax.add_patch(model_box)
    ax.text(2.8, 6.75, '🧠 TFLite模型', ha='center', va='center', fontsize=10, weight='bold')
    ax.text(2.8, 6.25, 'CatClassifier\n_512V2_2.tflite\n(11MB)', ha='center', va='center', fontsize=9)
    
    # 4. Docker卷挂载
    mount_box = FancyBboxPatch((5, 7), 3, 2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['docker'], 
                               edgecolor='orange', linewidth=2)
    ax.add_patch(mount_box)
    ax.text(6.5, 8.2, '🐳 Docker卷挂载', ha='center', va='center', fontsize=12, weight='bold')
    ax.text(6.5, 7.8, '-v $(pwd):/app/models', ha='center', va='center', fontsize=10)
    ax.text(6.5, 7.4, '-v $(pwd)/../data/test:/app/test_images', ha='center', va='center', fontsize=10)
    
    # 5. 容器内文件系统
    container_box = FancyBboxPatch((9, 6), 3, 4, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor=colors['docker'], 
                                   edgecolor='orange', linewidth=2)
    ax.add_patch(container_box)
    ax.text(10.5, 9.5, '🐧 Linux容器', ha='center', va='center', fontsize=12, weight='bold')
    ax.text(10.5, 9, '/app/models/model.tflite', ha='center', va='center', fontsize=10)
    ax.text(10.5, 8.5, '/app/test_images/rd.jpg', ha='center', va='center', fontsize=10)
    ax.text(10.5, 8, '/app/lite_client.py', ha='center', va='center', fontsize=10)
    
    # 6. 图像预处理
    preprocess_box = FancyBboxPatch((9, 3.5), 3, 2, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor=colors['process'], 
                                    edgecolor='purple', linewidth=2)
    ax.add_patch(preprocess_box)
    ax.text(10.5, 4.7, '🔄 图像预处理', ha='center', va='center', fontsize=12, weight='bold')
    ax.text(10.5, 4.3, 'PIL读取 → 调整尺寸', ha='center', va='center', fontsize=10)
    ax.text(10.5, 3.9, '归一化 → NumPy数组', ha='center', va='center', fontsize=10)
    
    # 7. TFLite推理
    inference_box = FancyBboxPatch((9, 1), 3, 2, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor=colors['model'], 
                                   edgecolor='green', linewidth=2)
    ax.add_patch(inference_box)
    ax.text(10.5, 2.2, '⚡ TFLite推理', ha='center', va='center', fontsize=12, weight='bold')
    ax.text(10.5, 1.8, 'XNNPACK加速', ha='center', va='center', fontsize=10)
    ax.text(10.5, 1.4, 'CPU优化执行', ha='center', va='center', fontsize=10)
    
    # 8. 结果输出
    result_box = FancyBboxPatch((13, 2), 3, 3, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['result'], 
                                edgecolor='red', linewidth=2)
    ax.add_patch(result_box)
    ax.text(14.5, 4, '📊 预测结果', ha='center', va='center', fontsize=12, weight='bold')
    ax.text(14.5, 3.5, '类别: Ragdolls', ha='center', va='center', fontsize=11)
    ax.text(14.5, 3, '置信度: 0.9822', ha='center', va='center', fontsize=11)
    ax.text(14.5, 2.5, '🎯 98.22%准确率', ha='center', va='center', fontsize=10, color='red')
    
    # 添加箭头连接
    arrows = [
        # 宿主机到Docker挂载
        ((3.5, 8), (5, 8)),
        ((1.25, 6), (6.5, 7)),
        ((2.8, 6), (6.5, 7)),
        
        # Docker挂载到容器
        ((8, 8), (9, 8)),
        
        # 容器内部流程
        ((10.5, 6), (10.5, 5.5)),
        ((10.5, 3.5), (10.5, 3)),
        
        # 推理到结果
        ((12, 2), (13, 3)),
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # 添加数据流标签
    ax.text(4.25, 8.5, '文件映射', ha='center', va='center', fontsize=10, color='blue')
    ax.text(8.5, 8.5, '卷挂载', ha='center', va='center', fontsize=10, color='orange')
    ax.text(11.5, 5.75, '读取处理', ha='center', va='center', fontsize=10, color='purple')
    ax.text(11.5, 3.25, '模型推理', ha='center', va='center', fontsize=10, color='green')
    ax.text(12.5, 2.5, '输出', ha='center', va='center', fontsize=10, color='red')
    
    # 设置坐标轴
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 11)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 添加标题
    plt.title('🐱 猫类别识别系统 - 图片传递流程图', fontsize=16, weight='bold', pad=20)
    
    # 添加说明文字
    explanation = """
    数据流向说明：
    1. 宿主机Mac系统存储原始图片和TFLite模型
    2. Docker卷挂载将文件映射到容器内部
    3. 容器内Python脚本读取并预处理图片
    4. TensorFlow Lite执行推理计算
    5. 输出分类结果和置信度到终端
    """
    
    ax.text(0.5, 0.5, explanation, ha='left', va='bottom', fontsize=11, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/Users/ding/Desktop/NUS-proj/lite/data_flow_diagram.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    create_data_flow_diagram()
    print("✅ 图片传递流程图已生成: data_flow_diagram.png")
