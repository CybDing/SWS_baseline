#!/usr/bin/env python3
"""
Docker图片传递机制演示脚本
模拟展示图片从宿主机到容器的完整过程
"""

import os
import sys
from pathlib import Path

def print_separator(title):
    """打印分隔线"""
    print("\n" + "="*60)
    print(f"🔍 {title}")
    print("="*60)

def demonstrate_file_mapping():
    """演示文件映射过程"""
    print_separator("Docker卷挂载文件映射演示")
    
    # 宿主机路径
    host_paths = {
        "models": "/Users/ding/Desktop/NUS-proj/lite/",
        "images": "/Users/ding/Desktop/NUS-proj/data/test/"
    }
    
    # 容器内路径
    container_paths = {
        "models": "/app/models/",
        "images": "/app/test_images/"
    }
    
    print("📁 宿主机 → 容器 文件映射:")
    print(f"   宿主机模型目录: {host_paths['models']}")
    print(f"   容器内映射路径: {container_paths['models']}")
    print(f"   Docker挂载命令: -v \"$(pwd)\":/app/models")
    print()
    print(f"   宿主机图片目录: {host_paths['images']}")
    print(f"   容器内映射路径: {container_paths['images']}")
    print(f"   Docker挂载命令: -v \"$(pwd)/../data/test\":/app/test_images")

def demonstrate_image_processing():
    """演示图片处理流程"""
    print_separator("图片处理流程演示")
    
    print("🖼️ 图片处理步骤:")
    print("1. 📂 容器内读取: /app/test_images/rd.jpg")
    print("2. 🔄 PIL加载: Image.open(image_path).convert('RGB')")
    print("3. 📏 尺寸调整: image.resize((512, 512))  # 或 (224, 224)")
    print("4. 🔢 数组转换: np.array(image, dtype=np.float32)")
    print("5. 📊 归一化: image_array / 255.0  # [0,255] → [0,1]")
    print("6. 📦 批次维度: np.expand_dims(image_array, axis=0)  # (H,W,C) → (1,H,W,C)")
    print()
    print("📐 数据形状变化:")
    print("   原始图片: (高度, 宽度, 3通道)")
    print("   调整后: (512, 512, 3)")
    print("   归一化: (512, 512, 3) 数值范围 [0,1]")
    print("   批次维度: (1, 512, 512, 3)")

def demonstrate_model_inference():
    """演示模型推理过程"""
    print_separator("TensorFlow Lite推理演示")
    
    print("🧠 模型推理步骤:")
    print("1. 📥 加载模型: tflite.Interpreter(model_path)")
    print("2. 🔧 分配张量: interpreter.allocate_tensors()")
    print("3. 📋 获取输入输出信息:")
    print("   - input_details = interpreter.get_input_details()")
    print("   - output_details = interpreter.get_output_details()")
    print("4. 📊 设置输入数据: interpreter.set_tensor(input_index, image_array)")
    print("5. ⚡ 执行推理: interpreter.invoke()")
    print("6. 📈 获取结果: interpreter.get_tensor(output_index)")
    print()
    print("🎯 结果处理:")
    print("   预测数组: [0.0123, 0.0456, 0.9822, 0.0234, 0.0365]")
    print("   最大索引: np.argmax(predictions) = 2")
    print("   对应类别: CLASS_NAMES[2] = 'Ragdolls'")
    print("   置信度: predictions[2] = 0.9822")

def demonstrate_docker_command():
    """演示Docker命令结构"""
    print_separator("Docker命令结构解析")
    
    command_parts = [
        ("docker run", "运行Docker容器"),
        ("--rm", "运行结束后自动删除容器"),
        ("-v \"$(pwd)\":/app/models", "挂载当前目录到容器/app/models"),
        ("-v \"$(pwd)/../data/test\":/app/test_images", "挂载测试图片目录"),
        ("cat-classifier", "使用的Docker镜像名称"),
        ("--model /app/models/model.tflite", "指定模型文件路径(容器内)"),
        ("--image /app/test_images/rd.jpg", "指定输入图片路径(容器内)")
    ]
    
    print("🐳 Docker命令拆解:")
    for i, (command, description) in enumerate(command_parts, 1):
        print(f"{i}. {command}")
        print(f"   💡 {description}")
        print()

def demonstrate_complete_flow():
    """演示完整数据流"""
    print_separator("完整数据流演示")
    
    flow_steps = [
        ("🖥️ 宿主机Mac", "存储原始图片 rd.jpg 和模型 model.tflite"),
        ("📦 Docker挂载", "将宿主机文件映射到容器内部"),
        ("🐧 Linux容器", "容器内可以访问映射的文件"),
        ("🔄 图片预处理", "PIL读取 → 调整尺寸 → 归一化 → NumPy数组"),
        ("🧠 模型推理", "TensorFlow Lite执行推理计算"),
        ("📊 结果处理", "argmax获取类别，提取置信度"),
        ("🎯 类别映射", "索引映射到具体类别名称"),
        ("💻 输出显示", "结果返回到宿主机终端显示")
    ]
    
    print("🔄 端到端数据流:")
    for i, (stage, description) in enumerate(flow_steps, 1):
        print(f"{i}. {stage}")
        print(f"   {description}")
        if i < len(flow_steps):
            print("   ⬇️")
    
    print("\n🎉 最终输出:")
    print("   类别: Ragdolls")
    print("   置信度: 0.9822")
    print("   INFO: Created TensorFlow Lite XNNPACK delegate for CPU.")

def main():
    """主函数"""
    print("🐱 Docker猫类别识别系统 - 原理演示")
    print("="*60)
    
    demonstrate_file_mapping()
    demonstrate_image_processing()
    demonstrate_model_inference()
    demonstrate_docker_command()
    demonstrate_complete_flow()
    
    print("\n" + "="*60)
    print("✅ 原理演示完成！")
    print("💡 关键优势:")
    print("   • 跨平台兼容性 (Mac开发 → Linux部署)")
    print("   • 环境隔离 (避免依赖冲突)")
    print("   • 轻量级部署 (TensorFlow Lite)")
    print("   • 高效推理 (XNNPACK CPU加速)")
    print("="*60)

if __name__ == "__main__":
    main()
