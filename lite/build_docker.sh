#!/bin/bash

# 构建Docker镜像
echo "🏗️  构建Docker镜像..."
docker build -t cat-classifier .

# 检查构建是否成功
if [ $? -eq 0 ]; then
    echo "✅ Docker镜像构建成功！"
    
    # 运行测试示例
    echo "🧪 运行测试示例..."
    echo "使用以下命令测试："
    echo ""
    echo "docker run --rm -v \$(pwd):/app/models -v \$(pwd)/../data/test:/app/test_images cat-classifier --model /app/models/CatClassifier_512V2_2.tflite --image /app/test_images/rd.jpg"
    echo ""
    echo "或者使用简化脚本："
    echo "./test_docker.sh"
else
    echo "❌ Docker镜像构建失败！"
    exit 1
fi
