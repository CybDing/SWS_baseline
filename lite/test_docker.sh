#!/bin/bash

# 检查Docker是否运行
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker未运行，请先启动Docker Desktop"
    exit 1
fi

# 设置变量
MODEL_PATH="CatClassifier_512V2_2.tflite"
TEST_IMAGE="../data/test/rd.jpg"

# 检查模型文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ 模型文件不存在: $MODEL_PATH"
    echo "请确保模型文件在当前目录下"
    exit 1
fi

# 检查测试图片是否存在
if [ ! -f "$TEST_IMAGE" ]; then
    echo "❌ 测试图片不存在: $TEST_IMAGE"
    echo "请检查图片路径"
    exit 1
fi

echo "🚀 开始测试猫类别识别..."
echo "📦 模型文件: $MODEL_PATH"
echo "🖼️  测试图片: $TEST_IMAGE"
echo ""

# 运行Docker容器
docker run --rm \
    -v "$(pwd)":/app/models \
    -v "$(pwd)/../data/test":/app/test_images \
    cat-classifier \
    --model /app/models/$MODEL_PATH \
    --image /app/test_images/$(basename $TEST_IMAGE)

echo ""
echo "✅ 测试完成！"
