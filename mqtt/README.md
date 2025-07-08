# 🚗 树莓派小车远程控制系统

## 系统概述

这是一个完整的树莓派小车远程控制系统，支持通过Web界面进行远程控制，实时获取传感器数据，并提供拍照功能。系统采用MQTT协议进行Mac和树莓派之间的通信。

## 系统架构

```
Mac端 (Web界面) <---> MQTT服务器 <---> 树莓派 <---> Arduino小车
                                              |
                                         摄像头模块
```

## 功能特性

- ✅ **实时视频流**: 通过MJPG Streamer提供实时视频流
- ✅ **远程控制**: 支持键盘(WASD)和鼠标控制
- ✅ **传感器监控**: 实时显示超声波传感器、编码器、陀螺仪数据
- ✅ **拍照功能**: 支持远程拍照并保存
- ✅ **安全保护**: 前方障碍物检测自动刹车
- ✅ **状态监控**: 实时显示连接状态和设备状态

## 安装和配置

### 树莓派端设置

1. **安装依赖包**
   ```bash
   # 更新系统
   sudo apt update && sudo apt upgrade -y
   
   # 安装MQTT服务器
   sudo apt install mosquitto mosquitto-clients -y
   
   # 安装MJPG Streamer
   sudo apt install cmake libjpeg-dev -y
   git clone https://github.com/jacksonliam/mjpg-streamer.git
   cd mjpg-streamer/mjpg-streamer-experimental
   make
   sudo make install
   
   # 安装Python依赖
   pip3 install -r requirements_raspberry_pi.txt
   ```

2. **配置MQTT服务器**
   ```bash
   # 启动MQTT服务器
   sudo systemctl start mosquitto
   sudo systemctl enable mosquitto
   
   # 配置WebSocket支持
   sudo nano /etc/mosquitto/mosquitto.conf
   # 添加以下内容:
   listener 1883
   protocol mqtt
   
   listener 9001
   protocol websockets
   ```

3. **运行树莓派控制器**
   ```bash
   chmod +x start_raspberry_pi.sh
   ./start_raspberry_pi.sh
   ```

### Mac端设置

1. **安装依赖包**
   ```bash
   # 安装MQTT客户端 (可选，如果需要本地MQTT服务器)
   brew install mosquitto
   
   # 安装Python依赖
   pip3 install -r requirements_mac.txt
   ```

2. **配置网络**
   - 修改 `config.py` 中的IP地址配置
   - 确保Mac和树莓派在同一网络中

3. **运行Mac客户端**
   ```bash
   chmod +x start_mac.sh
   ./start_mac.sh
   ```

## 使用方法

### 1. 启动系统

**树莓派端:**
```bash
./start_raspberry_pi.sh
```

**Mac端:**
```bash
./start_mac.sh
```

### 2. 控制小车

**键盘控制:**
- `W` - 前进
- `S` - 后退
- `A` - 左转
- `D` - 右转
- `X` - 拍照

**鼠标控制:**
- 点击方向按钮进行控制
- 点击拍照按钮进行拍照

### 3. 监控数据

Web界面会实时显示：
- 前方、左侧、右侧超声波距离
- 左轮、右轮编码器数据
- 角加速度数据
- 系统连接状态

## 文件结构

```
mqtt_new/
├── raspberry_pi_controller.py  # 树莓派控制器
├── mac_client.py              # Mac端客户端
├── web_control.html           # Web控制界面
├── config.py                  # 系统配置
├── requirements_raspberry_pi.txt  # 树莓派依赖
├── requirements_mac.txt       # Mac依赖
├── start_raspberry_pi.sh      # 树莓派启动脚本
├── start_mac.sh              # Mac启动脚本
└── README.md                 # 使用说明
```

## 网络配置

确保以下端口可以访问：
- `1883` - MQTT协议端口
- `9001` - MQTT WebSocket端口
- `8080` - MJPG Streamer端口
- `8888` - Mac端Web服务器端口

## 故障排除

### 1. 视频流无法显示
- 检查MJPG Streamer是否正常运行
- 确认摄像头连接正常
- 检查防火墙设置

### 2. 控制命令无响应
- 检查MQTT服务器状态
- 确认串口连接正常
- 检查Arduino代码是否正确

### 3. 传感器数据不更新
- 检查串口数据格式
- 确认传感器硬件连接
- 查看系统日志

## 系统要求

**树莓派:**
- Raspberry Pi 4 或更高版本
- Raspbian OS
- Python 3.7+
- 摄像头模块
- Arduino控制器

**Mac:**
- macOS 10.14+
- Python 3.7+
- 现代浏览器 (Chrome, Firefox, Safari)

## 安全注意事项

1. **网络安全**: 在公共网络中使用时请配置MQTT认证
2. **物理安全**: 确保小车在安全环境中运行
3. **自动保护**: 系统会在检测到前方障碍物时自动停止

## 技术支持

如果遇到问题，请检查：
1. 网络连接状态
2. 系统日志信息
3. 硬件连接情况
4. 配置文件设置

## 更新日志

- v1.0.0: 初始版本发布
  - 基本远程控制功能
  - 实时视频流
  - 传感器数据监控
  - 拍照功能

## 许可证

此项目遵循MIT许可证。
