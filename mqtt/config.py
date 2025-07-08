# 系统配置文件
# 请根据您的实际环境修改以下配置

# 网络配置
RASPBERRY_PI_IP = "192.168.148.103"  # 树莓派IP地址
MAC_IP = "192.168.148.100"           # Mac IP地址

# MQTT配置
MQTT_BROKER_IP = "192.168.148.103"   # MQTT服务器IP (通常是树莓派)
MQTT_PORT = 1883                     # MQTT端口
MQTT_WEBSOCKET_PORT = 9001           # MQTT WebSocket端口

# 串口配置
SERIAL_PORT = "/dev/ttyACM0"         # Arduino串口
SERIAL_BAUD = 9600                   # 波特率

# 摄像头配置
MJPG_STREAMER_PORT = 8080            # MJPG流媒体端口
CAMERA_RESOLUTION = "320x240"        # 摄像头分辨率
CAMERA_FPS = 25                      # 帧率

# Web服务器配置
WEB_SERVER_PORT = 8888               # Mac端Web服务器端口

# 安全配置
AUTO_BRAKE_DISTANCE = 20             # 自动刹车距离 (cm)
SENSOR_UPDATE_INTERVAL = 0.1         # 传感器更新间隔 (秒)
CONNECTION_TIMEOUT = 30              # 连接超时时间 (秒)

# 文件路径
PHOTO_DIRECTORY = "./photos"         # 照片保存目录
LOG_DIRECTORY = "./logs"             # 日志目录
