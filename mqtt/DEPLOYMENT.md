# 🚀 部署指南

## 快速开始

### 1. 测试系统功能

在开始完整部署之前，建议先测试Web界面：

```bash
# 在Mac上运行测试服务器
python3 test_server.py
```

然后在浏览器中访问 `http://localhost:8888` 查看Web界面是否正常工作。

### 2. 树莓派部署

#### 2.1 准备工作

1. **更新系统**
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

2. **安装MQTT服务器**
   ```bash
   sudo apt install mosquitto mosquitto-clients -y
   
   # 配置MQTT服务器
   sudo nano /etc/mosquitto/mosquitto.conf
   ```
   
   添加以下内容到配置文件：
   ```
   listener 1883
   protocol mqtt
   
   listener 9001
   protocol websockets
   ```

3. **安装MJPG Streamer**
   ```bash
   sudo apt install cmake libjpeg-dev -y
   git clone https://github.com/jacksonliam/mjpg-streamer.git
   cd mjpg-streamer/mjpg-streamer-experimental
   make
   sudo make install
   ```

#### 2.2 配置网络

1. **查看树莓派IP地址**
   ```bash
   ip addr show wlan0
   ```

2. **修改配置文件**
   编辑 `config.py` 文件，更新以下配置：
   ```python
   RASPBERRY_PI_IP = "你的树莓派IP地址"
   MQTT_BROKER_IP = "你的树莓派IP地址"
   ```

#### 2.3 部署应用

1. **将文件复制到树莓派**
   ```bash
   scp -r mqtt_new/ pi@你的树莓派IP:/home/pi/
   ```

2. **在树莓派上运行**
   ```bash
   cd /home/pi/mqtt_new
   chmod +x start_raspberry_pi.sh
   ./start_raspberry_pi.sh
   ```

### 3. Mac端部署

#### 3.1 安装依赖

1. **安装Python依赖**
   ```bash
   pip3 install -r requirements_mac.txt
   ```

2. **安装MQTT客户端（可选）**
   ```bash
   brew install mosquitto
   ```

#### 3.2 配置和运行

1. **更新配置文件**
   修改 `config.py` 中的IP地址：
   ```python
   RASPBERRY_PI_IP = "你的树莓派IP地址"
   MAC_IP = "你的Mac IP地址"
   ```

2. **启动Mac客户端**
   ```bash
   ./start_mac.sh
   ```

### 4. 验证部署

#### 4.1 检查服务状态

**树莓派端:**
```bash
# 检查MQTT服务器
sudo systemctl status mosquitto

# 检查MJPG Streamer
ps aux | grep mjpg_streamer

# 检查Python控制器
ps aux | grep python
```

**Mac端:**
```bash
# 检查Web服务器
curl http://localhost:8888/data
```

#### 4.2 测试连接

1. **测试视频流**
   - 在浏览器中访问：`http://你的树莓派IP:8080/?action=stream`

2. **测试控制界面**
   - 在浏览器中访问：`http://localhost:8888`

3. **测试MQTT连接**
   ```bash
   # 在树莓派上测试
   mosquitto_pub -h localhost -t "robot/command" -m "F"
   
   # 在Mac上测试
   mosquitto_sub -h 你的树莓派IP -t "robot/data"
   ```

### 5. 常见问题解决

#### 5.1 视频流问题

**问题**: 视频流无法显示
**解决方案**:
```bash
# 检查摄像头
vcgencmd get_camera

# 重启MJPG Streamer
sudo pkill mjpg_streamer
mjpg_streamer -i "input_libcamera.so -f 25 -r 320x240" -o "output_http.so -p 8080"
```

#### 5.2 MQTT连接问题

**问题**: MQTT连接失败
**解决方案**:
```bash
# 检查MQTT服务器状态
sudo systemctl status mosquitto

# 检查端口是否开放
sudo netstat -tuln | grep 1883

# 重启MQTT服务器
sudo systemctl restart mosquitto
```

#### 5.3 串口问题

**问题**: 串口连接失败
**解决方案**:
```bash
# 检查串口设备
ls -l /dev/ttyACM*

# 添加用户权限
sudo usermod -a -G dialout pi

# 重启系统
sudo reboot
```

### 6. 性能优化

#### 6.1 系统优化

1. **GPU内存分配**
   ```bash
   sudo raspi-config
   # Advanced Options > Memory Split > 128
   ```

2. **禁用不必要的服务**
   ```bash
   sudo systemctl disable bluetooth
   sudo systemctl disable cups
   ```

#### 6.2 网络优化

1. **设置静态IP**
   ```bash
   sudo nano /etc/dhcpcd.conf
   ```
   
   添加：
   ```
   interface wlan0
   static ip_address=192.168.1.100/24
   static routers=192.168.1.1
   static domain_name_servers=192.168.1.1
   ```

2. **优化WiFi**
   ```bash
   sudo nano /etc/wpa_supplicant/wpa_supplicant.conf
   ```
   
   添加：
   ```
   country=CN
   ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
   update_config=1
   
   network={
       ssid="你的WiFi名称"
       psk="你的WiFi密码"
   }
   ```

### 7. 监控和日志

#### 7.1 系统监控

1. **创建监控脚本**
   ```bash
   #!/bin/bash
   # monitor.sh
   
   echo "=== 系统状态 ==="
   date
   uptime
   
   echo "=== 内存使用 ==="
   free -h
   
   echo "=== 磁盘使用 ==="
   df -h
   
   echo "=== 网络连接 ==="
   ip addr show wlan0
   
   echo "=== 进程状态 ==="
   ps aux | grep -E "(python|mjpg_streamer|mosquitto)" | grep -v grep
   ```

2. **设置定时任务**
   ```bash
   crontab -e
   # 添加：每5分钟检查一次
   */5 * * * * /home/pi/mqtt_new/monitor.sh >> /home/pi/system_monitor.log
   ```

#### 7.2 日志管理

1. **应用日志**
   ```python
   import logging
   
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(levelname)s - %(message)s',
       handlers=[
           logging.FileHandler('/home/pi/mqtt_new/logs/app.log'),
           logging.StreamHandler()
       ]
   )
   ```

2. **系统日志**
   ```bash
   # 查看系统日志
   journalctl -u mosquitto -f
   
   # 查看网络日志
   tail -f /var/log/daemon.log
   ```

### 8. 备份和恢复

#### 8.1 配置备份

```bash
# 备份配置文件
tar -czf config_backup_$(date +%Y%m%d).tar.gz config.py *.txt *.sh

# 备份照片
rsync -av photos/ backup/photos/
```

#### 8.2 系统备份

```bash
# 创建系统镜像
sudo dd if=/dev/mmcblk0 of=/path/to/backup.img bs=4M status=progress

# 压缩镜像
gzip backup.img
```

### 9. 自动启动

#### 9.1 创建系统服务

1. **创建服务文件**
   ```bash
   sudo nano /etc/systemd/system/robot-controller.service
   ```
   
   内容：
   ```ini
   [Unit]
   Description=Robot Controller Service
   After=network.target
   
   [Service]
   Type=simple
   User=pi
   WorkingDirectory=/home/pi/mqtt_new
   ExecStart=/usr/bin/python3 raspberry_pi_controller.py
   Restart=always
   RestartSec=5
   
   [Install]
   WantedBy=multi-user.target
   ```

2. **启用服务**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable robot-controller.service
   sudo systemctl start robot-controller.service
   ```

#### 9.2 开机自启动

```bash
# 编辑开机启动脚本
sudo nano /etc/rc.local

# 在 'exit 0' 之前添加：
cd /home/pi/mqtt_new
python3 raspberry_pi_controller.py &
```

### 10. 故障排除清单

- [ ] 检查网络连接
- [ ] 验证IP地址配置
- [ ] 测试MQTT服务器
- [ ] 检查串口权限
- [ ] 验证摄像头功能
- [ ] 测试Web界面
- [ ] 检查防火墙设置
- [ ] 验证Python依赖
- [ ] 检查系统日志
- [ ] 测试Arduino连接

完成以上步骤后，您的树莓派小车远程控制系统应该可以正常运行了！
