import torch
import torch.nn as nn
import torch.nn.functional as F
import paho.mqtt.client as mqtt
import json
import numpy as np

MQTT_BROKER = "localhost"  
MQTT_PORT = 1883
MQTT_USERNAME = 'main'
MQTT_PASSWORD = 'sws3009-20'
COMMAND_INTERVAL = 0.01
VIDEO_RESOLUTION = (400, 300)

MQTT_TOPIC_CENTRE = 'robot/object-detection'
MQTT_TOPIC_COMMAND = 'robot/command'

'''
object = {
    x: pos_x
    y: pos_y
    area: area
}
'''

def on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[*] MQTT connection successful")
        client.subscribe(MQTT_TOPIC_CENTRE)
    else:
        print(f"[!] MQTT connection failed: {rc}")

def init_mqtt():
    global mqtt_client
    
    try:
        mqtt_client = mqtt.Client()
        mqtt_client.on_connect = on_mqtt_connect
        mqtt_client.on_message = on_mqtt_message
        
        if MQTT_USERNAME and MQTT_PASSWORD:
            mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
        
        print(f"[*] MQTT has connected to {MQTT_BROKER}:{MQTT_PORT}")
        return True
        
    except Exception as e:
        print(f"[!] MQTT failed to initialize: {e}")
        return False
    
global cur_command
global timeout
timeout = 0
def simple_control_command(pos_x):
    if pos_x == -1000 and timeout < 6: 
        timeout = timeout + 1
        return 'R' if cur_command == 'L' else 'L'
    if timeout >= 6:
        print("Cat not found! Stop searching for cats...")
        return 'S'
    timeout = 0
    if(np.abs(pos_x - VIDEO_RESOLUTION[1] / 2) < VIDEO_RESOLUTION[1]/6):
        return 'F'
    elif(pos_x < VIDEO_RESOLUTION[1] / 2):
        cur_command = 'L'
        return 'L'
    else:
        cur_command = 'R'
        return 'R'

def on_mqtt_message(client, userdata, msg):
    try:
        topic = msg.topic
        payload = msg.payload.decode('utf-8')
        try:
            pos = json.loads(payload)
            pos_x = pos['x']
            command = simple_control_command(pos_x)
            mqtt_client.publish(MQTT_TOPIC_COMMAND, command)
            
        except Exception as e:
            print("payload parsing error!")
    except:
        print("MQTT message decoding error!")

def main():
   init_mqtt()
   while(True):
       pass
        
if __name__ == "__main__":
    main()

