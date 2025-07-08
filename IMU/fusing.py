import numpy as np
# import flask
import matplotlib.pyplot as plt
import time

class IMU:
    def __init__(self, ):
        self.topic = 'IMU'
        self.YAWangle = 0
        self.past_EncoderValue = list()
        self.past_YAW_angular_acc = list()
        self.encoder_gain = 1
        self.speed = None
        self.turning = False
        self.positions = [(0, 0)] # cur_pos, last_pos
        self.timestep = 0
        self.time = None
        _ = plt.figure(figsize=(8,6))
        
    
    def restart(self, ):
        self.YAWangle = 0
        self.past_EncoderValue = list(0)
        self.past_YAW_angular_speed = list()
        self.time = time.time()
        self.speed = None
        self.turning = False
        self.positions = [(0, 0)]
        self.index = 0
        plt.plot(self.positions)
        plt.show()

        while(True):
            self.receive_all()
            self.filter_angle_acc()
            self.filter_angle_acc()
            self._update_position()


    def _filter_angle_acc(self, ):
        '''
        Use Karman filter to filter IMU yaw angles
        '''
        if self.turning == False: return 0
        # Help me filter the angular acc


    def _filter_encoder(self, mode=3):
        '''
        Return the speed of the car using averaging method
        '''
        if self.turning == True: return 0
        assert mode >= 1 and isinstance(mode, int or np.int32)
        return np.sum(self.past_EncoderValue[-mode:])/float(self.mode)

    def _receive_angular_acc(self, ):
        '''
        Implementing data receiving from arduino on raspberry
        '''

    def _receive_encoder(self, ):
        '''
        Implement data receiving from arduino on raspberry
        '''
    
    def _receive_command(self, ):
        '''
        Implement command receiving(turning left or right),
        '''

    def receive_all(self, ):
        self._receive_command()
        self._receive_angular_acc()
        self._receive_encoder()
        cur_time = time.time()
        self.timestep = cur_time - self.time
        self.time = cur_time

    # def post_cur_position(self,):
    #     '''
    #     Implement a flask client, that could
    #     '''
    def _update_position(self, ):
        if self.turning == True: 
            # should update angle here?
            return

        R = np.array([
            [np.cos(self.YAWangle), -np.sin(self.YAWangle)],
            [np.sin(self.YAWangle), np.cos(self.YAWangle)]
        ])
        
        self.position[-1] = self.position[-1] + self.speed * R[:, 1]
        self.index = self.index + 1

    def _draw_cur_pos(self, ):
        if(self.index % 5 == 0):
            plt.plot(self.positions)
        