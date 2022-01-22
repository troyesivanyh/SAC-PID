import  numpy as  np
import math
class Agent():
    def __init__(self,):
        pass
class Car(Agent):
    def __init__(self,config):
        super(Car, self).__init__()
        self.wheel_r=config.getfloat('agent','wheel_r')
        self.v_max=config.getfloat('agent','v_max')
        self.v_min=config.getfloat('agent','v_min')
        self.car_radius=config.getfloat('agent','car_radius')
        self.goal=(0.0,0.0)
        self.width=config.getfloat('agent','width')
        self.length=config.getfloat('agent','length')
        x=config.getfloat('agent','init_pos_x')
        y=config.getfloat('agent','init_pos_y')
        self.init_pos=[x,y]
        self.init_yaw=config.getfloat('agent','init_yaw')
