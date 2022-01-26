import pybullet as p
import pybullet_data
from pid import PID
import random
from numpy.linalg import norm
import numpy as np
from agent import Car
import time
import math
import torch
import matplotlib.pyplot as plt

#configure bullent, car and map
'''
problem2: sim without gui
'''
class Bullet():
    def __init__(self,car,config,gui=True):
        self.gui=gui
        self.palneId=None
        self.wheel=None
        self.robotId=None
        self.sample_T=config.getfloat('env','sample_T')
        self.sample_number=config.getint('env','sample_number')
        self.targetx_list=[0.0]
        self.targety_list=[0.0]
        self.targettheta_list=[]
        self.car=car


    #conect pyhsics engine
    def connect(self):
        if self.gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -10)


    # init car and map
    def init(self):
        self.palneId = p.loadURDF("plane.urdf")
        # init position
        cubeStartPos = [self.car.init_pos[0], self.car.init_pos[1], 0]
        # init orientation
        cubeStartOrientation = p.getQuaternionFromEuler([0, 0, self.car.init_yaw-math.pi/2])
        self.robotId= p.loadURDF("mecanum_simple.urdf", cubeStartPos, cubeStartOrientation,flags=p.URDF_USE_INERTIA_FROM_FILE)
        #print(self.robotId)
        self.wheel= {'BR':0,'FR':20, 'BL':40, 'FL':60}
        self.sample()



        #print(len(self.targetx_list),len(self.targettheta_list))

    def reset(self):
        #print(self.robotId)
        self.set_vel()
        cubeStartOrientation = p.getQuaternionFromEuler([0, 0, self.car.init_yaw-math.pi/2])
        p.resetBasePositionAndOrientation(self.robotId,[self.car.init_pos[0],self.car.init_pos[1],0],cubeStartOrientation)
        for i in range(240):
            p.stepSimulation()
    # draw the line:just considerate straight line along y axis
    def sample(self):
        for i in range(self.sample_number):
            delt_y= random.uniform(0.05,0.1)
            y=self.targety_list[-1]+delt_y
            x=math.sin(y * 0.2) * y / 2.0+math.cos(y)*y/5

            theta= math.atan2(y-self.targety_list[-1], x-self.targetx_list[-1])

            self.targettheta_list.append(theta)
            self.targety_list.append(y)
            self.targetx_list.append(x)
            #errors.append(norm((self.targetx_list[i+1]-self.targetx_list[i],self.targety_list[i+1]-self.targety_list[i])))


            p.addUserDebugLine([self.targetx_list[i],self.targety_list[i],0.0],[self.targetx_list[i+1],self.targety_list[i+1],0.0],lineColorRGB=[1, 0, 0], lifeTime=float('inf'), lineWidth=3)
        self.car.goal=[self.targetx_list[-4],self.targety_list[-4]]
        print('goal',self.car.goal)
        #print(max(errors))
        #print(self.car.goal)


   # set velocity for each wheel
    def set_vel(self,x_vel=0.0,y_vel=0.0,angel_vel=0.0):
        '''
        :param x_vel: it has some prblem about car run along axis x
        :param y_vel: unit:m/s
        :param angel_vel: uniy: hudu/s
        :param a: the distance along axis x between the center of the car and the center of wheel
        :param b: the distance along axis y between the center of the car and the center of wheel
        '''
        b=self.car.length/2
        a=self.car.width/2
        # wheel linear vel
        fr_vel=-x_vel+y_vel+angel_vel*(a+b)
        fl_vel=x_vel+y_vel-angel_vel*(a+b)
        bl_vel=-x_vel+y_vel-angel_vel*(a+b)
        br_vel=x_vel+y_vel+angel_vel*(a+b)
        # whell angel vel
        fr_vel/=self.car.wheel_r
        fl_vel/=self.car.wheel_r
        bl_vel/=self.car.wheel_r
        br_vel/=self.car.wheel_r


        p.setJointMotorControl2(bodyUniqueId=self.robotId,jointIndex=self.wheel['BR'],controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=br_vel,force=500)
        p.setJointMotorControl2(bodyUniqueId=self.robotId, jointIndex=self.wheel['FR'], controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=fr_vel, force=500)
        p.setJointMotorControl2(bodyUniqueId=self.robotId, jointIndex=self.wheel['BL'], controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=bl_vel, force=500)
        p.setJointMotorControl2(bodyUniqueId=self.robotId, jointIndex=self.wheel['FL'], controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=fl_vel, force=500)

    #start sim
    def start(self):
        p.setRealTimeSimulation(1)
      # p.stepSimulation()

    # ger current position and orientation of car
    def get_state(self):
        pos,ori=p.getBasePositionAndOrientation(self.robotId)
        pos=[pos[0],pos[1]]
        ori = p.getEulerFromQuaternion(ori)
        yaw=ori[2]+math.pi/2
        return pos,yaw

    def disconnect(self):
        p.disconnect()

# the env about use SAC to train PID
# pid1:vel pid
# pid2:angel_vel pid
class PidEnv():
    def __init__(self,bullet,car,x_pid,y_pid,w_pid,config):
        self.car=car
        self.x_pid=x_pid
        self.y_pid=y_pid
        self.w_pid=w_pid
        self.bullet=bullet
        self.state_dim=config.getint('env','state_dim')
        self.action_dim=config.getint('env','action_dim')

        self.max_error=config.getfloat('env','max_error')
        # limit the output range of PID param
        self.PIDpara_min=config.getfloat('env','pidpara_min')
        self.PIDpara_max=config.getfloat('env','pidpara_max')
        self.x=[]
        self.y=[]


    # 转换矩阵
    # Trans=[[cos(theta) sin(theta) 0] [-sin(theta) cos(theta) 0] [0 0 1]]
    def transfermor(self, error_x, error_y, yaw):
        ex = math.cos(yaw) * error_x + math.sin(yaw) * error_y
        ey = -1 * math.sin(yaw) * error_x + math.cos(yaw) * error_y
        return ex, ey

    #calcu error for pid
    def compute_error(self, pos, yaw,i):
        theta_error=self.bullet.targettheta_list[i]-yaw
        error_x=self.bullet.targetx_list[i+1]-pos[0]
        error_y=self.bullet.targety_list[i+1]-pos[1]
        ex,ey=self.transfermor(error_x,error_y,yaw)

        return  ex,ey,theta_error



    def reward(self,next_set,next_pos):
        '''total_y_error = self.car.goal[1] - self.car.init_pos[1]
        now_y_error = self.car.goal[1] - next_pos[1]
        relative_error = now_y_error / total_y_error
        relative_error = np.clip(relative_error, 0, 1)
        reward1 = (1 - relative_error)*1
        error_dis = norm((next_set[0] - next_pos[0], next_set[1] - next_pos[1]))
        if error_dis<=0.1:
            return 1.0+reward1
        elif 0.1<error_dis<self.max_error:
            return -2.5*error_dis+1.25+reward1
        else:
            return -10'''
        error_goal=norm((next_pos[0]-self.car.goal[0],next_pos[1]-self.car.goal[1]))
        error=norm((next_set[0] - next_pos[0], next_set[1] - next_pos[1]))
        if error_goal<=self.car.car_radius:
            return 3
        elif error>self.max_error:
            return -10
        elif 0<=error<=self.max_error:
            return -4*error+1



    # update for PID with RL
    def step(self,action,i):

        done = 0
        success=False
        #print(action)

        self.x_pid.set(action[0], action[1], action[2])
        self.y_pid.set(action[3], action[4], action[5])
        self.w_pid.set(action[6], action[7], action[8])

        pos,yaw=self.bullet.get_state()

        ex,ey,theta_error=self.compute_error(pos,yaw,i)
        v = self.x_pid.update(ex)
        w = self.y_pid.update(ey) + self.w_pid.update(theta_error)
        v=np.clip(v,self.car.v_min,self.car.v_max)
        w=np.clip(w,-np.pi,np.pi)

        self.bullet.set_vel(y_vel=v,angel_vel=w)
        #time.sleep(self.bullet.sample_T)
        a=time.time()
        for i in range(50):
            a=time.time()

            p.stepSimulation()
            print(time.time()-a)
        next_position,next_yaw=self.bullet.get_state()
        self.x.append(next_position[0])
        self.y.append(next_position[1])

        set_pos=[self.bullet.targetx_list[i+1],self.bullet.targety_list[i+1]]
        reward=self.reward(set_pos,next_position)

        deltx = self.bullet.targetx_list[i+2] - next_position[0]
        delty = self.bullet.targety_list[i+2] - next_position[1]
        deltaO = self.bullet.targettheta_list[i+1] - next_yaw
        deltxx = self.bullet.targetx_list[3+i] - next_position[0]
        deltyy = self.bullet.targety_list[3+i] - next_position[1]
        deltOO = self.bullet.targettheta_list[2+i] -next_yaw
        deltxxx=self.bullet.targetx_list[i+4]-next_position[0]
        deltyyy=self.bullet.targety_list[i+4]-next_position[1]
        deltOOO=self.bullet.targettheta_list[3+i]-next_yaw

        error=norm((set_pos[0]-next_position[0],set_pos[1]-next_position[1]))
        #if norm((self.car.goal[0]-next_position[0],self.car.goal[1]-next_position[1]))<=self.car.car_radius or \
                #error>self.max_error :
        if error>self.max_error or (set_pos[0]==self.car.goal[0] and set_pos[1]==self.car.goal[1]):
            done=1
        if norm((self.car.goal[0]-next_position[0],self.car.goal[1]-next_position[1]))<=self.car.car_radius+0.15:
            success=True



        next_state=[deltx,delty,deltaO,deltxx,deltyy,deltOO,deltxxx,deltyyy,deltOOO,v,w]

        return next_state,reward,done,error,success

    # update for PID without RL
    def step2(self,i):
        pos,yaw=self.bullet.get_state()
        ex,ey,theta_error=self.compute_error(pos,yaw,i)
        v=self.x_pid.update(ex)
        w=self.y_pid.update(ey)+self.w_pid.update(theta_error)


        self.bullet.set_vel(y_vel=v,angel_vel=w)
        time.sleep(self.bullet.sample_T )
        return  v


    def reset(self):
        self.bullet.reset()
        self.x_pid.reset()
        self.y_pid.reset()
        self.w_pid.reset()
        position,yaw=self.bullet.get_state()
        self.x.append(position[0])
        self.y.append(position[1])
        vel=0
        angel_vel=0.0
        deltx=self.bullet.targetx_list[1]-position[0]
        delty=self.bullet.targety_list[1]-position[1]
        deltaO=self.bullet.targettheta_list[0]-yaw
        deltxx=self.bullet.targetx_list[2]-position[0]
        deltyy=self.bullet.targety_list[2]-position[1]
        deltOO=self.bullet.targettheta_list[1]-yaw
        deltxxx=self.bullet.targetx_list[3]-position[0]
        deltyyy=self.bullet.targety_list[3]-position[1]
        deltOOO=self.bullet.targettheta_list[2]-yaw

        return deltx,delty,deltaO,deltxx,deltyy,deltOO,deltxxx,deltyyy,deltOOO,vel,angel_vel

if __name__=='__main__':

    car=Car()
    bullet = Bullet(car)
    bullet.connect()
    bullet.init()
    bullet.start()
    x_pid=PID()
    y_pid=PID()
    w_pid=PID()
    PidEnv=PidEnv(bullet,car,x_pid,y_pid,w_pid)
    pos,yaw=bullet.get_state()
    state=[pos[0],pos[1],yaw]
    vs=[]
    for epoch in range(bullet.sample_number):
        vs.append(PidEnv.step2(epoch))

    plt.plot(vs)
    plt.ylabel('vs')
    plt.xlabel("Episode")
    plt.grid(True)
    plt.show()

















