import numpy as np

from sim_env import Bullet
from pid import PID
from sim_env import PidEnv
from SAC import SAC
from agent import Car
from tensorboardX import  SummaryWriter
import collections
import argparse
import configparser
import torch
import os
import time
import random
import shutil
import pickle
import matplotlib.pyplot as plt


def main():


    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--config', type=str, default='config.config')
    parser.add_argument('--output_dir', type=str, default='data/model_2')
    parser.add_argument('--seed', type=int,default=777)
    parser.add_argument('--gpu', default=True, action='store_true')
    args = parser.parse_args()

    # seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    #config
    config = configparser.RawConfigParser()
    config.read(args.config)
    gamma=config.getfloat('train','gamma')
    buffer_maxlen=config.getint('train','buffer_maxlen')
    learn_rate=config.getfloat('train','learn_rate')
    batch_size=config.getint('train','batch_size')
    episodes=config.getint('train','episodes')

    #init
    writer = SummaryWriter()
    car=Car(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    bullet=Bullet(car,config,True)
    x_pid=PID()
    y_pid=PID()
    w_pid=PID()
    env=PidEnv(bullet=bullet,car=car,x_pid=x_pid,y_pid=y_pid,w_pid=w_pid,config=config)
    policy=SAC(env,gamma,config,buffer_maxlen,device,writer,q_lr=learn_rate,actor_lr=learn_rate)

    # make dir/path and load model
    make_new_dir=True
    model_path = os.path.join(args.output_dir, 'model.pth')
    rewards_path=os.path.join(args.output_dir,'rewards.pkl')
    success_path=os.path.join(args.output_dir,'success.pkl')
    error_path=os.path.join(args.output_dir,'error.pkl')
    vel_path=os.path.join(args.output_dir,'vel.pkl')
    w_path=os.path.join(args.output_dir,'w.pkl')

    if os.path.exists(args.output_dir):
        key = input('Output directory already exists! Overwrite the folder? (y/n)')

        if key == 'y' :
            shutil.rmtree(args.output_dir)
        else:
            make_new_dir = False
            policy.load_model(model_path)
    if make_new_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)




    #connect pybullet and start sim
    bullet.connect()
    bullet.init()
    #bullet.start()

    print('Using device: %s', device)
    #reinforcement learning
    max_return = 0.0
    return_mean=0.0
    return_list=[]
    error_list=[]
    success_list=[]
    vel_list=[]
    w_list=[]
    success_que=collections.deque(maxlen=30)
    updates=0
    for episode in range(episodes):
        score = 0
        errors=[]
        vels=[]
        w=[]
        state=env.reset()
        time.sleep(0.5)
        flag=False

        for i in range(bullet.sample_number+10):

            action = policy.select_action(state)
            next_state, reward,done,error,flag= env.step(action,i)
            policy.buffer.push(state, action, reward, next_state,done)
            state=next_state

            vels.append(next_state[9])
            errors.append(error)
            w.append(next_state[10])

            score += reward
            if done:
                break
            #if len(policy.buffer)>batch_size:
                #if bullet.sample_T!=0.18:
                    #bullet.sample_T=0.18
                #policy.update_parameters(batch_size,updates)
            updates+=1


        if flag:
            success_que.append(1)
        else:
            success_que.append(0)
        return_mean = (return_mean * (episode) + score) / (episode + 1)
        rate=sum(success_que)/len(success_que)
        print("episode:{}, Return:{},SuccessRate:{}, buffer_capacity:{}".format(episode, score,rate, len(policy.buffer)))
        writer.add_scalar('return',return_mean,episode )
        writer.add_scalar('success_rate', rate, episode)
        #print('error',errors)
        #ax1=plt.subplot(2,2,1)
        #plt.plot(np.arange(147),errors)
        #plt.legend()
        #axs=plt.subplot(2,2,2)
        #plt.plot(bullet.targetx_list[:-3],bullet.targety_list[:-3])
        #plt.plot(env.x,env.y)
        #plt.legend()


        #plt.savefig('errors')
        #plt.show()

        # save error of one episode
        if score > max_return and flag:
            max_return = score
            error_list = errors
            with open(error_path, "wb") as f:
                pickle.dump(error_list, f, pickle.HIGHEST_PROTOCOL)

        return_list.append(return_mean)
        success_list.append(rate)
        vel_list.append(vels)
        w_list.append(w)


        # save model and data
        if episode>100 and episode%20==0:
                policy.save_model(model_path)
                with open(rewards_path, "wb") as f:
                    pickle.dump(return_list, f, pickle.HIGHEST_PROTOCOL)
                with open(success_path, "wb") as f:
                    pickle.dump(success_list, f, pickle.HIGHEST_PROTOCOL)

                with open(vel_path , "wb") as f:
                    pickle.dump(vel_list, f, pickle.HIGHEST_PROTOCOL)
                with open(w_path,"wb") as f:
                    pickle.dump(w_list,f,pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    main()
