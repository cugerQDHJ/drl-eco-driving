
import sys,os
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path

import traci

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from network import Actor, Critic
from memory import ReplayBuffer

from collections import defaultdict

from traffic import *

WEIGHT_DECAY = 0    # L2 weight decay,论文的设置是0.01

class OUNoise(object):
    '''Ornstein–Uhlenbeck
    '''
    def __init__(self, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = 1
        self.low          = -2.0
        self.high         = 2.0
        self.reset()
        
    def reset(self):
        self.obs = np.ones(self.action_dim) * self.mu
        
    def evolve_obs(self):
        x  = self.obs
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.obs = x + dx
        return self.obs
    
    def get_action(self, action, t=0):
        ou_obs = self.evolve_obs()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_obs, self.low, self.high)

class DDPGAgent:
    def __init__(self, state_dim, action_dim, cfg):
        self.device = cfg.device
        self.critic = Critic(state_dim, action_dim, cfg.hidden_dim, cfg.random_seed).to(cfg.device)
        self.actor = Actor(state_dim, action_dim, cfg.hidden_dim, cfg.random_seed).to(cfg.device)
        self.target_critic = Critic(state_dim, action_dim, cfg.hidden_dim, cfg.random_seed).to(cfg.device)
        self.target_actor = Actor(state_dim, action_dim, cfg.hidden_dim, cfg.random_seed).to(cfg.device)

        # copy parameters to target net
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),  lr=cfg.critic_lr, weight_decay=WEIGHT_DECAY)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)

        self.noise = OUNoise()
        self.epsilon = cfg.epsilon
        
        self.memory = ReplayBuffer(cfg.memory_capacity)

        self.batch_size = cfg.batch_size
        self.soft_tau = cfg.soft_tau
        self.gamma = cfg.gamma
        self.cfg = cfg

        self.num_states = state_dim 


    def get_current_state(self):
        vehicle_id_entering = []
        
        current_state_dict = defaultdict(list)
        for lane in self.cfg.entering_lanes:
            vehicle_id_entering.extend(traci.lane.getLastStepVehicleIDs(lane))
        for i in range(len(vehicle_id_entering)):
            vehicle_id = vehicle_id_entering[i]
            #用字典记录当前车辆从环境中观测到的信息
            #先获取当前车辆的加速度与速度
            current_state_dict[vehicle_id].append(round(traci.vehicle.getSpeed(vehicle_id), 6))
            current_state_dict[vehicle_id].append(round(traci.vehicle.getAcceleration(vehicle_id), 6))

            #leading_veh_flag, leading_veh_speed,  leading_veh_id, dist + 1
            leader_veh_info = get_leader_veh_info(vehicle_id, vehicle_id_entering, self.cfg)
            #标记当前车辆前方有无汽车
            current_state_dict[vehicle_id].append(leader_veh_info[0])
            #前车的速度
            current_state_dict[vehicle_id].append(round(leader_veh_info[1], 6))
            #前车与当前车辆的距离
            current_state_dict[vehicle_id].append(round(leader_veh_info[2], 6))
            #再获取当前车辆到交叉路口处的距离
            dist_to_intersection = traci.lane.getLength(self.cfg.entering_lanes[0])- traci.vehicle.getLanePosition(vehicle_id)
            current_state_dict[vehicle_id].append(round(dist_to_intersection, 6))
            #先是判断当前车辆所在车道方向上的红绿灯变绿了无，若变绿，则获取当前绿灯的剩余时间
            #若没有变绿，则获得当前方向的绿灯相位还有多长时间到来
            tls_info = get_green_phase_duration(vehicle_id, self.cfg)
            current_state_dict[vehicle_id].append(tls_info[0])
            current_state_dict[vehicle_id].append(tls_info[1])
                
        return current_state_dict

    def choose_action(self, current_state_dict, curr_step, add_noise=True):
        action_dict = defaultdict(float)
        for key in current_state_dict.keys():
            state = torch.FloatTensor(current_state_dict[key]).unsqueeze(0).to(self.device)
            #在这里eval,batchnorm层会停止计算和更新mean和var，加速运算
            self.actor.eval()
            #no_grad层的作用:用于停止autograd模块的工作，以起到加速和节省显存的作用，但不会影响batch norm和dropout层的行为
            with torch.no_grad():
                action = self.actor(state)   
                action = action.detach().cpu().numpy()[0, 0]  
            self.actor.train()
            #print('添加噪声之前的动作:{}'.format(action))
            if add_noise:
                action += self.noise.get_action(action, curr_step)
            action_dict[key] = np.clip(action, -2.0, 2.0)
            #print('添加噪声之后的动作:{}'.format(action_dict[key]))
        return action_dict

    def step(self, current_state_dict, action_dict):
        for key in current_state_dict.keys():
            current_speed = current_state_dict[key][0]
            desired_speed = current_speed + action_dict[key]
            traci.vehicle.setSpeed(key, desired_speed)

        #未已经通过十字路口的车辆设置速度，跟驰模型为IDM
        set_speed_for_other_vehicles(self.cfg)
        #根据刚才给每一辆车的赋予的速度，进行单步仿真
        traci.simulationStep()
        #在仿真中，你想要设定的加速度，未必是车辆真正在下一步仿真的加速度(原因是可能由于拥堵，导致车辆此时的加速度无法达到想要的加速度)
        for key in action_dict.keys():    
            action_dict[key] = traci.vehicle.getAcceleration(key)
        next_state_dict = self.get_current_state()
        return next_state_dict, action_dict

    def get_reward(self, current_state_dict, action_dict):
        #奖励的设计原则，离散的奖励设置更能够避免车辆陷入不好的状态
        #而连续的奖励设置，则能够使奖励收敛的过程变得平滑
        reward_dict = defaultdict(list)
        
        reward_speed = 0.0
        reward_accel = 0.0
        reward_tls = 0.0
        reward_leader = 0.0 #与前车交互过程中获得的奖励,前车与后车应该保持安全距离
        extra_reward = 0.0

        f_speed = -0.05
        f_accel = -0.07
        f_leader =  1

        #必须是存在于上一时刻和当前时刻的车辆，才能获取奖励
        for key in current_state_dict.keys():
            accel = action_dict[key]    #车辆根据当前状态选择的加速度
            current_speed = current_state_dict[key][0]  #当前车辆的速度
            dist_to_intersec = current_state_dict[key][-3] #当前车辆到十字路口的距离
            light_flag = current_state_dict[key][-2]   #当前车辆行驶方向的车道是绿灯还是红灯
            time_to_green = current_state_dict[key][-1]   #当前相位还有多少时间变绿/或者当前绿灯相位持续多少时间


            leader_flag = current_state_dict[key][2] #当前车辆行驶车道上前方有无车辆
            leader_veh_speed = current_state_dict[key][3] #前车的行驶速度
            dist_to_leader = current_state_dict[key][4]   #当前车辆与前车的距离

            #-----------车辆行驶过程速度获得的奖励，速度太慢和太快都会获得惩罚----------#
            reward_speed = max(0, current_speed - self.cfg.max_speed) + max(0, self.cfg.target_speed - current_speed)
            #------------车辆行驶时加速度获得的奖励，对大的加速度应该给予惩罚-----------#
            reward_accel = accel ** 2
            #------------------------获取通过交通灯时得到的奖励-----------------------#
            if light_flag == 0:
                #此时为红灯
                #若车辆以当前车速行驶，通过十字路口所需的时间小于交通灯由红变绿的时间,则车辆会停在十字路口
                if current_speed * time_to_green > dist_to_intersec:   
                    reward_tls = -2  #-1.2是个不错的值
                else:  
                    reward_tls = 0.5
            else:
                #此时为绿灯,这里的time_to_green实际是当前绿灯的剩余时长
                #如果车辆以当前速度行驶，未能在绿灯结束前通过十字路口，则获得惩罚
                if current_speed * time_to_green < dist_to_intersec:
                    reward_tls = -2
                else:
                    reward_tls = 0.5

            #---------------------------前车与后车交互过程获得的奖励--------------------#           
            if leader_flag == 0:
                #前方没有车辆的情况,那么车辆只需要放心通过十字路口即可
                reward_leader = 0.
            else:
                #如果前方有车辆，则需要计算二车的最小安全距离
                safe_dist = get_safe_dist(current_speed, leader_veh_speed, max_deceleration=5)
                if safe_dist < dist_to_leader:
                    #此时二车之间的距离超过了安全距离，因此是安全的
                    reward_leader = 0
                else:
                    reward_leader = -1
            #--------------------------------------------------------------------------#
            total_reward = f_speed * reward_speed + f_accel * reward_accel + reward_tls + f_leader * reward_leader + extra_reward

            reward_dict[key].append(total_reward)
            reward_dict[key].append(f_speed * reward_speed)
            reward_dict[key].append(f_accel * reward_accel)
            reward_dict[key].append(reward_tls)
            reward_dict[key].append(f_leader * reward_leader)
        return reward_dict

    def update(self, batch_size):
        if len(self.memory) < self.batch_size:
            return
        state, action, reward, next_state = self.memory.sample(batch_size)

        state = torch.FloatTensor(np.array(state)).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        action = torch.FloatTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
       
        #更新critic的参数
        next_action = self.target_actor(next_state)
        target_value = self.target_critic(next_state, next_action.detach())
        expected_value = reward +  self.gamma * target_value
        eval_value = self.critic(state, action)
        value_loss = nn.MSELoss()(eval_value, expected_value.detach())

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        
        #更新actor的参数
        policy_loss = self.critic(state, self.actor(state))
        policy_loss = -policy_loss.mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        #软更新target actor和target critic的网络
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )
            
    def reset(self):
        self.noise.reset()
                 
