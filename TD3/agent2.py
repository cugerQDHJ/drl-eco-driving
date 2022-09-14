from network import Actor, Critic
import copy
import torch
import numpy as np
import torch.nn.functional as F
import traci
from collections import defaultdict
from memory import ReplayBuffer
from traffic import *

class TD3(object):
    def __init__(self, state_dim, action_dim, cfg): 
        self.device = cfg.device
        self.batch_size = cfg.batch_size

        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.max_action = cfg.max_action

        self.policy_noise = 0.2 * cfg.max_action #默认是0.2
        self.noise_clip = 0.5 * cfg.max_action #默认是0.5
        self.policy_freq = cfg.policy_freq #默认是2, 这个参数不对最后的结果造成影响
        self.tau = cfg.tau  #默认是0.005,测试0.01， 对应td_train7
        self.gamma = cfg.gamma
        
        self.actor = Actor(state_dim, action_dim, cfg.max_action).to(cfg.device)
        self.target_actor = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        
        self.critic = Critic(state_dim, action_dim).to(cfg.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.target_critic = copy.deepcopy(self.critic)

        self.memory = ReplayBuffer(cfg.memory_capacity)
        self.update_count = 0
        self.cfg = cfg

    def get_current_state(self):
        vehicle_id_entering = []
        
        current_state_dict = defaultdict(list)
        for lane in self.cfg.entering_lanes:
            vehicle_id_entering.extend(traci.lane.getLastStepVehicleIDs(lane))
        for i in range(len(vehicle_id_entering)):
            vehicle_id = vehicle_id_entering[i]
            #velocity, accel
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
            dist_to_intersection = traci.lane.getLength(self.cfg.entering_lanes[0]) - traci.vehicle.getLanePosition(vehicle_id)
            current_state_dict[vehicle_id].append(round(dist_to_intersection, 6))
            #先是判断当前车辆所在车道方向上的红绿灯变绿了无，若变绿，则获取当前绿灯的剩余时间
            #若没有变绿，则获得当前方向的绿灯相位还有多长时间到来
            tls_info = get_green_phase_duration(vehicle_id, self.cfg)
            current_state_dict[vehicle_id].append(tls_info[0])
            current_state_dict[vehicle_id].append(tls_info[1])
                
        return current_state_dict    
    
    def choose_action(self, current_state_dict):
        action_dict = defaultdict(float)
        for key in current_state_dict.keys():
            state = torch.FloatTensor(current_state_dict[key]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.actor(state)   
                action = action.detach().cpu().numpy()[0, 0]  
            action_dict[key] = np.clip(action, -self.cfg.max_action, self.max_action)
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
        old_action_dict = copy.deepcopy(action_dict)
        #在仿真中，你想要设定的加速度，未必是车辆真正在下一步仿真的加速度(原因是可能由于拥堵，导致车辆此时的加速度无法达到想要的加速度)
        for key in action_dict.keys():    
            action_dict[key] = traci.vehicle.getAcceleration(key)
        #然后把想要设定的加速度和实际的加速度存起来
        next_state_dict = self.get_current_state()
        return next_state_dict, action_dict, old_action_dict

    def get_reward(self, current_state_dict, action_dict, safe_count):
        reward_dict = defaultdict(list)
        reward_speed = 0.0
        reward_accel = 0.0
        reward_tls = 0.0
        reward_leader = 0.0 #与前车交互过程中获得的奖励,前车与后车应该保持安全距离
        extra_reward = 0.0

        f_speed = 0.05
        f_accel = -0.24
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
            if current_speed > 12 and current_speed <= 15:
                reward_speed = 1
            else:
                reward_speed = -5
            #------------车辆行驶时加速度获得的奖励，对大的加速度应该给予惩罚-----------#
            reward_accel = accel ** 2
            #------------------------获取通过交通灯时得到的奖励-----------------------#
            if light_flag == 0:
                #此时为红灯
                #若车辆以当前车速行驶，通过十字路口所需的时间小于交通灯由红变绿的时间,则车辆会停在十字路口
                if current_speed * time_to_green > dist_to_intersec:   
                    reward_tls = -2 #-1.2是个不错的值
                else:
                    reward_tls = 0.1
            else:
                #此时为绿灯,这里的time_to_green实际是当前绿灯的剩余时长
                #如果车辆以当前速度行驶，未能在绿灯结束前通过十字路口，则获得惩罚
                if current_speed * time_to_green < dist_to_intersec:
                    reward_tls = -2 #从-1.8往后回退
                else:
                    reward_tls = 0.1
            # #---------------------------前车与后车交互过程获得的奖励--------------------#           
            if leader_flag == 0:
                #前方没有车辆的情况,那么车辆只需要放心通过十字路口即可
                reward_leader = 0.
            else:
                #如果前方有车辆，则需要计算二车的最小安全距离
                safe_dist = get_safe_dist(current_speed, leader_veh_speed, max_deceleration=3)
                #将安全距离和实际距离记录下来，方便看出距离的变化
                
                if safe_dist < dist_to_leader:
                    #此时二车之间的距离超过了安全距离，因此是安全的
                    reward_leader = 0
                else:
                    safe_count += 1
                    #到底是应该设置一个常数值，还是变得数值
                    reward_leader = -0.7
            # #--------------------------------------------------------------------------#
            total_reward = f_speed * reward_speed + f_accel * reward_accel + reward_tls + f_leader * reward_leader + extra_reward
            #这里分别记录每一个单独的奖励在训练过程中的变化图，
            reward_dict[key].append(total_reward)
            reward_dict[key].append(f_speed * reward_speed)
            reward_dict[key].append(f_accel * reward_accel)
            reward_dict[key].append(reward_tls)
            reward_dict[key].append(f_leader * reward_leader)
        return reward_dict, safe_count


    def update(self, batch_size):
        if len(self.memory) < self.batch_size:
            return
        state, action, reward, next_state = self.memory.sample(batch_size)

        ############################################################

        state = torch.FloatTensor(np.array(state)).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        action = torch.FloatTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            
            next_action = (
                self.target_actor(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            #计算target q
            target_q1, target_q2 = self.target_critic(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + self.gamma * target_q
        
        #得到Q估计
        current_q1, current_q2 = self.critic(state, action)
        
        #计算评论家网络的损失
        critic_loss = F.mse_loss(current_q1, target_q) + \
                      F.mse_loss(current_q2, target_q)
        
        #optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        #delayed policy update(延迟更新policy net)
        if self.update_count % self.policy_freq == 0:
            #计算policy loss,与DDPG的policy的loss计算方法相同
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def get_reward_safe(self, current_state_dict, action_dict, safe_dict):
        reward_dict = defaultdict(list)
        reward_speed = 0.0
        reward_accel = 0.0
        reward_tls = 0.0
        reward_leader = 0.0 #与前车交互过程中获得的奖励,前车与后车应该保持安全距离
        extra_reward = 0.0

        f_speed = 0.05
        f_accel = -0.24
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
            if current_speed > 12 and current_speed <= 15:
                reward_speed = 1
            else:
                reward_speed = -5
            #------------车辆行驶时加速度获得的奖励，对大的加速度应该给予惩罚-----------#
            reward_accel = accel ** 2
            #------------------------获取通过交通灯时得到的奖励-----------------------#
            if light_flag == 0:
                #此时为红灯
                #若车辆以当前车速行驶，通过十字路口所需的时间小于交通灯由红变绿的时间,则车辆会停在十字路口
                if current_speed * time_to_green > dist_to_intersec:   
                    reward_tls = -2 #-1.2是个不错的值
                else:
                    reward_tls = 0.1
            else:
                #此时为绿灯,这里的time_to_green实际是当前绿灯的剩余时长
                #如果车辆以当前速度行驶，未能在绿灯结束前通过十字路口，则获得惩罚
                if current_speed * time_to_green < dist_to_intersec:
                    reward_tls = -2 #从-1.8往后回退
                else:
                    reward_tls = 0.1

            # #---------------------------前车与后车交互过程获得的奖励--------------------#           
            if leader_flag == 0:
                #前方没有车辆的情况,那么车辆只需要放心通过十字路口即可
                reward_leader = 0.
            else:
                #如果前方有车辆，则需要计算二车的最小安全距离
                safe_dist = get_safe_dist(current_speed, leader_veh_speed, max_deceleration=3)
                #将安全距离和实际距离记录下来，方便看出距离的变化
                safe_dict[key]['safe'].append(safe_dist)
                safe_dict[key]['real'].append(dist_to_leader)
                if safe_dist < dist_to_leader:
                    #此时二车之间的距离超过了安全距离，因此是安全的
                    reward_leader = 0
                else:
                    #到底是应该设置一个常数值，还是变得数值
                    reward_leader = -0.7
            # #--------------------------------------------------------------------------#
            
            total_reward = f_speed * reward_speed + f_accel * reward_accel + reward_tls + f_leader * reward_leader + extra_reward
            #这里分别记录每一个单独的奖励在训练过程中的变化图，
            reward_dict[key].append(total_reward)
            reward_dict[key].append(f_speed * reward_speed)
            reward_dict[key].append(f_accel * reward_accel)
            reward_dict[key].append(reward_tls)
            reward_dict[key].append(f_leader * reward_leader)
        return reward_dict, safe_dict