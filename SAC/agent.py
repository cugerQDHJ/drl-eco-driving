import sys,os
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path
from memory import ReplayBuffer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim

import traci
from traffic import *
from collections import defaultdict



class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w = 3e-3):
        super(ValueNet, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)


    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

class SoftQNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, init_w=3e-3):
        super(SoftQNet, self).__init__()

        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, init_w=3e-3, 
        log_std_min=-20, log_std_max=2):
        super(PolicyNet, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)


    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std


    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)  
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, z, mean, log_std

    def get_action(self, current_state_dict, cfg):
        action_dict = defaultdict(float)
        for key in current_state_dict.keys():
            state = torch.FloatTensor(current_state_dict[key]).unsqueeze(0).to(cfg.device)
            mean, log_std = self.forward(state)
            std = log_std.exp()
            normal = Normal(mean, std)
            z = normal.sample()
            action = torch.tanh(z)
            action_dict[key] = action.detach().cpu().numpy()[0, 0]
        return action_dict


class SAC:
    def __init__(self, state_dim, action_dim, cfg) -> None:
        self.batch_size = cfg.batch_size
        self.memory = ReplayBuffer(cfg.capacity)
        self.device = cfg.device
        self.value_net = ValueNet(state_dim, cfg.hidden_dim).to(self.device)
        self.target_value_net = ValueNet(state_dim, cfg.hidden_dim).to(self.device)
        self.soft_q_net = SoftQNet(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.policy_net = PolicyNet(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=cfg.value_lr)
        self.soft_q_optimizer = optim.Adam(self.soft_q_net.parameters(), lr=cfg.soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.policy_lr)

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        self.value_criterion = nn.MSELoss()
        self.soft_q_criterion = nn.MSELoss()

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
            dist_to_intersection = traci.lane.getLength(self.cfg.entering_lanes[0])- traci.vehicle.getLanePosition(vehicle_id)
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
                action = self.policy_net.get_action(state, self.cfg)[0]
            action_dict[key] = np.clip(action, -self.cfg.max_action, self.cfg.max_action)
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
        reward_dict = defaultdict(list)
        reward_speed = 0.0
        reward_accel = 0.0
        reward_tls = 0.0
        reward_leader = 0.0 #与前车交互过程中获得的奖励,前车与后车应该保持安全距离
        extra_reward = 0.0

        f_speed = -0.11 #默认是0.11
        f_accel = -0.08  #默认是0.08
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
                    reward_tls = 0.1
            else:
                #此时为绿灯,这里的time_to_green实际是当前绿灯的剩余时长
                #如果车辆以当前速度行驶，未能在绿灯结束前通过十字路口，则获得惩罚
                if current_speed * time_to_green < dist_to_intersec:
                    reward_tls = -2
                else:
                    reward_tls = 0.1

            # #---------------------------前车与后车交互过程获得的奖励--------------------#           
            if leader_flag == 0:
                #前方没有车辆的情况,那么车辆只需要放心通过十字路口即可
                reward_leader = 0.
            else:
                #如果前方有车辆，则需要计算二车的最小安全距离
                safe_dist = get_safe_dist(current_speed, leader_veh_speed, max_deceleration=4)
                if safe_dist < dist_to_leader:
                    #此时二车之间的距离超过了安全距离，因此是安全的
                    reward_leader = 0
                else:
                    #到底是应该设置一个常数值，还是变得数值
                    reward_leader = -0.2
            # #--------------------------------------------------------------------------#
            
            total_reward = f_speed * reward_speed + f_accel * reward_accel + reward_tls + f_leader * reward_leader + extra_reward
            #这里分别记录每一个单独的奖励在训练过程中的变化图，
            reward_dict[key].append(total_reward)
            reward_dict[key].append(f_speed * reward_speed)
            reward_dict[key].append(f_accel * reward_accel)
            reward_dict[key].append(reward_tls)
            reward_dict[key].append(f_leader * reward_leader)
        return reward_dict

    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return
        state, action, reward, next_state = self.memory.sample(self.batch_size)
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)


        expected_q_value = self.soft_q_net(state, action)
        expected_value = self.value_net(state)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)

        target_value = self.target_value_net(next_state)
        next_q_value = reward + self.cfg.gamma * target_value

        q_value_loss = self.soft_q_criterion(expected_q_value, next_q_value.detach())

        expected_new_q_value = self.soft_q_net(state, new_action)
        next_value = expected_new_q_value - log_prob
        value_loss = self.value_criterion(expected_value, next_value.detach())

        log_prob_target = expected_new_q_value - expected_value
        policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()

        mean_loss = self.cfg.mean_lambda * mean.pow(2).mean()
        std_loss = self.cfg.std_lambda * log_std.pow(2).mean()
        z_loss = self.cfg.z_lambda * z.pow(2).sum(1).mean()

        policy_loss += mean_loss + std_loss + z_loss

        self.soft_q_optimizer.zero_grad()
        q_value_loss.backward()
        self.soft_q_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.cfg.soft_tau) + param.data * self.cfg.soft_tau
            )








