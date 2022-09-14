from agent import SAC
from functools import partial
import numpy as np
import torch
import random
import os
from collections import defaultdict
import traci

import sys
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path
from common.traffic import get_avg_elec_cons
from common.traffic import get_avg_speed
from common.traffic import get_avg_halting_num

from sumocfg import generate_cfg_file
from sumocfg import generate_rou_file
from sumocfg import set_sumo


def init_rand_seed(seed_value):
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。

    torch.manual_seed(seed_value)     # 为CPU设置随机种子
    torch.cuda.manual_seed(seed_value)      # 为当前GPU设置随机种子（只用一块GPU）
    torch.cuda.manual_seed_all(seed_value)   # 为所有GPU设置随机种子（多块GPU）
    torch.backends.cudnn.deterministic = True

class SACConfig:
    def __init__(self):
        self.env = "sumo"
        self.algo = "SAC"
        self.train_eps = 100

        self.gamma = 0.99
        self.mean_lambda=1e-3
        self.std_lambda=1e-3
        self.z_lambda=0.0
        self.soft_tau=1e-2
        self.value_lr = 3e-4
        self.soft_q_lr = 3e-4
        self.policy_lr = 3e-4
        self.capacity = 1000000
        self.hidden_dim = 256
        self.batch_size  = 256
        self.device = torch.device('cpu')
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_action = 2.0
        self.simulation_steps = 3600
        self.max_speed = 20
        self.target_speed = 13.8
        self.entering_lanes = ['WE_0', 'EW_0', 'NS_0', 'SN_0']
        self.depart_lanes = ['-WE_0', '-EW_0', '-NS_0', '-SN_0']
        self.intersection_lanes = [':intersection_0_0', ':intersection_1_0', ':intersection_2_0', ':intersection_3_0']
        self.yellow_duration = 4
        self.green_phase_duration = 30
        

def run_simulation(agent, sumo_cmd, cfg):
    traci.start(sumo_cmd)
    print('Simulation......')
    ep_reward = 0
    ep_speed_reward = 0
    ep_accel_reward = 0
    ep_tls_reward = 0
    ep_extra_reward = 0

    avg_speed_list = []
    avg_halt_num_list = []
    avg_elec_cons_list = []

    accel_speed_dict = defaultdict(partial(defaultdict, list))

    i_step = 0
    while traci.simulation.getMinExpectedNumber() > 0 and i_step <= cfg.simulation_steps:
        i_step += 1
        #只有有车才进行学习
        if traci.vehicle.getIDCount() > 0:
            current_state_dict = agent.get_current_state()

            vehicle_id_entering = []
            for lane in cfg.entering_lanes:
                vehicle_id_entering.extend(traci.lane.getLastStepVehicleIDs(lane))
                       
            #exploring
            action_dict = defaultdict(float)
            action_dict = agent.policy_net.get_action(current_state_dict, cfg)
            
            #获取平均速度，平均能耗
            avg_speed_step = get_avg_speed(cfg.entering_lanes)
            avg_halt_num_step = get_avg_halting_num(cfg.entering_lanes)
            avg_elec_cons = get_avg_elec_cons(cfg.entering_lanes)
            avg_speed_list.append(avg_speed_step)
            avg_halt_num_list.append(avg_halt_num_step)
            avg_elec_cons_list.append(avg_elec_cons)
            #############################################
            next_state_dict, action_dict = agent.step(current_state_dict, action_dict)
            #根据执行完动作到达的下一状态，来获取当前行为的奖励
            reward_dict = agent.get_reward(current_state_dict, action_dict)
            for key in reward_dict.keys():
                ep_reward += reward_dict[key][0]
                ep_speed_reward += reward_dict[key][1]
                ep_accel_reward += reward_dict[key][2]
                ep_tls_reward += reward_dict[key][3]
                ep_extra_reward += reward_dict[key][4]
            #再将刚才与环境互动得到的四元组存储起来
            agent.memory.push(current_state_dict, action_dict, reward_dict, next_state_dict)
            #如果互动得到的经验超过batch_size，则进行学习
            agent.update(cfg.batch_size)
        else:
            traci.simulationStep()
    traci.close()
    ep_reward_list = [ep_reward, ep_speed_reward, ep_accel_reward, ep_tls_reward, ep_extra_reward]
    print("总奖励:{}, 速度奖励:{}, 加速度奖励:{}, 交通灯奖励:{}, 与前车进行交互的惩罚:{}".format(ep_reward, ep_speed_reward, ep_accel_reward, ep_tls_reward, ep_extra_reward))
    return ep_reward_list, avg_speed_list, avg_halt_num_list, avg_elec_cons_list, accel_speed_dict
    
def train():
    ma_rewards = []
    rewards = []
    speed_reward = []
    accel_reward = []
    tls_reward = []
    extra_reward = []
    cfg = SACConfig()

    seed_value = 2020  # 设定随机数种子,默认是2020
    init_rand_seed(seed_value)
    
    curr_path = os.path.dirname(os.path.abspath(__file__))

    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    # train
    # 设定指定的GPU进行训练，有两个GPU0和GPU1
    
    #torch.cuda.set_device(1)
    ###############################################

    agent = SAC(state_dim=8, action_dim=1, cfg=cfg)
    episode_avg_speed_list = []
    episode_avg_elec_list = []
    episode_avg_halt_list = []

    #models/sac_reward_s0.11_a8e-2_t(-2_0.1)_cpu.pth
    #
    dest_path = 'models/sac_reward_s0.11_a8e-2_t(-2_0.1)_safe(4,0.2)_cpu.pth'
    print('-----------------------------------------')
    print(dest_path)
    print('-----------------------------------------')
    best_episode_reward = -1000000
    best_episode_reward_idx = 0

    #使用early stop技术防止训练出现过拟合,如果十个回合内，没有出现奖励的改进，就提前停止训练
    for i_ep in range(cfg.train_eps):
        #generate_rou_file(train_eps = i_ep + 1, path="rou_net")    #######
        generate_cfg_file(train_eps = i_ep + 1)    #######
        cfg_file_name = 'rou_net/intersection' + str(i_ep + 1) + '.sumocfg'
        cfg_file = os.path.join(curr_path, cfg_file_name)
        ############################################################################
        sumo_cmd = set_sumo(gui=True, sumocfg_file_name = cfg_file, max_steps=3600)
        ############################################################################
        ep_reward_list, avg_speed_list, avg_halt_num_list, avg_elec_cons_list, accel_speed_dict = run_simulation(agent, sumo_cmd, cfg)
        if ep_reward_list[0] > best_episode_reward:
            best_episode_reward = ep_reward_list[0]
            best_episode_reward_idx = i_ep
        print('best reward:{}'.format(best_episode_reward))
        print('Episode:{}/{}, Reward:{}'.format(i_ep + 1, cfg.train_eps, ep_reward_list[0]))
        print('Episode:{}/{}, avg speed:{}'.format(i_ep + 1, cfg.train_eps, np.sum(avg_speed_list) / len(avg_speed_list)))
        print('Episode:{}/{}, avg halt num:{}'.format(i_ep + 1, cfg.train_eps, np.sum(avg_halt_num_list) / len(avg_halt_num_list)))
        print('Episode:{}/{}, avg elec cons:{}'.format(i_ep + 1, cfg.train_eps, np.sum(avg_elec_cons_list) / len(avg_elec_cons_list)))
        episode_avg_speed_list.append(np.sum(avg_speed_list) / len(avg_speed_list))
        episode_avg_elec_list.append(np.sum(avg_elec_cons_list) / len(avg_elec_cons_list))
        episode_avg_halt_list.append(np.sum(avg_halt_num_list) / len(avg_halt_num_list))
        rewards.append(ep_reward_list[0])
        speed_reward.append(ep_reward_list[1])
        accel_reward.append(ep_reward_list[2])
        tls_reward.append(ep_reward_list[3])
        extra_reward.append(ep_reward_list[4])

        if ma_rewards:
            ma_rewards.append(0.8*ma_rewards[-1]+0.2*ep_reward_list[0])
        else:
            ma_rewards.append(ep_reward_list[0])


        #如果当前已经连续十回合，奖励上没有改进，那么则提前停止训练
        if i_ep - best_episode_reward_idx >= 10:
            print('early stop!!!')
            break

    actor_pth_file_path = os.path.join(curr_path, dest_path)  
    torch.save(agent.policy_net.state_dict(), actor_pth_file_path) #####
   

    print('model saved successfully!')
    print('Complete training!')
    print("final reward list:{}".format(rewards))
    print("final speed list:{}".format(episode_avg_speed_list))
    print("final elec cons list:{}".format(episode_avg_elec_list))
    print("final halt list:{}".format(episode_avg_halt_list))
    print('speed reward list:{}'.format(speed_reward))
    print('accel reward list:{}'.format(accel_reward))
    print('tls reward list:{}'.format(tls_reward))
    return rewards, ma_rewards

if __name__ == "__main__":
    rewards, ma_rewards = train()