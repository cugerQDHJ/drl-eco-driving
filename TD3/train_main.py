from collections import defaultdict
import sys, os

curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path

import traci
import datetime
import torch
import numpy as np
from agent import TD3

from traffic import get_avg_elec_cons
from traffic import get_avg_speed
from traffic import get_avg_halting_num

from sumocfg import generate_cfg_file
from sumocfg import generate_rou_file
from sumocfg import set_sumo
from functools import partial

import random


def init_rand_seed(seed_value):
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。

    torch.manual_seed(seed_value)     # 为CPU设置随机种子
    torch.cuda.manual_seed(seed_value)      # 为当前GPU设置随机种子（只用一块GPU）
    torch.cuda.manual_seed_all(seed_value)   # 为所有GPU设置随机种子（多块GPU）
    torch.backends.cudnn.deterministic = True


curr_time = datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S")  # obtain current time


class TD3Config:
    def __init__(self):
        self.env = "sumo"
        self.algo = "TD3"
        self.gamma = 0.99
        self.max_action = 2.0 #max_action就是最大的加速度值
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory_capacity = 800000
        self.batch_size = 512
        self.train_eps = 100
        self.epsilon_start = 3
        self.expl_noise = 0.10 # Std of Gaussian exploration noise
        self.actor_lr = 1e-4 #是1e-4 a11是从20e-4开始的
        self.critic_lr = 5e-3 #5e-3 train1_10是1e-3到10e-3, 11-19是1e-4到9e-4

        self.simulation_steps = 3600
        self.max_speed = 20
        self.target_speed = 13.8 #从13.5到10，一共八个， train1->train8

        self.yellow_duration = 4
        self.green_phase_duration = 30
        self.policy_freq = 2
        self.tau = 0.005

        self.entering_lanes = ['WE_0', 'EW_0', 'NS_0', 'SN_0']
        self.depart_lanes = ['-WE_0', '-EW_0', '-NS_0', '-SN_0']
        self.intersection_lanes = [':intersection_0_0', ':intersection_1_0', ':intersection_2_0', ':intersection_3_0']
        

def run_simulation(agent, sumo_cmd, i_ep, cfg):
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

    accel_dict = defaultdict(partial(defaultdict, list))
    danger_count = 0
    total_elec_cons = 0.0

    i_step = 0
    while traci.simulation.getMinExpectedNumber() > 0 and i_step <= cfg.simulation_steps:
        i_step += 1
        #只有有车才进行学习
        if traci.vehicle.getIDCount() > 0:
            current_state_dict = agent.get_current_state()

            vehicle_id_entering = []
            for lane in cfg.entering_lanes:
                vehicle_id_entering.extend(traci.lane.getLastStepVehicleIDs(lane))
            for i in range(len(vehicle_id_entering)):
                total_elec_cons += traci.vehicle.getElectricityConsumption(vehicle_id_entering[i])
                       
            #exploring
            action_dict = defaultdict(float)
            action_dict = agent.choose_action(current_state_dict)
            for key in action_dict.keys():
                noise = np.random.normal(0, cfg.max_action * cfg.expl_noise)
                action_dict[key] = (
                    action_dict[key] + noise
                ).clip(-cfg.max_action, cfg.max_action)
            
            #获取平均速度，平均能耗
            avg_speed_step = get_avg_speed(cfg.entering_lanes)
            avg_halt_num_step = get_avg_halting_num(cfg.entering_lanes)
            avg_elec_cons = get_avg_elec_cons(cfg.entering_lanes)
            avg_speed_list.append(avg_speed_step)
            avg_halt_num_list.append(avg_halt_num_step)
            avg_elec_cons_list.append(avg_elec_cons)
            #############################################
            next_state_dict, action_dict, old_action_dict = agent.step(current_state_dict, action_dict)
            for key in action_dict.keys():
                accel_dict[key]['desired'].append(old_action_dict[key])
                accel_dict[key]['actual'].append(action_dict[key])
            #根据执行完动作到达的下一状态，来获取当前行为的奖励
            reward_dict, danger_count = agent.get_reward(current_state_dict, action_dict, danger_count)
            #reward_dict, safe_dict = agent.get_reward_safe(current_state_dict, action_dict, safe_dict)
            for key in reward_dict.keys():
                ep_reward += reward_dict[key][0]
                ep_speed_reward += reward_dict[key][1]
                ep_accel_reward += reward_dict[key][2]
                ep_tls_reward += reward_dict[key][3]
                ep_extra_reward += reward_dict[key][4]
            #再将刚才与环境互动得到的四元组存储起来
            agent.memory.push(current_state_dict, action_dict, reward_dict, next_state_dict)
            #如果互动得到的经验超过batch_size，则进行学习
            if i_ep+1 >= cfg.epsilon_start:
                agent.update(cfg.batch_size)
        else:
            traci.simulationStep()
    traci.close()
    ep_reward_list = [ep_reward, ep_speed_reward, ep_accel_reward, ep_tls_reward, ep_extra_reward]
    print("总奖励:{}, 速度奖励:{}, 加速度奖励:{}, 交通灯奖励:{}, 与前车进行交互的惩罚:{}".format(ep_reward, ep_speed_reward, ep_accel_reward, ep_tls_reward, ep_extra_reward))
    
    #将desired_Accel_dict和actual_accel_dict的内容存起来
    # json_accel = json.dumps(accel_dict)
    # json_safe = json.dumps(safe_dict)
    # accel_json_file = os.path.join('accel_json', str(i_ep)+'.json')
    # safe_json_file = os.path.join('safe_json', str(i_ep)+'.json')
    # with open(accel_json_file, 'w') as json_file:
    #     json_file.write(json_accel)
    # with open(safe_json_file, 'w') as json_file:
    #     json_file.write(json_safe)


    return ep_reward_list, avg_speed_list, avg_halt_num_list, avg_elec_cons_list, total_elec_cons, danger_count
    
def train():
    ma_rewards = []
    rewards = []
    speed_reward = []
    accel_reward = []
    tls_reward = []
    extra_reward = []
    cfg = TD3Config()

    seed_value = 70 # 设定随机数种子,默认是2022
    init_rand_seed(seed_value)
    
    curr_path = os.path.dirname(os.path.abspath(__file__))

    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    # 设定指定的GPU进行训练，有两个GPU0和GPU1
    
    #torch.cuda.set_device(1)
    ###############################################

    agent = TD3(state_dim=8, action_dim=1, cfg=cfg)
    episode_avg_speed_list = []
    episode_avg_elec_list = []
    episode_avg_halt_list = []
    episode_danger_count_list = []
    episode_elec_cons_list = []

    #s speed, a accel t tls
    #dest_path = 'models/candidate1.pth' pn policy noise nc: noise clip
    dest_path = 'models/reward_s5e-2(-5, 1)_a24e-2_t(-2_0.1)_safe(3, -0.7)_70.pth'
    print('-----------------------------------------')
    print(dest_path)
    print('-----------------------------------------')

    for i_ep in range(cfg.train_eps):
        #generate_rou_file(train_eps = i_ep + 1, path="rou_net")    
        #generate_cfg_file(train_eps = i_ep + 1)    
        #generate_rou_file(train_eps = i_ep + 1, car_count_per_lane=125,  path="rou_net")    
        generate_cfg_file(train_eps = i_ep + 1) 
        cfg_file_name = 'rou_net/intersection' + str(i_ep + 1) + '.sumocfg'
        cfg_file = os.path.join(curr_path, cfg_file_name)
        ############################################################################
        sumo_cmd = set_sumo(gui=False, sumocfg_file_name = cfg_file, max_steps=3600)
        ############################################################################

        
        ep_reward_list, avg_speed_list, avg_halt_num_list, avg_elec_cons_list, total_elec_cons, danger_count = run_simulation(agent, sumo_cmd, i_ep, cfg)
        print('Episode:{}/{}, Reward:{}'.format(i_ep + 1, cfg.train_eps, ep_reward_list[0]))
        print('Episode:{}/{}, total elec cons:{}'.format(i_ep + 1, cfg.train_eps, total_elec_cons))
        print('Episode:{}/{}, avg speed:{}'.format(i_ep + 1, cfg.train_eps, np.sum(avg_speed_list) / len(avg_speed_list)))
        print('Episode:{}/{}, avg halt num:{}'.format(i_ep + 1, cfg.train_eps, np.sum(avg_halt_num_list) / len(avg_halt_num_list)))
        print('Episode:{}/{}, avg elec cons:{}'.format(i_ep + 1, cfg.train_eps, np.sum(avg_elec_cons_list) / len(avg_elec_cons_list)))
        episode_avg_speed_list.append(np.sum(avg_speed_list) / len(avg_speed_list))
        episode_avg_elec_list.append(np.sum(avg_elec_cons_list) / len(avg_elec_cons_list))
        episode_avg_halt_list.append(np.sum(avg_halt_num_list) / len(avg_halt_num_list))
        episode_danger_count_list.append(danger_count)
        episode_elec_cons_list.append(total_elec_cons)
        rewards.append(ep_reward_list[0])
        speed_reward.append(ep_reward_list[1])
        accel_reward.append(ep_reward_list[2])
        tls_reward.append(ep_reward_list[3])
        extra_reward.append(ep_reward_list[4])

        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward_list[0])
        else:
            ma_rewards.append(ep_reward_list[0])


    actor_pth_file_path = os.path.join(curr_path, dest_path)  
    torch.save(agent.actor.state_dict(), actor_pth_file_path) ######
   

    print('model saved successfully!')
    print('Complete training!')
    print("final reward list:{}".format(rewards))
    print("final speed list:{}".format(episode_avg_speed_list))
    print("final elec cons list:{}".format(episode_avg_elec_list))
    print("final halt list:{}".format(episode_avg_halt_list))
    print('speed reward list:{}'.format(speed_reward))
    print('accel reward list:{}'.format(accel_reward))
    print('tls reward list:{}'.format(tls_reward))
    print('extra reward list:{}'.format(extra_reward))
    print('danger count list:{}'.format(episode_danger_count_list))
    print('elec cons list:{}'.format(episode_elec_cons_list))
    return rewards, ma_rewards

if __name__ == "__main__":
    rewards, ma_rewards = train()