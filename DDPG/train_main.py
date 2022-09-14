from collections import defaultdict
import sys,os
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path

import traci
import datetime
import torch
import random
import numpy as np

from traffic import get_avg_elec_cons
from traffic import get_avg_speed
from traffic import get_avg_halting_num

from sumocfg import generate_cfg_file
from sumocfg import generate_rou_file
from sumocfg import set_sumo
from functools import partial

from agent import DDPGAgent

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


class DDPGConfig:
    def __init__(self):
        self.algo = 'DDPG'
        self.env = 'SUMO' # env name

        self.random_seed = 2020
        self.gamma = 0.99
        self.epsilon = 0.001

        self.critic_lr = 5e-3
        self.actor_lr = 1e-4

        self.memory_capacity = 1000000
        self.batch_size = 512

        self.train_eps = 100
        self.eval_eps = 10

        self.max_speed = 20
        self.target_speed = 13.8

        self.target_update = 4
        self.hidden_dim = 256
        self.soft_tau = 0.005
        self.max_action = 2


        self.entering_lanes = ['WE_0', 'EW_0', 'NS_0', 'SN_0']
        self.depart_lanes = ['-WE_0', '-EW_0', '-NS_0', '-SN_0']
        self.intersection_lanes = [':intersection_0_0', ':intersection_1_0', ':intersection_2_0', ':intersection_3_0']
        self.yellow_duration = 4
        self.green_phase_duration = 30
        #self.device = torch.device("cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.simulation_steps = 3600

def run_simulation(agent, sumo_cmd, cfg):
    # generate_rou_file(simulation_steps, car_count_per_lane)
    traci.start(sumo_cmd)
    print('Simulation......')
    ep_reward = 0
    ep_speed_reward = 0
    ep_accel_reward = 0
    ep_tls_reward = 0
    ep_tls_reward_list = []
    ep_extra_reward = 0

    avg_speed_list = []
    avg_halt_num_list = []
    avg_elec_cons_list = []

    accel_speed_dict = defaultdict(partial(defaultdict, list))
    position_dict = defaultdict(list)
    

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
                #----------------------------------------------#
                #记录当前车辆每个仿真步长中的速度和加速度
                accel_speed_dict[vehicle_id_entering[i]]['accel'].append(traci.vehicle.getAcceleration(vehicle_id_entering[i]))
                accel_speed_dict[vehicle_id_entering[i]]['speed'].append(traci.vehicle.getSpeed(vehicle_id_entering[i])) 
                #记录当前车辆每个步长中的位置，方便之后计算其平均速度
                position_dict[vehicle_id_entering[i]].append(traci.vehicle.getLanePosition(vehicle_id_entering[i]))
                #-----------------------------------------------#           
          
            action_dict = agent.choose_action(current_state_dict, i_step, add_noise=True)
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
            # 注意的是，在这里应该还要传入记录车辆位置的字典和当前存在于仿真中的车辆名字的字典
            # 这样可以方便求取平均速度对应的奖励
            reward_dict = agent.get_reward(current_state_dict, action_dict)
            #reward_dict = agent.get_reward(current_state_dict, action_dict)
            for key in reward_dict.keys():
                ep_reward += reward_dict[key][0]
                ep_speed_reward += reward_dict[key][1]
                ep_accel_reward += reward_dict[key][2]
                ep_tls_reward += reward_dict[key][3]
                ep_tls_reward_list.append(reward_dict[key][3])
                ep_extra_reward += reward_dict[key][4]
            #再将刚才与环境互动得到的四元组存储起来
            exp_count_step = agent.memory.push(current_state_dict, action_dict, reward_dict, next_state_dict)
            # #如果互动得到的经验超过batch_size，则进行学习
            agent.update(cfg.batch_size)
        else:
            traci.simulationStep()
    traci.close()
    ep_reward_list = [ep_reward, ep_speed_reward, ep_accel_reward, ep_tls_reward, ep_extra_reward]
    print('交通灯最大奖励:{}, 交通灯最小奖励:{}'.format(np.max(ep_tls_reward_list), np.min(ep_tls_reward_list)))
    print("总奖励:{}, 速度奖励:{}, 加速度奖励:{}, 交通灯奖励:{}, 与前车进行交互的惩罚:{}".format(ep_reward, ep_speed_reward, ep_accel_reward, ep_tls_reward, ep_extra_reward))

    return ep_reward_list, avg_speed_list, avg_halt_num_list, avg_elec_cons_list, accel_speed_dict
    
def train():
    ma_rewards = []
    rewards = []
    speed_reward = []
    accel_reward = []
    tls_reward = []
    extra_reward = []
    cfg = DDPGConfig()

    init_rand_seed(cfg.random_seed)
    
    curr_path = os.path.dirname(os.path.abspath(__file__))

    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')


    torch.cuda.set_device(1)


    dest_path = 'models/reward_s0.05_a7e-2_t(-2_2)_safe(5,-1).pth'
    print('-----------------------------------------')
    print(dest_path)
    print('-----------------------------------------')
    # train
    agent = DDPGAgent(state_dim=8, action_dim=1, cfg=cfg)

    episode_avg_speed_list = []
    episode_avg_elec_list = []
    episode_avg_halt_list = []

    for i_episode in range(cfg.train_eps):
        ########################################################################
        #generate_rou_file(train_eps = i_episode + 1, path='rou_net')    #######第二次是不需要更新的
        generate_cfg_file(train_eps = i_episode + 1)    #######
        cfg_file_name = 'rou_net/intersection' + str(i_episode + 1) + '.sumocfg'
        cfg_file = os.path.join(curr_path, cfg_file_name)
        sumo_cmd = set_sumo(gui=False, sumocfg_file_name = cfg_file, max_steps=3600)

        agent.reset() #重置噪声
        
        ep_reward_list, avg_speed_list, avg_halt_num_list, avg_elec_cons_list, accel_speed_dict = run_simulation(agent, sumo_cmd, cfg)
        print('Episode:{}/{}, Reward:{}'.format(i_episode + 1, cfg.train_eps, ep_reward_list[0]))
        print('Episode:{}/{}, avg speed:{}'.format(i_episode + 1, cfg.train_eps, np.sum(avg_speed_list) / len(avg_speed_list)))
        print('Episode:{}/{}, avg halt num:{}'.format(i_episode + 1, cfg.train_eps, np.sum(avg_halt_num_list) / len(avg_halt_num_list)))
        print('Episode:{}/{}, avg elec cons:{}'.format(i_episode + 1, cfg.train_eps, np.sum(avg_elec_cons_list) / len(avg_elec_cons_list)))
        episode_avg_speed_list.append(np.sum(avg_speed_list) / len(avg_speed_list))
        episode_avg_elec_list.append(np.sum(avg_elec_cons_list) / len(avg_elec_cons_list))
        episode_avg_halt_list.append(np.sum(avg_halt_num_list) / len(avg_halt_num_list))
        rewards.append(ep_reward_list[0])
        speed_reward.append(ep_reward_list[1])
        accel_reward.append(ep_reward_list[2])
        tls_reward.append(ep_reward_list[3])
        extra_reward.append(ep_reward_list[4])


        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward_list[0])
        else:
            ma_rewards.append(ep_reward_list[0])

    actor_pth_file_path = os.path.join(curr_path, dest_path)  ##########这个无需注释
  
    torch.save(agent.actor.state_dict(), actor_pth_file_path)

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
    return rewards, ma_rewards

if __name__ == "__main__":
    rewards, ma_rewards = train()
    