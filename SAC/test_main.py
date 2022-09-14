from collections import defaultdict
from functools import partial

import os
import sys

curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path

import traci
import timeit
import torch
import numpy as np

from sumocfg import set_sumo
from sumocfg import generate_rou_file
from sumocfg import generate_test_cfg_file

from traffic import *
from agent import SAC


class SACConfig:
    def __init__(self):
        self.env = "sumo"
        self.algo = "SAC"
        self.eval_eps = 15

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_action = 2.0
        self.simulation_steps = 3600
        self.max_speed = 20
        self.target_speed = 13.8
        self.entering_lanes = ['WE_0', 'EW_0', 'NS_0', 'SN_0']
        self.depart_lanes = ['-WE_0', '-EW_0', '-NS_0', '-SN_0']
        self.intersection_lanes = [':intersection_0_0', ':intersection_1_0', ':intersection_2_0', ':intersection_3_0']
        self.yellow_duration = 4
        self.green_phase_duration = 30


def run_simulation_eval(agent, sumo_cmd, cfg):
    '''测试过程中用的应该是固定的路网文件'''
    traci.start(sumo_cmd)
    print('Simulation......')
    ep_reward = 0
    i_step = 0
    start_time = timeit.default_timer()

    avg_speed_list = []
    avg_halt_num_list = []
    avg_elec_cons_list = []
    total_elec_cons = 0.0

    position_time_dict = defaultdict(partial(defaultdict, list))

    while traci.simulation.getMinExpectedNumber() > 0 and i_step <= cfg.simulation_steps:
        i_step += 1
        if traci.vehicle.getIDCount() > 0:
            current_state_dict = agent.get_current_state()
            vehicle_id_entering = []
            for lane in cfg.entering_lanes:
                vehicle_id_entering.extend(traci.lane.getLastStepVehicleIDs(lane))
                
            for i in range(len(vehicle_id_entering)):
                #----------------------------------------------#
                # #记录当前车辆每个仿真中的位置[0, 250]m
                position_time_dict[vehicle_id_entering[i]]['position'].append(traci.vehicle.getLanePosition(vehicle_id_entering[i]))
                # #另外一方面记录当前交通灯的相位，以及持续时间  
                position_time_dict[vehicle_id_entering[i]]['tls_phase'].append(traci.trafficlight.getPhaseName('J10'))  
                #记录每个仿真中的能耗. wh/s
                position_time_dict[vehicle_id_entering[i]]['electricity'].append(traci.vehicle.getElectricityConsumption(vehicle_id_entering[i]))
                #-----------------------------------------------#
                #获取总能耗            
                total_elec_cons += traci.vehicle.getElectricityConsumption(vehicle_id_entering[i])
            action_dict = agent.policy_net.get_action(current_state_dict)
            next_state_dict, action_dict = agent.step(current_state_dict, action_dict)

            avg_speed_step = get_avg_speed(cfg.entering_lanes)
            avg_halt_num_step = get_avg_halting_num(cfg.entering_lanes)
            avg_elec_cons = get_avg_elec_cons(cfg.entering_lanes)

            avg_speed_list.append(avg_speed_step)
            avg_halt_num_list.append(avg_halt_num_step)
            avg_elec_cons_list.append(avg_elec_cons)

            reward_dict = agent.get_reward(current_state_dict, action_dict)
            for key in reward_dict.keys():
                ep_reward += reward_dict[key][0]
            agent.memory.push(current_state_dict, action_dict, reward_dict, next_state_dict)
        else:
            traci.simulationStep()
    traci.close()

    #计算当前仿真所有车辆通过十字路口时间的和
    episode_travel_time = 0
    for key in position_time_dict.keys():
        episode_travel_time += len(position_time_dict[key]['position'])

    simulation_time = round(timeit.default_timer() - start_time, 1)
    print('Simulation time:{}'.format(simulation_time))

    return ep_reward, avg_speed_list, avg_halt_num_list, avg_elec_cons_list, total_elec_cons, episode_travel_time

def eval():
    ma_rewards = []
    rewards = []
    cfg = SACConfig()

    curr_path = os.path.dirname(os.path.abspath(__file__))
     
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')

    ####################################
    torch.cuda.set_device(1)
    ####################################
    agent = SAC(state_dim=8, action_dim=1, cfg=cfg)

    curr_actor_path = os.path.dirname(os.path.abspath(__file__))
    
    dest_pth = 'models/reward_s0.11_a8e-2_t(-2_0.1)_safe(4, 0.9).pth'
    print('------------------------------------------')
    print(dest_pth)
    print('------------------------------------------')
    actor_file_path = os.path.join(curr_actor_path, dest_pth) #######
    agent.policy_net.load_state_dict(torch.load(actor_file_path))

    episode_avg_speed_list = []
    episode_avg_elec_list = []
    episode_avg_halt_list = []
    total_elec_cons_list = []
    travel_time_list = []

   
    
    print('网络参数导入成功!')
    agent.policy_net.eval()
    for i_ep in range(cfg.eval_eps):
        #########################################################################
        #测试的过程中生成一遍新的路网文件，这里不能与训练中用的路网一样
        #generate_rou_file(train_eps = i_ep + 1, path="test_rou_net", random_seed=2022)    #######
        generate_test_cfg_file(train_eps = i_ep + 1)    #######

        cfg_file_name = 'test_rou_net/intersection' + str(i_ep + 1) + '.sumocfg'
        cfg_file = os.path.join(curr_path, cfg_file_name)
        sumo_cmd = set_sumo(gui=False, sumocfg_file_name = cfg_file, max_steps=3600)
        ###########################################################################

        

        ep_reward, avg_speed_list, avg_halt_num_list, avg_elec_cons_list, total_elec_cons, episode_travel_time = run_simulation_eval(agent, sumo_cmd, cfg)
        print('Episode:{}/{}, Reward:{}'.format(i_ep + 1, cfg.eval_eps, ep_reward))
        print('Episode:{}/{}, avg speed:{}'.format(i_ep + 1, cfg.eval_eps, np.sum(avg_speed_list) / len(avg_speed_list)))
        print('Episode:{}/{}, avg halt num:{}'.format(i_ep + 1, cfg.eval_eps, np.sum(avg_halt_num_list) / len(avg_halt_num_list)))
        print('Episode:{}/{}, total elec cons:{}'.format(i_ep + 1, cfg.eval_eps, total_elec_cons))
        episode_avg_speed_list.append(np.sum(avg_speed_list) / len(avg_speed_list))
        episode_avg_elec_list.append(np.sum(avg_elec_cons_list) / len(avg_elec_cons_list))
        episode_avg_halt_list.append(np.sum(avg_halt_num_list) / len(avg_halt_num_list))
        total_elec_cons_list.append(total_elec_cons)
        travel_time_list.append(episode_travel_time)
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
        
    print('Complete eval! ')
    print("final reward list:{}".format(rewards))
    print("final speed list:{}".format(episode_avg_speed_list))
    print("final halt list:{}".format(episode_avg_halt_list))
    print("final episode elec cons list:{}".format(total_elec_cons_list))
    print("final travel time list:{}".format(travel_time_list))

    #训练完顺便将agent.memory中保存的数据全部存入csv文件中，方便对加速度，速度和奖励的关系做分析
    # saved_idx = [0, agent.state_dim, agent.state_dim + 1]
    # pd.DataFrame(agent.memory[:agent.memory_counter,saved_idx], columns=['speed', 'accel', 'reward']).to_csv('2e-1_8e-2_6e-2_gpu_Y.csv') ######
    return rewards, ma_rewards

if __name__ == '__main__':
    # nohup python -u test_main.py > TD3_test5.log 2>&1 &
    eval()