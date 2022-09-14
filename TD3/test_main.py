from collections import defaultdict
from functools import partial

import os
import sys
import json

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
from agent import TD3


class TD3Config:
    def __init__(self):
        self.env = "sumo"
        self.algo = "TD3"
        self.max_action = 2.0 #max_action就是最大的加速度值
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory_capacity = 800000
        self.batch_size = 512
        self.eval_eps = 10
        self.epsilon_start = 3
        self.expl_noise = 0.10 # Std of Gaussian exploration noise
        self.policy_freq = 2
        self.tau = 0.005
        self.gamma = 0.99
        self.actor_lr = 1e-4 #是1e-4 a11是从20e-4开始的
        self.critic_lr = 5e-3 #5e-3 train1_10是1e-3到10e-3, 11-19是1e-4到9e-4

        self.simulation_steps = 3600
        self.max_speed = 20
        self.target_speed = 10

        self.yellow_duration = 4
        self.green_phase_duration = 30

        self.entering_lanes = ['WE_0', 'EW_0', 'NS_0', 'SN_0']
        self.depart_lanes = ['-WE_0', '-EW_0', '-NS_0', '-SN_0']
        self.intersection_lanes = [':intersection_0_0', ':intersection_1_0', ':intersection_2_0', ':intersection_3_0']


def run_simulation_eval(agent, i_ep, sumo_cmd, cfg):
    '''测试过程中用的应该是固定的路网文件'''
    traci.start(sumo_cmd)
    print('Simulation......')
    i_step = 0
    start_time = timeit.default_timer()

    avg_speed_list = []
    avg_halt_num_list = []
    avg_elec_cons_list = []
    total_elec_cons = 0.0
    danger_count = 0

    position_time_dict = defaultdict(partial(defaultdict, list))
    safe_dict = defaultdict(partial(defaultdict, list))

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
            action_dict = agent.choose_action(current_state_dict)
            next_state_dict, action_dict, old_action_dict = agent.step(current_state_dict, action_dict)

            avg_speed_step = get_avg_speed(cfg.entering_lanes)
            avg_halt_num_step = get_avg_halting_num(cfg.entering_lanes)
            avg_elec_cons = get_avg_elec_cons(cfg.entering_lanes)

            avg_speed_list.append(avg_speed_step)
            avg_halt_num_list.append(avg_halt_num_step)
            avg_elec_cons_list.append(avg_elec_cons)

            reward_dict, safe_dict = agent.get_reward(current_state_dict, action_dict, danger_count)
        else:
            traci.simulationStep()
    traci.close()

    #计算当前仿真所有车辆通过十字路口时间的和
    episode_travel_time = 0
    for key in position_time_dict.keys():
        episode_travel_time += len(position_time_dict[key]['position'])

    json_safe = json.dumps(safe_dict)
    safe_json_file = os.path.join('safe_json', str(i_ep)+'.json')

    with open(safe_json_file, 'w') as json_file:
        json_file.write(json_safe)

    simulation_time = round(timeit.default_timer() - start_time, 1)
    print('Simulation time:{}'.format(simulation_time))

    return avg_speed_list, avg_halt_num_list, avg_elec_cons_list, total_elec_cons, episode_travel_time

def eval():
    cfg = TD3Config()

    curr_path = os.path.dirname(os.path.abspath(__file__))
     
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')

    ####################################
    # torch.cuda.set_device(1)
    ####################################
    agent = TD3(state_dim=8, action_dim=1, cfg=cfg)

    curr_actor_path = os.path.dirname(os.path.abspath(__file__))
    
    dest_pth = 'models/reward_s5e-2(-5, 1)_a16e-2_t(-2_0.1)_safe(3, -0.7)_2033.pth'
    print('------------------------------------------')
    print(dest_pth)
    print('------------------------------------------')
    actor_file_path = os.path.join(curr_actor_path, dest_pth) #######
    agent.actor.load_state_dict(torch.load(actor_file_path))

    episode_avg_speed_list = []
    episode_avg_elec_list = []
    episode_avg_halt_list = []
    total_elec_cons_list = []
    travel_time_list = []

   
    
    print('网络参数导入成功!')
    agent.actor.eval()
    for i_ep in range(cfg.eval_eps):
        #########################################################################
        #测试的过程中生成一遍新的路网文件，这里不能与训练中用的路网一样
        #generate_rou_file(train_eps = i_ep + 1, path="test_rou_net", random_seed=2022)    #######
        #generate_test_cfg_file(eval_eps = i_ep + 1)    #######

        cfg_file_name = 'test_rou_net/intersection' + str(i_ep + 1) + '.sumocfg'
        cfg_file = os.path.join(curr_path, cfg_file_name)
        sumo_cmd = set_sumo(gui=False, sumocfg_file_name = cfg_file, max_steps=3600)
        ###########################################################################

        

        avg_speed_list, avg_halt_num_list, avg_elec_cons_list, total_elec_cons, episode_travel_time = run_simulation_eval(agent, i_ep, sumo_cmd, cfg)
        print('Episode:{}/{}, avg speed:{}'.format(i_ep + 1, cfg.eval_eps, np.sum(avg_speed_list) / len(avg_speed_list)))
        print('Episode:{}/{}, avg halt num:{}'.format(i_ep + 1, cfg.eval_eps, np.sum(avg_halt_num_list) / len(avg_halt_num_list)))
        print('Episode:{}/{}, total elec cons:{}'.format(i_ep + 1, cfg.eval_eps, total_elec_cons))
        episode_avg_speed_list.append(np.sum(avg_speed_list) / len(avg_speed_list))
        episode_avg_elec_list.append(np.sum(avg_elec_cons_list) / len(avg_elec_cons_list))
        episode_avg_halt_list.append(np.sum(avg_halt_num_list) / len(avg_halt_num_list))
        total_elec_cons_list.append(total_elec_cons)
        travel_time_list.append(episode_travel_time)

        
    print('Complete eval! ')
    print("final speed list:{}".format(episode_avg_speed_list))
    print("final halt list:{}".format(episode_avg_halt_list))
    print("final episode elec cons list:{}".format(total_elec_cons_list))
    print("final travel time list:{}".format(travel_time_list))


if __name__ == '__main__':
    # nohup python -u test_main.py > TD3_test5.log 2>&1 &
    eval()