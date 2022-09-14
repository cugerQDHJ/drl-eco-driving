import random
import numpy as np
import os
import traci
import math

seed_value = 2020   # 设定随机数种子

np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。

def get_avg_speed(entering_lanes):
    #获取每一个车道上车辆的平均速度，然后除以车道数，最后得到整个交叉路口的车辆平均速度
    avg_speed = 0.0
    for lane in entering_lanes:
        avg_speed += traci.lane.getLastStepMeanSpeed(lane)
    return avg_speed / len(entering_lanes)

def get_avg_elec_cons(entering_lanes):
    #获取每一个车道上车辆的平均速度，然后除以车道数，最后得到整个交叉路口的车辆平均速度
    total_elec_cons = 0.0
    for lane in entering_lanes:
        total_elec_cons += traci.lane.getElectricityConsumption(lane)
    return total_elec_cons / len(entering_lanes)

def get_avg_halting_num(entering_lanes):
    total_halting_num = 0
    for lane in entering_lanes:
        total_halting_num += traci.lane.getLastStepHaltingNumber(lane)
    return total_halting_num / len(entering_lanes)


def get_leader_veh_info(veh_id, vehicle_id_entering, cfg):
    #leading_veh_flag:1表示当前车辆前方有车辆；0表示没有
    leading_veh_flag = 0
    leading_veh_speed = cfg.max_speed
    leading_veh_id = None
    dist = 249.0         #lane length
    leading_veh_info = traci.vehicle.getLeader(veh_id, 250)

    if leading_veh_info != None:
        leading_veh_id, dist = leading_veh_info
        if veh_id in vehicle_id_entering and leading_veh_id in vehicle_id_entering:
            leading_veh_flag = 1
            leading_veh_speed = traci.vehicle.getSpeed(leading_veh_id)
        if dist > 249.0:
            dist = 249.0
    return leading_veh_flag, leading_veh_speed,  dist + 1, leading_veh_id


def get_green_phase_duration(veh_id, cfg):
    veh_lane = traci.vehicle.getLaneID(veh_id)
    #计算当前交通灯相位还剩下多少时间
    current_phase_duration = traci.trafficlight.getNextSwitch('J10') - traci.simulation.getTime()
    time_to_green_ns = 0.0
    time_to_green_ew = 0.0
    current_phase_name = traci.trafficlight.getPhaseName('J10')
    if current_phase_name == 'ns_green':
        time_to_green_ns = current_phase_duration
        time_to_green_ew = current_phase_duration + cfg.yellow_duration
    elif current_phase_name == 'we_green':
        time_to_green_ew = current_phase_duration
        time_to_green_ns = current_phase_duration + cfg.yellow_duration
    elif current_phase_name == 'we_yellow':
        time_to_green_ew = current_phase_duration + cfg.green_phase_duration
        time_to_green_ns = current_phase_duration
    else:
        time_to_green_ns = current_phase_duration + cfg.green_phase_duration
        time_to_green_ew = current_phase_duration
    if veh_lane == cfg.entering_lanes[0] or veh_lane == cfg.entering_lanes[1]:
        #当前车辆在东西向的车道上,先判断此时的交通相位
        if current_phase_name == 'we_green':
            return 1, time_to_green_ew
        else:
            return 0, time_to_green_ew
    else:
        #当前车辆在南北向的车道上
        if current_phase_name == 'ns_green':
            return 1, time_to_green_ns
        else:
            return 0, time_to_green_ns



def set_speed_for_other_vehicles(cfg):
    vehicle_id_depart = []
    vehicle_id_inter = []
    #再为已经通过十字路口的车辆设置下一时刻的速度，选用的跟驰模型为IDM
    for lane in cfg.depart_lanes:
        vehicle_id_depart.extend(traci.lane.getLastStepVehicleIDs(lane))
    for veh in vehicle_id_depart:
        #leading_veh_flag, leading_veh_speed, leading_veh_accel, leading_veh_pos
        leading_veh_info = get_leader_veh_info(veh, vehicle_id_depart, cfg)
        desired_speed = IDM(veh, leading_veh_info)
        traci.vehicle.setSpeed(veh, desired_speed)
    #最后为正在十字路口处的车辆设置下一时刻的速度，选用的跟驰模型是IDM
    for lane in cfg.intersection_lanes:
        vehicle_id_inter.extend(traci.lane.getLastStepVehicleIDs(lane))
    for veh in vehicle_id_inter:
        leading_veh_info = get_leader_veh_info(veh, vehicle_id_inter, cfg)
        desired_speed = IDM(veh, leading_veh_info)
        traci.vehicle.setSpeed(veh, desired_speed)


#得到当前二车的最小安全距离
def get_safe_dist(current_speed, leader_speed, max_deceleration = 3, 
                        min_gap = 1, time_react = 0.5, time_delay = 0.3, time_i = 0.1):
    '''
    time_react:反应时间，因为是AV，所以在这里设置0.5
    time_delay:制动协调时间
    time_i:减速增长时间
    a_max:汽车减速过程中的最大加速度
    '''  
    safe_dist = current_speed * (time_react + time_delay + time_i / 2) + \
        ((current_speed - leader_speed)**2) / 2 * max_deceleration + min_gap
    return safe_dist

def IDM(veh_id, leader_veh_info):
    target_speed = 13.8
    max_accel = 2.0
    max_decel = 2.0
    min_gap = 1.0
    safe_time_headway = 1.5
    speed_difference = traci.vehicle.getSpeed(veh_id) - leader_veh_info[1]
    s = leader_veh_info[2] + min_gap

    if leader_veh_info[0] == 0:
        accel_value = max_accel * (1 - math.pow(traci.vehicle.getSpeed(veh_id) / target_speed, 2))
    else:
        accel_value =  max_accel * (1 - math.pow(traci.vehicle.getSpeed(veh_id) / target_speed, 4) - \
            math.pow((min_gap + traci.vehicle.getSpeed(veh_id) * safe_time_headway + \
            (traci.vehicle.getSpeed(veh_id) * speed_difference / 2 * math.sqrt(max_accel * max_decel))) / s, 2))

    if accel_value > 2:
        accel_value = 2
    if accel_value < -2:
        accel_value = -2
    desired_speed = traci.vehicle.getSpeed(veh_id) + accel_value
    if desired_speed < 0:
        desired_speed = 0.0
    elif desired_speed > 13.8:
        desired_speed = 13.8
    return desired_speed