import math
import random
import os
import numpy as np
import sys
from sumolib import checkBinary

# seed_value = 2020   # 设定随机数种子

# np.random.seed(seed_value)
# random.seed(seed_value)
# os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。


def set_sumo(gui, sumocfg_file_name, max_steps):
    """
    Configure various parameters of SUMO
    """
    # sumo things - we need to import python modules from the $SUMO_HOME/tools directory
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    # setting the cmd mode or the visual mode    
    if gui == False:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
 
    # setting the cmd command to run sumo at simulation time
    sumo_cmd = [sumoBinary, "-c", sumocfg_file_name, "--no-step-log", "true", "--waiting-time-memory", str(max_steps)]

    return sumo_cmd

def generate_cfg_file(train_eps):
    curr_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file_name = 'rou_net/intersection' + str(train_eps) + '.sumocfg'
    rou_file_name = 'rou_net/intersection' + str(train_eps) + '.rou.xml'
    net_file_name = 'rou_net/intersection.net.xml'
    cfg_file = os.path.join(curr_path, cfg_file_name)
    rou_file = os.path.join(curr_path, rou_file_name)
    net_file = os.path.join(curr_path, net_file_name)

    with open(cfg_file, "w") as route:
        #EV的能耗是Wh每秒 carFollowModel="IDM"
        print("""<configuration>
    <input>
        <net-file value="{}"/>
        <route-files value="{}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="1"/>
    </time>
    <tripinfo-output value="tripinfo.xml"/>
</configuration>""".format(net_file, rou_file), file=route)


def generate_mix_cfg_file(train_eps):
    curr_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file_name = 'mix_rou_net/intersection' + str(train_eps) + '.sumocfg'
    rou_file_name = 'mix_rou_net/intersection' + str(train_eps) + '.rou.xml'
    net_file_name = 'mix_rou_net/intersection.net.xml'
    cfg_file = os.path.join(curr_path, cfg_file_name)
    rou_file = os.path.join(curr_path, rou_file_name)
    net_file = os.path.join(curr_path, net_file_name)

    with open(cfg_file, "w") as route:
        #EV的能耗是Wh每秒 carFollowModel="IDM"
        print("""<configuration>
    <input>
        <net-file value="{}"/>
        <route-files value="{}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="1"/>
    </time>
    <tripinfo-output value="unseen_tripinfo.xml"/>
</configuration>""".format(net_file, rou_file), file=route)


def generate_test_cfg_file(train_eps, path='test_rou_net'):
    curr_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file_name = path+'/intersection' + str(train_eps) + '.sumocfg'
    rou_file_name = path+'/intersection' + str(train_eps) + '.rou.xml'
    net_file_name = path+'/intersection.net.xml'
    cfg_file = os.path.join(curr_path, cfg_file_name)
    rou_file = os.path.join(curr_path, rou_file_name)
    net_file = os.path.join(curr_path, net_file_name)

    with open(cfg_file, "w") as route:
        #EV的能耗是Wh每秒 carFollowModel="IDM"
        print("""<configuration>
    <input>
        <net-file value="{}"/>
        <route-files value="{}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="1"/>
    </time>
    <tripinfo-output value="tripinfo.xml"/>
</configuration>""".format(net_file, rou_file), file=route)

def generate_rou_file(train_eps, simulation_steps = 3600, car_count_per_lane = 100, path='test_rou_net'):
    #在4000s内随机生成400辆车
    random.seed(42)  #设置随机数种子，能够让结果重现
    timings_ns = np.random.weibull(2, car_count_per_lane)
    timings_sn = np.random.weibull(2, car_count_per_lane)
    timings_we = np.random.weibull(2, car_count_per_lane)
    timings_ew = np.random.weibull(2, car_count_per_lane)
    timings_ns = np.sort(timings_ns)
    timings_sn = np.sort(timings_sn)
    timings_we = np.sort(timings_we)
    timings_ew = np.sort(timings_ew)
    

    #reshape the distribution to fit the interval 0:max_steps
    car_gen_steps_ns = []
    car_gen_steps_sn = []
    car_gen_steps_we = []
    car_gen_steps_ew = []
    min_old_ns = math.floor(timings_ns[1])
    max_old_ns = math.ceil(timings_ns[-1])
    min_old_sn = math.floor(timings_sn[1])
    max_old_sn = math.ceil(timings_sn[-1])
    min_old_we = math.floor(timings_we[1])
    max_old_we = math.ceil(timings_we[-1])
    min_old_ew = math.floor(timings_ew[1])
    max_old_ew = math.ceil(timings_ew[-1])
    min_new = 0
    max_new = simulation_steps
    for i in range(len(timings_ns)):
        car_gen_steps_ns.append(((max_new - min_new) / (max_old_ns - min_old_ns)) * (timings_ns[i] - max_old_ns) + max_new)
        car_gen_steps_sn.append(((max_new - min_new) / (max_old_sn - min_old_sn)) * (timings_sn[i] - max_old_sn) + max_new)
        car_gen_steps_we.append(((max_new - min_new) / (max_old_we - min_old_we)) * (timings_we[i] - max_old_we) + max_new)
        car_gen_steps_ew.append(((max_new - min_new) / (max_old_ew - min_old_ew)) * (timings_ew[i] - max_old_ew) + max_new)

    car_gen_steps_ns = np.rint(car_gen_steps_ns)  # 对时间进行取整
    car_gen_steps_sn = np.rint(car_gen_steps_sn)  
    car_gen_steps_we = np.rint(car_gen_steps_we) 
    car_gen_steps_ew = np.rint(car_gen_steps_ew) 

    curr_path = os.path.dirname(os.path.abspath(__file__))
    rou_file_name = path+'/intersection' + str(train_eps) + '.rou.xml'
    rou_cfg_file = os.path.join(curr_path, rou_file_name)
    #print(rou_cfg_file)
   
    with open(rou_cfg_file, "w") as route:
        #EV的能耗是Wh每秒 carFollowModel="IDM"
        print("""<routes>

        <vType id = 'ego_car' vclass="evehicle" emissionClass="Energy" tau="1.5" accel="3.0" decel="3.0" color="#00FFFF" sigma="0.2" length="4.90" minGap="1.0" maxSpeed="20" carFollowing="IDM" guiShape="passenger"/>

        <route id="N2S" edges="NS -SN"/>
        <route id="S2N" edges="SN -NS"/>
        <route id="W2E" edges="WE -EW"/>
        <route id="E2W" edges="EW -WE"/>""", file=route)
        depart_list = []
        for i in range(car_count_per_lane):
            #随机选择一个车辆行驶的方向，随机选择车辆的类型
            depart_list.append(['N2S', car_gen_steps_ns[i]])           
            depart_list.append(['S2N', car_gen_steps_sn[i]])
            depart_list.append(['W2E', car_gen_steps_we[i]])
            depart_list.append(['E2W', car_gen_steps_ew[i]])
        depart_list = sorted(depart_list, key = lambda x:x[1])
        for i in range(car_count_per_lane * 4):
            print('   <vehicle id="%s_%i" type="ego_car" route="%s" depart="%i" departSpeed="10" departLane="best"/>' % (depart_list[i][0], i + 1, depart_list[i][0], depart_list[i][1]), file = route)             
        print('</routes>', file=route)


def generate_uniform_rou_file(episode, simulation_steps=3600, car_count_per_lane = 100, path='uniform_rou_net'):
    #生成均匀车辆的路网
    curr_path = os.path.dirname(os.path.abspath(__file__))
    rou_file_name = path+'/intersection' + str(episode) + '.rou.xml'
    rou_cfg_file = os.path.join(curr_path, rou_file_name)

    rou_list = ['S2N', 'E2W', 'W2E', 'N2S']
    depart_time_list = [0, 0, 0, 0] #每个方向的车道上第一辆车出发的时间
   
    with open(rou_cfg_file, "w") as route:
        #EV的能耗是Wh每秒 carFollowModel="IDM"
        print("""<routes>

        <vType id = 'ego_car' vclass="evehicle" emissionClass="Energy" tau="1.5" accel="3.0" decel="3.0" color="#00FFFF" sigma="0.2" length="4.90" minGap="1.0" maxSpeed="20" carFollowing="IDM" guiShape="passenger"/>

        <route id="N2S" edges="NS -SN"/>
        <route id="S2N" edges="SN -NS"/>
        <route id="W2E" edges="WE -EW"/>
        <route id="E2W" edges="EW -WE"/>""", file=route)

        for i in range(car_count_per_lane * 4):
            curr_rou_idx = i % len(rou_list)
            if depart_time_list[curr_rou_idx] > simulation_steps:
                break #防止越界
            print('   <vehicle id="%s_%i" type="ego_car" route="%s" depart="%i" departSpeed="10" departLane="best"/>' % (rou_list[curr_rou_idx], i + 1, rou_list[curr_rou_idx], depart_time_list[curr_rou_idx]), file = route)  
            depart_time_list[curr_rou_idx] += int(simulation_steps / car_count_per_lane)          
        print('</routes>', file=route)



def generate_single_rou_file(episode, simulation_steps=3600, car_count_per_lane = 600, path='uniform_rou_net'):
    #生成均匀车辆的路网
    curr_path = os.path.dirname(os.path.abspath(__file__))
    rou_file_name = path+'/intersection' + str(episode) + '.rou.xml'
    rou_cfg_file = os.path.join(curr_path, rou_file_name)

    rou_list = ['S2N', 'N2S'] #这里可以选择另外东西向的
    depart_time_list = [0, 2] 
   
    with open(rou_cfg_file, "w") as route:
        #EV的能耗是Wh每秒 carFollowModel="IDM"
        print("""<routes>

        <vType id = 'ego_car' vclass="evehicle" emissionClass="Energy" accel="3.0" decel="3.0" color="#00FFFF" length="4.90" minGap="2.0" maxSpeed="20" guiShape="passenger"/>

        <route id="N2S" edges="NS -SN"/>
        <route id="S2N" edges="SN -NS"/>
        <route id="W2E" edges="WE -EW"/>
        <route id="E2W" edges="EW -WE"/>""", file=route)

        for i in range(car_count_per_lane * 2):
            curr_rou_idx = i % len(rou_list)
            if depart_time_list[curr_rou_idx] > simulation_steps:
                break #防止越界
            print('   <vehicle id="%s_%i" type="ego_car" route="%s" depart="%i" departSpeed="10" departLane="best"/>' % (rou_list[curr_rou_idx], i + 1, rou_list[curr_rou_idx], depart_time_list[curr_rou_idx]), file = route)  
            depart_time_list[curr_rou_idx] += int(simulation_steps / car_count_per_lane)          
        print('</routes>', file=route)


def generate_mix_rou_file(train_eps, cav_penetration, simulation_steps = 3600, car_count_per_lane = 100, path='mix_rou_net'):
    #在4000s内随机生成400辆车
    random.seed(42)  #设置随机数种子，能够让结果重现
    timings_ns = np.random.weibull(2, car_count_per_lane)
    timings_sn = np.random.weibull(2, car_count_per_lane)
    timings_we = np.random.weibull(2, car_count_per_lane)
    timings_ew = np.random.weibull(2, car_count_per_lane)
    timings_ns = np.sort(timings_ns)
    timings_sn = np.sort(timings_sn)
    timings_we = np.sort(timings_we)
    timings_ew = np.sort(timings_ew)
    

    #reshape the distribution to fit the interval 0:max_steps
    car_gen_steps_ns = []
    car_gen_steps_sn = []
    car_gen_steps_we = []
    car_gen_steps_ew = []
    min_old_ns = math.floor(timings_ns[1])
    max_old_ns = math.ceil(timings_ns[-1])
    min_old_sn = math.floor(timings_sn[1])
    max_old_sn = math.ceil(timings_sn[-1])
    min_old_we = math.floor(timings_we[1])
    max_old_we = math.ceil(timings_we[-1])
    min_old_ew = math.floor(timings_ew[1])
    max_old_ew = math.ceil(timings_ew[-1])
    min_new = 0
    max_new = simulation_steps
    for i in range(len(timings_ns)):
        car_gen_steps_ns.append(((max_new - min_new) / (max_old_ns - min_old_ns)) * (timings_ns[i] - max_old_ns) + max_new)
        car_gen_steps_sn.append(((max_new - min_new) / (max_old_sn - min_old_sn)) * (timings_sn[i] - max_old_sn) + max_new)
        car_gen_steps_we.append(((max_new - min_new) / (max_old_we - min_old_we)) * (timings_we[i] - max_old_we) + max_new)
        car_gen_steps_ew.append(((max_new - min_new) / (max_old_ew - min_old_ew)) * (timings_ew[i] - max_old_ew) + max_new)

    car_gen_steps_ns = np.rint(car_gen_steps_ns)  # 对时间进行取整
    car_gen_steps_sn = np.rint(car_gen_steps_sn)  
    car_gen_steps_we = np.rint(car_gen_steps_we) 
    car_gen_steps_ew = np.rint(car_gen_steps_ew) 

    curr_path = os.path.dirname(os.path.abspath(__file__))
    rou_file_name = path+'/intersection' + str(train_eps) + '.rou.xml'
    rou_cfg_file = os.path.join(curr_path, rou_file_name)
    #print(rou_cfg_file)
   
    with open(rou_cfg_file, "w") as route:
        #EV的能耗是Wh每秒 carFollowModel="IDM"
        print("""<routes>

        <vType id = 'ego_car' vclass="evehicle" emissionClass="Energy" tau="1.5" accel="3.0" decel="3.0" color="#00FFFF" sigma="0.2" length="4.90" minGap="1.0" maxSpeed="20" carFollowing="IDM" guiShape="passenger"/>

        <route id="N2S" edges="NS -SN"/>
        <route id="S2N" edges="SN -NS"/>
        <route id="W2E" edges="WE -EW"/>
        <route id="E2W" edges="EW -WE"/>""", file=route)
        depart_list = []
        for i in range(car_count_per_lane):
            #随机选择一个车辆行驶的方向，随机选择车辆的类型
            depart_list.append(['N2S', car_gen_steps_ns[i]])           
            depart_list.append(['S2N', car_gen_steps_sn[i]])
            depart_list.append(['W2E', car_gen_steps_we[i]])
            depart_list.append(['E2W', car_gen_steps_ew[i]])
        depart_list = sorted(depart_list, key = lambda x:x[1]) #按照出发的时间进行排序
        for i in range(car_count_per_lane * 4):
            #现在选择是human-drive还是autonomous-drive
            if random.random() <=  cav_penetration:
                print('   <vehicle id="%s_%i" type="ego_car" route="%s" depart="%i" departSpeed="10" departLane="best"/>' % (depart_list[i][0], i + 1, depart_list[i][0], depart_list[i][1]), file = route)   
            else:
                #为人驾驶的汽车
                print('   <vehicle id="h%s_%i" type="ego_car" route="%s" depart="%i" departSpeed="10" departLane="best"/>' % (depart_list[i][0], i + 1, depart_list[i][0], depart_list[i][1]), file = route)             
        print('</routes>', file=route)

    