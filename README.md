# DRL-based eco-driving
This is a Deep Reinforcement Learning-based vehicle scheduling algorithm at the signalized intersection. In our sheduling algorithm, each CAV is regarded as an agent.The agent have three goals:

1) Reducing the energy consumption for passing the intersection of the CAV. 

2) Minimizing the time for passing the intersection of the CAV.

3) Ensuring driving safety.

**state**

The CAV can obtain a observation $s_t$ from the enviroment at time-step t is a 8-dimention vector:
$$
\boldsymbol{o}_{\boldsymbol{t}}=\left(v_{lead}, t_{flag}, t_{dur}, d_{inter}, v_{ego}, a_{ego}, f_{lead}, d_{lead}\right)
$$
1) V2X: leading vehicle’s speed $v_{lead}$, current traffic phase $t_{flag}$, rest time of the current traffic phase $t_{dur}$, the distance to the stop line of the intersection $d_{inter}$. 

2) On-Board Diagnosis (OBD) Unit: ego vehicle’s speed $v_{ego}$ and acceleration $a_{ego}$. 

3) Vehicle-mounted sensors: $f_{lead}$ indicates whether there is a vehicle ahead of the ego-vehicle. $d_{lead}$ denotes the distance between the ego-vehicle and the leading vehicle. 

**action**

Obviously, controlling the vehicle’s acceleration discretely will damage driving stability. Therefore, we choose to generate the vehicle’s acceleration $a_{ego}$ from a continuous acceleration space [−2, 2] $m/s^2$.

**reward**

our reward function contains four sub-reward items: 

1)reward for vehicle speed $r_{speed}$

2)reward for acceleration $r_{accel}$

3)reward for pass the green light $r_{tls}$

4)reward for driving safety $r_{safe}$

So the total reward of the CAV can obtain at time-step $t$ is:
$$
r=r_{tls}+f_1 \cdot r_{accel}+f_2 \cdot r_{speed}+r_{safe}
$$

**How to run our code**
Environment configuration

simulation platform: sumo, python == 3.9.7, pytorch == 1.12.1, traci == 1.13.0