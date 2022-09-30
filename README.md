# drl-eco-driving
A Deep Reinforcement Learning-based vehicle scheduling algorithm at the signalized intersection.

**state**
The CAV can obtain a observation $s_t$ from the enviroment at time-step t is a 10-dimention vector.

**action**

**reward**
our reward function contains four sub-reward items: 
1)reward for vehicle speed: $r_{speed}$

2)reward for acceleration $r_{accel}$ 

3)reward for pass the green light $r_{tls}$

4)reward for driving safety $r_{safe}$

