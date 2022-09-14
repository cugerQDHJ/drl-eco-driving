import random


class ReplayBuffer:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state_dict, action_dict, reward_dict, next_state_dict):
        for key in state_dict.keys():
            if key in next_state_dict.keys():
                if len(self.buffer) < self.capacity:
                    self.buffer.append(None)
                #print('state:{}, action:{}, reward:{}, next_state:{}'.format(state_dict[key], action_dict[key], reward_dict[key][0], next_state_dict[key]))
                self.buffer[self.position] = (state_dict[key], action_dict[key], reward_dict[key][0], next_state_dict[key])
                self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state =  zip(*batch)
        return state, action, reward, next_state
    
    def __len__(self):
        return len(self.buffer)
