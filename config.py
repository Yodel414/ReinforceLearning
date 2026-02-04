import argparse
from typing import Union
import numpy as np
def test_config1():
    env_size = (3,3)
    start_state=(0,0) 
    target_state=(2,2)
    forbidden_states=[(2,1),(0,2)]
    reward_target = 1
    reward_forbidden = -10
    reward_step = 0
    reward_boundary = -1
    parser = argparse.ArgumentParser("Grid World Environment")
    parser.add_argument("--env-size", type=Union[list, tuple, np.ndarray], default=env_size )   
    parser.add_argument("--start-state", type=Union[list, tuple, np.ndarray], default=start_state)
    parser.add_argument("--target-state", type=Union[list, tuple, np.ndarray], default=target_state)
    parser.add_argument("--forbidden-states", type=list, default=forbidden_states)
    parser.add_argument("--reward-target", type=float, default = reward_target)
    parser.add_argument("--reward_boundary", type=float, default = reward_boundary)
    parser.add_argument("--reward-forbidden", type=float, default = reward_forbidden)
    parser.add_argument("--reward-step", type=float, default = reward_step)
    parser.add_argument("--action-space", type=list, default=[(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)] )  # up, right, down, left, stay  
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--animation-interval", type=float, default = 0.2)
    return parser.parse_args()

def test_config2():
    env_size = (5,5)
    start_state=(0,0) 
    target_state=(2,3)
    forbidden_states=[ (1, 1), (2, 1), (2, 2),(1, 3),(3, 3),(1, 4)]
    reward_target = 1
    reward_forbidden = -10
    reward_step = 0
    reward_boundary = -1
    parser = argparse.ArgumentParser("Grid World Environment")
    parser.add_argument("--env-size", type=Union[list, tuple, np.ndarray], default=env_size )   
    parser.add_argument("--start-state", type=Union[list, tuple, np.ndarray], default=start_state)
    parser.add_argument("--target-state", type=Union[list, tuple, np.ndarray], default=target_state)
    parser.add_argument("--forbidden-states", type=list, default=forbidden_states)
    parser.add_argument("--reward-target", type=float, default = reward_target)
    parser.add_argument("--reward_boundary", type=float, default = reward_boundary)
    parser.add_argument("--reward-forbidden", type=float, default = reward_forbidden)
    parser.add_argument("--reward-step", type=float, default = reward_step)
    parser.add_argument("--action-space", type=list, default=[(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)] )  # up, right, down, left, stay           
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--animation-interval", type=float, default = 0.2)
    return parser.parse_args()

def GetConfig(type):
    if type == 1:
        return test_config1()
    if type == 2:
        return test_config2()
