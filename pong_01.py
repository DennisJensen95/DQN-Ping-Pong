import gym
import torch
from lib.random_play import play_random
from lib.helper_functions import check_arg_sys_input
from lib.DQN_Network import DQN
from lib.train import train
option_dict = check_arg_sys_input()

UP_ACTION = 2
DOWN_ACTION = 3

env = gym.make("Pong-v0")

initial_observation = env.reset()

learning_rate = 0.002

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if 'cuda' in str(device):
    print('The GPU is being used')
else:
    print('The CPU is being used')

net = DQN(learning_rate).to(device)
episodes = 1000

if option_dict['random']:
    play_random(env, UP_ACTION, DOWN_ACTION, seconds=5)

if option_dict['train']:
    print("Training")
    train(episodes, env, net, UP_ACTION, DOWN_ACTION, initial_observation, device)



