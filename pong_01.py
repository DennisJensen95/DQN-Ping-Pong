import gym
import sys
import torch
from lib.random_play import play_random
from lib.helper_functions import check_arg_sys_input
from lib.train import train
from lib.Memory import Memory
from lib.Agent import Agent
from lib.atari_wrappers import make_env
from lib.DQN_Network import DQN
from lib.Test_old_network import test_old_network
option_dict = check_arg_sys_input()

UP_ACTION = 2
DOWN_ACTION = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Epislon greedy parameters
EPSILON_DECAY = 10**6
EPSILON_START = 1.0
EPSILON_FINAL = 0.02
epsilon_data = [EPSILON_FINAL, EPSILON_START, EPSILON_DECAY]


# Hyperparameters
learning_rate = 1e-4
REPLAY_SIZE = 10 ** 4 * 4
BATCH_SIZE = 32
TARGET_UPDATE_FREQ = 1000
DELAY_LEARNING = 20000
GAMMA = 0.99

model = 'CDDQN'

# Environment and neural networks
env = make_env('Pong-v0')
env_test = gym.make('Pong-v0')
net = DQN(env.observation_space.shape, env.action_space.n, learning_rate).to(device)
target_net = DQN(env.observation_space.shape, env.action_space.n, learning_rate).to(device)

# Agent and memory handling
memory = Memory(REPLAY_SIZE)
agent = Agent(env, memory)

initial_observation = env.reset()



if 'cuda' in str(device):
    print('The GPU is being used')
else:
    print('The CPU is being used')

if option_dict['random']:
    play_random(env, UP_ACTION, DOWN_ACTION, seconds=5)

if option_dict['train']:
    print("Training")
    print("ReplayMemory will require {}gb of GPU RAM".format(round(REPLAY_SIZE * 32 * 84 * 84 / 1e+9, 2)))
    agent.reset_environtment()
    train(env, net, target_net, epsilon_data, agent, memory, GAMMA, device,
                DELAY_LEARNING, TARGET_UPDATE_FREQ, BATCH_SIZE, model)

if option_dict['oldnetwork']:
    file_path = './pull/Best_performing_model/DDQN/DDQN-5233.dat'
    seconds = 120
    test_old_network(env, net, file_path, seconds, device)




