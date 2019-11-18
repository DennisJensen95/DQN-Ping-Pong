from lib.ImageProcess import prepro, discount_rewards
import torch
import numpy as np
from torch.autograd import Variable


def train(episodes, env, model, UP_ACTION, DOWN_ACTION, initial_observation, device):
    # main loop
    prev_input = None
    observation = initial_observation

    # initialization of variables used in the main loop
    reward_sum = 0

    # Turn into double
    model = model.double()

    # Hyperparameters
    gamma = 0.99

    for episode_nb in range(episodes):
        while True:
            # 1. preprocess the observation, set input as difference between images
            cur_input = prepro(observation)
            x = cur_input - prev_input if prev_input is not None else np.zeros(80 * 80)
            prev_input = cur_input
            x = torch.from_numpy(x).to(device)

            # 2. forward the network on the image pixels
            model.optimizer.zero_grad()
            proba = model(x)

            # 3. Select action depending on probability of action.
            action = UP_ACTION if np.random.uniform() < proba else DOWN_ACTION
            y = 1 if action == 2 else 0  # 0 and 1 are our labels

            # 5. do one step in our environment
            observation, reward, done, info = env.step(action)
            reward_sum += reward

            # end of an episode
            if done:
                print('At the end of episode', episode_nb, 'the total reward was :', reward_sum)

                cur_input = prepro(observation)
                x = cur_input - prev_input if prev_input is not None else np.zeros(80 * 80)
                prev_input = cur_input
                x = torch.from_numpy(x).to(device)

                # training
                with torch.no_grad():
                    Q1 = model(x)

                q_target = Q1.clone()
                q_target = reward + gamma * q_target.max().item()

                loss = model.loss(proba, q_target)
                loss.backward()
                model.optimizer.step()

                print(f'Loss is: {loss}')

                # Reinitialization
                observation = env.reset()
                reward_sum = 0
                prev_input = None

                break