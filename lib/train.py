import torch
import numpy as np
import time
import os
import re

def name_new(file, num=0):
    if os.path.exists(file) and num==0:
        return name_new(file + f'_{num+1}', num+1)
    else:
        if os.path.exists(file):
            # print("Here?")
            remove_str = re.findall(f'_\d+', file)[0]
            # print(remove_str)
            new_name = file.replace(remove_str, '') + f'_{num+1}'
            return name_new(new_name, num+1)
        else:
            # print(file)
            return file

def update(batch_size, memory, net, target_net, gamma, model, device):
    if model == 'DDQN' or model == 'DQN':
        net.optimizer.zero_grad()
        batch = memory.sample(batch_size)
        loss_t = net.calculate_loss(batch, net, target_net, gamma, model, device)
        loss_t.backward()
        net.optimizer.step()
    elif model == 'CDDQN':
        batch = memory.sample(batch_size)
        loss_1, loss_2 = net.calculate_loss(batch, net, target_net, gamma, model, device)

        net.optimizer.zero_grad()
        loss_1.backward()
        net.optimizer.step()

        target_net.optimizer.zero_grad()
        loss_2.backward()
        target_net.optimizer.step()



def train(env, net, target_net, epsilon_data, agent, memory, gamma, device,
          LEARNING_STARTS, TARGET_UPDATE_FREQ, batch_size, model):
    # main loop
    frame_num = 0
    prev_input = None

    # initialization of variables used in the main loop
    reward_sum = 0
    total_rewards = []
    start = time.time()
    timestep_frame = 0
    best_mean_reward = None
    mean_reward_bound = 20.5
    freq_saving_reward = 1000
    save_reward = False
    # print(os.getcwd())
    filename = './pong_v4/frames_reward'
    file_name = name_new(filename)
    # print(file_name)
    name_to_save = model

    while True:
        frame_num += 1
        epsilon = max(epsilon_data[0], epsilon_data[1] - frame_num / epsilon_data[2])

        reward = agent.play_action(net, epsilon, device)

        if frame_num & freq_saving_reward == 0:
            save_reward = True

        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_num - timestep_frame) / (time.time() - start)
            timestep_frame = frame_num
            start = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            print("{} frames: done {} games, mean reward {}, eps {}, speed {} f/s".format(
                frame_num, len(total_rewards), round(mean_reward, 3), round(epsilon, 2), round(speed, 2)))

            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), f'./pong_v4/{name_to_save}_10_6' + "-" + str(len(total_rewards)) + ".dat")
                if best_mean_reward is not None:
                    print("New best mean reward {} -> {}, model saved".format(round(best_mean_reward, 3),
                                                                              round(mean_reward, 3)))
                best_mean_reward = mean_reward

            if mean_reward > mean_reward_bound and len(total_rewards) > 10:
                print("Game solved in {} frames! Average score of {}".format(frame_num, mean_reward))
                break

            if save_reward:
                with open(file_name, 'a') as file:
                    file.write(f'{frame_num}:{round(mean_reward, 2)}:{round(epsilon, 2)}\n')

                save_reward = False

        if len(memory.buffer) < LEARNING_STARTS:
            continue

        if frame_num % TARGET_UPDATE_FREQ == 0 and model == 'DDQN':
            target_net.load_state_dict(net.state_dict())

        # Update depending on model
        update(batch_size, memory, net, target_net, gamma, model, device)

    env.close()