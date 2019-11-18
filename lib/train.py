import torch
import numpy as np
import time

def train(episodes, env, net, target_net, epsilon_data, agent, memory, gamma, device,
          LEARNING_STARTS, TARGET_UPDATE_FREQ, batch_size):
    # main loop
    frame_num = 0
    prev_input = None

    # initialization of variables used in the main loop
    reward_sum = 0
    total_rewards = []
    start = time.time()
    timestep_frame = 0
    best_mean_reward = None
    mean_reward_bound = 19.5

    for episode_nb in range(episodes):
        while True:
            frame_num += 1
            epsilon = max(epsilon_data[0], epsilon_data[1] - frame_num / epsilon_data[2])

            reward = agent.play_action(net, epsilon, device)
            if reward is not None:
                total_rewards.append(reward)
                speed = (frame_num - timestep_frame) / (time.time() - start)
                timestep_frame = frame_num
                start = time.time()
                mean_reward = np.mean(total_rewards[-100:])
                print("{} frames: done {} games, mean reward {}, eps {}, speed {} f/s".format(
                    frame_num, len(total_rewards), round(mean_reward, 3), round(epsilon, 2), round(speed, 2)))

                if best_mean_reward is None or best_mean_reward < mean_reward or len(total_rewards) % 25 == 0:
                    torch.save(net.state_dict(), "./data/Pong-v0" + "-" + str(len(total_rewards)) + ".dat")
                    if best_mean_reward is not None:
                        print("New best mean reward {} -> {}, model saved".format(round(best_mean_reward, 3),
                                                                                  round(mean_reward, 3)))
                    best_mean_reward = mean_reward

                if mean_reward > mean_reward_bound and len(total_rewards) > 10:
                    print("Game solved in {} frames! Average score of {}".format(frame_num, mean_reward))
                    break

                if len(memory.buffer) < LEARNING_STARTS:
                    continue

                if frame_num % TARGET_UPDATE_FREQ == 0:
                    target_net.load_state_dict(net.state_dict())

                net.optimizer.zero_grad()
                batch = memory.sample(batch_size)
                loss_t = net.calculate_loss(batch, net, target_net, gamma, device)
                loss_t.backward()
                net.optimizer.step()
            env.close()