import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker


def get_data(filename):
    x = []
    y = []
    e = []
    file = open(filename, 'r+')
    data = file.read()
    file.close()
    lines = data.split('\n')

    for line in lines:
        data_point = re.findall('(-?[\d.]+)', line)
        # print(data_point)
        if len(data_point) > 2:
            x_result, y_result, e_result = data_point[0], data_point[1], data_point[2]
            x.append(x_result)
            y.append(y_result)
            e.append(e_result)
        elif len(data_point) > 1:
            x_result, y_result = data_point[0], data_point[1]
            x.append(x_result)
            y.append(y_result)

    x = np.array(x).astype(np.float)
    y = np.array(y).astype(np.float)
    e = np.array(e).astype(np.float)
    return x, y, e

def plot_one():
    path = './../pull/data/frames_reward_8'
    # path = './../data/frames_reward.dat'
    frame, reward, e = get_data(path)
    frame = frame[1:]
    # print(max(reward))
    reward = reward[1:]
    e = e[1:]
    plt.figure()
    plt.plot(frame, reward)
    plt.ylabel('Reward')
    plt.xlabel('Number of frames')
    if len(e) > 1:
        plt.figure()
        plt.plot(frame, e)
        plt.xlabel('Number of frames')
        plt.ylabel('Epsilon')

    plt.show()

def millions(x, pos):
    'The two args are the value and tick position'
    return '%1.1fM' % (x * 1e-6)

def plot_epsilon():
    DDQN_path = './../pull/Best_performing_model/DDQN/DDQN_frames_reward'

    x, y, e = get_data(DDQN_path)
    fig, ax = plt.subplots()

    e = e[:200]
    x = x[:200]
    plt.plot(x, e)
    plt.xlabel('Number of frames')
    plt.ylabel('Epsilon value')
    # formatter = ticker.FuncFormatter(millions)
    # ax.xaxis.set_major_formatter(formatter)

    plt.show()

def plot_multiple():
    """"""
    DDQN_path = './../pull/pong_v4_data/frames_reward'
    DQN_path = './../pull/pong_v4_data/frames_reward_1'
    CDQN_path = './../pull/pong_v4_data/frames_reward_2'

    paths = [DQN_path, DDQN_path, CDQN_path]

    # path = './../data/frames_reward.dat'
    i = 0
    max_frame = 0
    min_frame = 0
    for file in paths:
        frame, reward, e = get_data(file)
        frame = frame[1:]
        reward = reward[1:]
        if max(frame) > max_frame:
            max_frame = max(frame)
        if min(frame) < min_frame:
            min_frame = min(frame)

        print(max_frame/len(frame))

        print(f'{max(reward)}')

        if i == 0:
            fig, ax = plt.subplots()
        x = []
        # frame = millions(frame)
        plt.plot(frame, reward)
        plt.ylabel('Reward')
        plt.xlabel('Number of frames')
        i += 1

    plt.hlines(-3, min_frame, max_frame)
    formatter = ticker.FuncFormatter(millions)

    plt.legend(['DQN', 'Double DQN', 'Clipped Double DQN', 'Average Human Performance'])
    ax.xaxis.set_major_formatter(formatter)

    plt.show()

if __name__ == '__main__':
    plot_multiple()
    # plot_epsilon()