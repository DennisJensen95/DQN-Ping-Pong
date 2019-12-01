import re
import numpy as np
import matplotlib.pyplot as plt


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
        print(data_point)
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

path = './../pull/data/frames_reward_3'
# path = './../data/frames_reward.dat'
frame, reward, e = get_data(path)
frame = frame[1:]
print(max(reward))
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
