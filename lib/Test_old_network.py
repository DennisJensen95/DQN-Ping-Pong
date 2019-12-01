import time
import numpy as np
import torch
import os

def test_old_network(env, net, file_path, seconds, device):
    # main loop
    timeout = seconds
    assert(os.path.exists(file_path)), "The filepath does not exist."

    net.load_state_dict(torch.load(file_path))

    start = time.time()
    new_state, reward, done, _ = env.reset()
    new_state, reward, done, _ = env.step(env.action_space.sample())
    while True:
        # render a frame
        env.render()

        # choose random action
        # action = random.randint(UP_ACTION, DOWN_ACTION)
        state = np.array([new_state], copy=False)
        state = torch.tensor(state).to(device)
        q_values = net(state)
        _, pref_action = torch.max(q_values, dim=1)
        action = int(pref_action.item())
        # run one step
        new_state, reward, done, info = env.step(action)
        time.sleep(0.05)

        # if the episode is over, reset the environment
        if done:
            env.reset()

        if time.time() - start > timeout:
            print("Times up!")
            break