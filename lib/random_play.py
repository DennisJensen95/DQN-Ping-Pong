import time
import random

def play_random(env, UP_ACTION, DOWN_ACTION, seconds):
    # main loop
    timeout = seconds
    start = time.time()
    while True:
        # render a frame
        env.render()

        # choose random action
        action = random.randint(UP_ACTION, DOWN_ACTION)

        # run one step
        observation, reward, done, info = env.step(action)

        # if the episode is over, reset the environment
        if done:
            env.reset()

        if time.time() - start > timeout:
            print("Times up!")
            break