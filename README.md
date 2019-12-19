{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Reinforcement Learning applied to an Atari Pong game\n",
    "\n",
    "By Dennis Jensen s155629 & Stefan Carius Larsen s164029\n",
    "\n",
    "This notebook displays and shows results of the implemented Deep Q-Networks applied to the atari game pong.\n",
    "Throughout the project three different models were applied and trained on the pong game. Each model implements a neural network which consists of a convolutional network with three layers and two dense layers. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "With opengym.ai an atari pong game setup is supplied with a reward system. With the gym module you have an agent playing against a standard computer playing. The agent gets rewarded if it wins a round and gets penalized if it looses a round. One win gives +1 and a lost gives -1. \n",
    "\n",
    "<img src=\"pong_rgb.png\">\n",
    "\n",
    "## Scaling image, Grayscale image and Skipping Frames\n",
    "For scaling, grayscaling and skipping frames the OpenAi Gym Wrappers are used.\n",
    "\n",
    "### OpenAi Gym Wrappers\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "import gym\n",
    "import collections\n",
    "import numpy as np\n",
    "import cv2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Taken from OpenAI baseline wrappers\n",
    "# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py\n",
    "\n",
    "class FireResetEnv(gym.Wrapper):\n",
    "    def __init__(self, env=None):\n",
    "        \"\"\"Take action on reset for environments that are fixed until firing.\"\"\"\n",
    "        super(FireResetEnv, self).__init__(env)\n",
    "        print(env.unwrapped.get_action_meanings())\n",
    "        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'\n",
    "        assert len(env.unwrapped.get_action_meanings()) >= 3\n",
    "\n",
    "    def step(self, action):\n",
    "        return self.env.step(action)\n",
    "\n",
    "    def reset(self):\n",
    "        self.env.reset()\n",
    "        obs, _, done, _ = self.env.step(1)\n",
    "        if done:\n",
    "            self.env.reset()\n",
    "        obs, _, done, _ = self.env.step(2)\n",
    "        if done:\n",
    "            self.env.reset()\n",
    "        return obs\n",
    "\n",
    "\n",
    "class MaxAndSkipEnv(gym.Wrapper):\n",
    "    def __init__(self, env=None, skip=4):\n",
    "        \"\"\"Return only every `skip`-th frame\"\"\"\n",
    "        super(MaxAndSkipEnv, self).__init__(env)\n",
    "        # most recent raw observations (for max pooling across time steps)\n",
    "        self._obs_buffer = collections.deque(maxlen=2)\n",
    "        self._skip = skip\n",
    "\n",
    "    def step(self, action):\n",
    "        total_reward = 0.0\n",
    "        done = None\n",
    "        for _ in range(self._skip):\n",
    "            obs, reward, done, info = self.env.step(action)\n",
    "            self._obs_buffer.append(obs)\n",
    "            total_reward += reward\n",
    "            if done:\n",
    "                break\n",
    "        max_frame = np.max(np.stack(self._obs_buffer), axis=0)\n",
    "        return max_frame, total_reward, done, info\n",
    "\n",
    "    def reset(self):\n",
    "        \"\"\"Clear past frame buffer and init to first obs\"\"\"\n",
    "        self._obs_buffer.clear()\n",
    "        obs = self.env.reset()\n",
    "        self._obs_buffer.append(obs)\n",
    "        return obs\n",
    "\n",
    "\n",
    "class ProcessFrame84(gym.ObservationWrapper):\n",
    "    \"\"\"\n",
    "    Downsamples image to 84x84\n",
    "    Greyscales image\n",
    "\n",
    "    Returns numpy array\n",
    "    \"\"\"\n",
    "    def __init__(self, env=None):\n",
    "        super(ProcessFrame84, self).__init__(env)\n",
    "        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)\n",
    "\n",
    "    def observation(self, obs):\n",
    "        return ProcessFrame84.process(obs)\n",
    "\n",
    "    @staticmethod\n",
    "    def process(frame):\n",
    "        if frame.size == 210 * 160 * 3:\n",
    "            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)\n",
    "        elif frame.size == 250 * 160 * 3:\n",
    "            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)\n",
    "        else:\n",
    "            assert False, \"Unknown resolution.\"\n",
    "        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114\n",
    "        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)\n",
    "        x_t = resized_screen[18:102, :]\n",
    "        x_t = np.reshape(x_t, [84, 84, 1])\n",
    "        return x_t.astype(np.uint8)\n",
    "\n",
    "\n",
    "class ImageToPyTorch(gym.ObservationWrapper):\n",
    "    def __init__(self, env):\n",
    "        super(ImageToPyTorch, self).__init__(env)\n",
    "        old_shape = self.observation_space.shape\n",
    "        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),\n",
    "                                                dtype=np.float32)\n",
    "\n",
    "    def observation(self, observation):\n",
    "        return np.moveaxis(observation, 2, 0)\n",
    "\n",
    "\n",
    "class ScaledFloatFrame(gym.ObservationWrapper):\n",
    "    \"\"\"Normalize pixel values in frame --> 0 to 1\"\"\"\n",
    "    def observation(self, obs):\n",
    "        return np.array(obs).astype(np.float32) / 255.0\n",
    "\n",
    "\n",
    "class BufferWrapper(gym.ObservationWrapper):\n",
    "    def __init__(self, env, n_steps, dtype=np.float32):\n",
    "        super(BufferWrapper, self).__init__(env)\n",
    "        self.dtype = dtype\n",
    "        old_space = env.observation_space\n",
    "        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),\n",
    "                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)\n",
    "\n",
    "    def reset(self):\n",
    "        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)\n",
    "        return self.observation(self.env.reset())\n",
    "\n",
    "    def observation(self, observation):\n",
    "        self.buffer[:-1] = self.buffer[1:]\n",
    "        self.buffer[-1] = observation\n",
    "        return self.buffer\n",
    "\n",
    "\n",
    "def make_env(env_name):\n",
    "    env = gym.make(env_name)\n",
    "    env = MaxAndSkipEnv(env)\n",
    "    env = FireResetEnv(env)\n",
    "    env = ProcessFrame84(env)\n",
    "    env = ImageToPyTorch(env)\n",
    "    env = BufferWrapper(env, 4)\n",
    "    return ScaledFloatFrame(env)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## The architechture of the DQN, DDQN, CDDQN\n",
    "\n",
    "Our Deep-Q-Network is made up of:\n",
    "\n",
    "3 convolution layers\n",
    "2 dense (fully connected) layers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "from torch import nn\n",
    "from torch import optim"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "class DQN(nn.Module):\n",
    "\n",
    "    def __init__(self, input_shape, n_actions, learning_rate):\n",
    "        super(DQN, self).__init__()\n",
    "\n",
    "        channels = input_shape[0]\n",
    "\n",
    "        # First conv layer\n",
    "        conv_out_channels = 32  # <-- Filters in your convolutional layer\n",
    "        kernel_size = 8  # <-- Kernel size\n",
    "        conv_stride = 4  # <-- Stride\n",
    "        conv_pad = 0  # <-- Padding\n",
    "\n",
    "        # Second conv layer\n",
    "        conv_out_channels_2 = 64  # <-- Filters in your convolutional layer\n",
    "        kernel_size_2 = 4  # <-- Kernel size\n",
    "        conv_stride_2 = 2  # <-- Stride\n",
    "        conv_pad_2 = 0  # <-- Padding\n",
    "\n",
    "        # Third conv layer\n",
    "        conv_out_channels_3 = 64  # <-- Filters in your convolutional layer\n",
    "        kernel_size_3 = 3  # <-- Kernel size\n",
    "        conv_stride_3 = 1  # <-- Stride\n",
    "        conv_pad_3 = 0  # <-- Padding\n",
    "\n",
    "        self.conv = nn.Sequential(\n",
    "            nn.Conv2d(channels, conv_out_channels, kernel_size, conv_stride, conv_pad),\n",
    "            nn.ReLU(),\n",
    "            nn.Conv2d(conv_out_channels, conv_out_channels_2, kernel_size_2, conv_stride_2, conv_pad_2),\n",
    "            nn.ReLU(),\n",
    "            nn.Conv2d(conv_out_channels_3, conv_out_channels_3, kernel_size_3, conv_stride_3, conv_pad_3),\n",
    "            nn.ReLU()\n",
    "        )\n",
    "\n",
    "\n",
    "        out_features = self.conv_out_features(input_shape)\n",
    "\n",
    "        self.out = nn.Sequential(\n",
    "            nn.Linear(out_features, 512),\n",
    "            nn.ReLU(),\n",
    "            nn.Linear(512, n_actions)\n",
    "        )\n",
    "\n",
    "        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)\n",
    "\n",
    "    def forward(self, x):\n",
    "        x = self.conv(x)\n",
    "        x = x.view(x.size()[0], -1)\n",
    "        return self.out(x)\n",
    "\n",
    "    def conv_out_features(self, shape):\n",
    "        o = self.conv(torch.zeros(1, *shape))\n",
    "        return int(np.prod(o.size()))\n",
    "\n",
    "\n",
    "    def calculate_loss(self, batch, net, target_net, GAMMA, model, device=\"cpu\"):\n",
    "        \"\"\"\n",
    "        This calculate loss function computes loss in three different ways depending on model chosen.\n",
    "        DQN - One network computing next state and current state\n",
    "        DDQN - Target network computing next state and training network computing current state\n",
    "        CDDQN - Seperate networks computes both next state and current state and then loss is computed from the\n",
    "        minimum of the two next states\n",
    "        \"\"\"\n",
    "        states, actions, rewards, dones, next_states = batch\n",
    "\n",
    "        states_v = torch.tensor(states).to(device)\n",
    "        next_states_v = torch.tensor(next_states).to(device)\n",
    "        actions_v = torch.tensor(actions).to(device)\n",
    "        rewards_v = torch.tensor(rewards).to(device)\n",
    "        done = torch.tensor(dones).to(device)\n",
    "\n",
    "        if model == 'DQN' or model == 'DDQN':\n",
    "            if model == 'DQN':\n",
    "                state_action_values = net(states_v).gather(1, actions_v.long().unsqueeze(-1)).squeeze(-1)\n",
    "                next_state_values = net(next_states_v).max(1)[0]\n",
    "            elif model == 'DDQN':\n",
    "                state_action_values = net(states_v).gather(1, actions_v.long().unsqueeze(-1)).squeeze(-1)\n",
    "                next_state_values = target_net(next_states_v).max(1)[0]\n",
    "\n",
    "            next_state_values[done] = 0.0\n",
    "            next_state_values = next_state_values.detach()\n",
    "\n",
    "            expected_state_action_values = next_state_values * GAMMA + rewards_v\n",
    "\n",
    "            return nn.MSELoss()(state_action_values, expected_state_action_values)\n",
    "\n",
    "        elif model == 'CDDQN':\n",
    "            state_Q1 = net(states_v).gather(1, actions_v.long().unsqueeze(-1)).squeeze(-1)\n",
    "            state_Q2 = target_net(states_v).gather(1, actions_v.long().unsqueeze(-1)).squeeze(-1)\n",
    "\n",
    "            next_Q1 = net(next_states_v).max(1)[0]\n",
    "            next_Q2 = target_net(next_states_v).max(1)[0]\n",
    "            # print(next_Q2)\n",
    "            next_Q = torch.min(\n",
    "                next_Q1,\n",
    "                next_Q2\n",
    "            )\n",
    "\n",
    "            # next_Q = next_Q.view(next_Q.size(0), 1)\n",
    "            next_Q[done] = 0.0\n",
    "            next_Q = next_Q.detach()\n",
    "\n",
    "            expected_Q = rewards_v + GAMMA * next_Q\n",
    "            # expected_Q = expected_Q.detach()\n",
    "\n",
    "            loss_1 = nn.MSELoss()(state_Q1, expected_Q)\n",
    "            loss_2 = nn.MSELoss()(state_Q2, expected_Q)\n",
    "\n",
    "            return loss_1, loss_2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Memory Class for Experience Replay\n",
    "\n",
    "The memory is for being able to learn from uncorrelated data points."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'done', 'new_state'))\n",
    "\n",
    "class Memory():\n",
    "    def __init__(self, capacity):\n",
    "        self.buffer = []\n",
    "        self.capacity = capacity\n",
    "        self.position = 0\n",
    "\n",
    "    def save_to_memory(self, *args):\n",
    "        if len(self.buffer) < self.capacity:\n",
    "            self.buffer.append(None)\n",
    "        self.buffer[self.position] = Transition(*args)\n",
    "        self.position = (self.position + 1) % self.capacity\n",
    "\n",
    "    def sample(self, batch_size):\n",
    "        indices = np.random.choice(len(self.buffer), batch_size, replace=False)\n",
    "        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])\n",
    "\n",
    "        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \\\n",
    "               np.array(dones, dtype=np.bool), np.array(next_states)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Agent Class \n",
    "\n",
    "The agent class is for playing the game and saving the plays to the memory to learn from it later."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Agent():\n",
    "    def __init__(self, env, memory):\n",
    "        self.env = env\n",
    "        self.memory = memory\n",
    "        self.env_state = None\n",
    "        self.env_set = False\n",
    "\n",
    "    def reset_environtment(self):\n",
    "        self.env_state = self.env.reset()\n",
    "        self.env_set = True\n",
    "        self.total_reward = 0.0\n",
    "\n",
    "    def random_epsilon_greedy(self, network, device):\n",
    "        assert (self.env_set), \"Please initialize game before playing by resetting or inputting a state\"\n",
    "        state = np.array([self.env_state], copy=False)\n",
    "        state = torch.tensor(state).to(device)\n",
    "        q_values = network(state)\n",
    "        _, pref_action = torch.max(q_values, dim=1)\n",
    "        action = int(pref_action.item())\n",
    "        return action\n",
    "\n",
    "    def play_action(self, network, e=0, device='cpu'):\n",
    "        done_reward = None\n",
    "        if np.random.random() < e:\n",
    "            action = self.env.action_space.sample()\n",
    "        else:\n",
    "            action = self.random_epsilon_greedy(network, device)\n",
    "\n",
    "        # Do action in the environment\n",
    "        new_state, reward, done, _ = self.env.step(action)\n",
    "        self.total_reward += reward\n",
    "        old_state = self.env_state\n",
    "        self.env_state = new_state\n",
    "\n",
    "        self.memory.save_to_memory(old_state, action, reward, done, self.env_state)\n",
    "\n",
    "        if done:\n",
    "            done_reward = self.total_reward\n",
    "            self.reset_environtment()\n",
    "\n",
    "        return done_reward\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Training the models\n",
    "\n",
    "Depending on the model, there is a different approach for updaing and training the model. All these are implemented in the following functions.\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "import time\n",
    "import os\n",
    "import re"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [],
   "source": [
    "def name_new(file, num=0):\n",
    "    if os.path.exists(file) and num==0:\n",
    "        return name_new(file + f'_{num+1}', num+1)\n",
    "    else:\n",
    "        if os.path.exists(file):\n",
    "            # print(\"Here?\")\n",
    "            remove_str = re.findall(f'_\\d+', file)[0]\n",
    "            # print(remove_str)\n",
    "            new_name = file.replace(remove_str, '') + f'_{num+1}'\n",
    "            return name_new(new_name, num+1)\n",
    "        else:\n",
    "            # print(file)\n",
    "            return file\n",
    "\n",
    "def update(batch_size, memory, net, target_net, gamma, model, device):\n",
    "    if model == 'DDQN' or model == 'DQN':\n",
    "        net.optimizer.zero_grad()\n",
    "        batch = memory.sample(batch_size)\n",
    "        loss_t = net.calculate_loss(batch, net, target_net, gamma, model, device)\n",
    "        loss_t.backward()\n",
    "        net.optimizer.step()\n",
    "    elif model == 'CDDQN':\n",
    "        batch = memory.sample(batch_size)\n",
    "        loss_1, loss_2 = net.calculate_loss(batch, net, target_net, gamma, model, device)\n",
    "\n",
    "        net.optimizer.zero_grad()\n",
    "        loss_1.backward()\n",
    "        net.optimizer.step()\n",
    "\n",
    "        target_net.optimizer.zero_grad()\n",
    "        loss_2.backward()\n",
    "        target_net.optimizer.step()\n",
    "\n",
    "\n",
    "\n",
    "def train(env, net, target_net, epsilon_data, agent, memory, gamma, device,\n",
    "          LEARNING_STARTS, TARGET_UPDATE_FREQ, batch_size, model):\n",
    "    # main loop\n",
    "    frame_num = 0\n",
    "    prev_input = None\n",
    "\n",
    "    # initialization of variables used in the main loop\n",
    "    reward_sum = 0\n",
    "    total_rewards = []\n",
    "    start = time.time()\n",
    "    timestep_frame = 0\n",
    "    best_mean_reward = None\n",
    "    mean_reward_bound = 20.5\n",
    "    freq_saving_reward = 1000\n",
    "    save_reward = False\n",
    "    # print(os.getcwd())\n",
    "    filename = './data/frames_reward'\n",
    "    file_name = name_new(filename)\n",
    "    # print(file_name)\n",
    "    name_to_save = model\n",
    "\n",
    "    while True:\n",
    "        frame_num += 1\n",
    "        epsilon = max(epsilon_data[0], epsilon_data[1] - frame_num / epsilon_data[2])\n",
    "\n",
    "        reward = agent.play_action(net, epsilon, device)\n",
    "\n",
    "        if frame_num & freq_saving_reward == 0:\n",
    "            save_reward = True\n",
    "\n",
    "        if reward is not None:\n",
    "            total_rewards.append(reward)\n",
    "            speed = (frame_num - timestep_frame) / (time.time() - start)\n",
    "            timestep_frame = frame_num\n",
    "            start = time.time()\n",
    "            mean_reward = np.mean(total_rewards[-100:])\n",
    "            print(\"{} frames: done {} games, mean reward {}, eps {}, speed {} f/s\".format(\n",
    "                frame_num, len(total_rewards), round(mean_reward, 3), round(epsilon, 2), round(speed, 2)))\n",
    "\n",
    "            if best_mean_reward is None or best_mean_reward < mean_reward:\n",
    "                torch.save(net.state_dict(), f'./data/{name_to_save}_10_6' + \"-\" + str(len(total_rewards)) + \".dat\")\n",
    "                if best_mean_reward is not None:\n",
    "                    print(\"New best mean reward {} -> {}, model saved\".format(round(best_mean_reward, 3),\n",
    "                                                                              round(mean_reward, 3)))\n",
    "                best_mean_reward = mean_reward\n",
    "\n",
    "            if mean_reward > mean_reward_bound and len(total_rewards) > 10:\n",
    "                print(\"Game solved in {} frames! Average score of {}\".format(frame_num, mean_reward))\n",
    "                break\n",
    "\n",
    "            if save_reward:\n",
    "                with open(file_name, 'a') as file:\n",
    "                    file.write(f'{frame_num}:{round(mean_reward, 2)}:{round(epsilon, 2)}\\n')\n",
    "\n",
    "                save_reward = False\n",
    "\n",
    "        if len(memory.buffer) < LEARNING_STARTS:\n",
    "            continue\n",
    "\n",
    "        if frame_num % TARGET_UPDATE_FREQ == 0 and model == 'DDQN':\n",
    "            target_net.load_state_dict(net.state_dict())\n",
    "\n",
    "        # Update depending on model\n",
    "        update(batch_size, memory, net, target_net, gamma, model, device)\n",
    "\n",
    "    env.close()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Learning Parameters & Other Parameters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Epsilon greedy parameters\n",
    "EPSILON_DECAY = 10**5\n",
    "EPSILON_START = 1.0\n",
    "EPSILON_FINAL = 0.02\n",
    "epsilon_data = [EPSILON_FINAL, EPSILON_START, EPSILON_DECAY]\n",
    "\n",
    "\n",
    "# Hyperparameters\n",
    "learning_rate = 1e-4\n",
    "REPLAY_SIZE = 100000\n",
    "BATCH_SIZE = 32\n",
    "TARGET_UPDATE_FREQ = 1000\n",
    "DELAY_LEARNING = 50000\n",
    "GAMMA = 0.99"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Initiating training, random play or playing pretrained model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']\n",
      "The GPU is being used\n"
     ]
    },
    {
     "ename": "NameError",
     "evalue": "name 'base' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-48-e9934ac4ac3b>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     39\u001b[0m     \u001b[0mfile_path\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m'./pull/Best_performing_model/DDQN_10_6-reward_7.97.dat'\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     40\u001b[0m     \u001b[0mseconds\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;36m30\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 41\u001b[0;31m     \u001b[0mtest_old_network\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0menv\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnet\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mfile_path\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mseconds\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdevice\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;32m~/Google Drev/DTU/7. Semester/DQN-Ping-Pong/lib/Test_old_network.py\u001b[0m in \u001b[0;36mtest_old_network\u001b[0;34m(env, net, file_path, seconds, device)\u001b[0m\n\u001b[1;32m     16\u001b[0m     \u001b[0;32mwhile\u001b[0m \u001b[0;32mTrue\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     17\u001b[0m         \u001b[0;31m# render a frame\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 18\u001b[0;31m         \u001b[0menv\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mrender\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     19\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     20\u001b[0m         \u001b[0;31m# choose random action\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/.local/lib/python3.6/site-packages/gym/core.py\u001b[0m in \u001b[0;36mrender\u001b[0;34m(self, mode, **kwargs)\u001b[0m\n\u001b[1;32m    231\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    232\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mrender\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmode\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'human'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 233\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0menv\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mrender\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmode\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    234\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    235\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mclose\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/.local/lib/python3.6/site-packages/gym/core.py\u001b[0m in \u001b[0;36mrender\u001b[0;34m(self, mode, **kwargs)\u001b[0m\n\u001b[1;32m    231\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    232\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mrender\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmode\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'human'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 233\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0menv\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mrender\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmode\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    234\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    235\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mclose\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/.local/lib/python3.6/site-packages/gym/core.py\u001b[0m in \u001b[0;36mrender\u001b[0;34m(self, mode, **kwargs)\u001b[0m\n\u001b[1;32m    231\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    232\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mrender\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmode\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'human'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 233\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0menv\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mrender\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmode\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    234\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    235\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mclose\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/.local/lib/python3.6/site-packages/gym/core.py\u001b[0m in \u001b[0;36mrender\u001b[0;34m(self, mode, **kwargs)\u001b[0m\n\u001b[1;32m    231\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    232\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mrender\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmode\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'human'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 233\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0menv\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mrender\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmode\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    234\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    235\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mclose\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/.local/lib/python3.6/site-packages/gym/core.py\u001b[0m in \u001b[0;36mrender\u001b[0;34m(self, mode, **kwargs)\u001b[0m\n\u001b[1;32m    231\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    232\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mrender\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmode\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'human'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 233\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0menv\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mrender\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmode\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    234\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    235\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mclose\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/.local/lib/python3.6/site-packages/gym/core.py\u001b[0m in \u001b[0;36mrender\u001b[0;34m(self, mode, **kwargs)\u001b[0m\n\u001b[1;32m    231\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    232\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mrender\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmode\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'human'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 233\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0menv\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mrender\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmode\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    234\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    235\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mclose\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/.local/lib/python3.6/site-packages/gym/core.py\u001b[0m in \u001b[0;36mrender\u001b[0;34m(self, mode, **kwargs)\u001b[0m\n\u001b[1;32m    231\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    232\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mrender\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmode\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'human'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 233\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0menv\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mrender\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmode\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    234\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    235\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mclose\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/.local/lib/python3.6/site-packages/gym/envs/atari/atari_env.py\u001b[0m in \u001b[0;36mrender\u001b[0;34m(self, mode)\u001b[0m\n\u001b[1;32m    150\u001b[0m             \u001b[0;32mreturn\u001b[0m \u001b[0mimg\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    151\u001b[0m         \u001b[0;32melif\u001b[0m \u001b[0mmode\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;34m'human'\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 152\u001b[0;31m             \u001b[0;32mfrom\u001b[0m \u001b[0mgym\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0menvs\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mclassic_control\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mrendering\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    153\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mviewer\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    154\u001b[0m                 \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mviewer\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mrendering\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mSimpleImageViewer\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/.local/lib/python3.6/site-packages/gym/envs/classic_control/rendering.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     25\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     26\u001b[0m \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 27\u001b[0;31m     \u001b[0;32mfrom\u001b[0m \u001b[0mpyglet\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mgl\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     28\u001b[0m \u001b[0;32mexcept\u001b[0m \u001b[0mImportError\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     29\u001b[0m     raise ImportError('''\n",
      "\u001b[0;32m~/.local/lib/python3.6/site-packages/pyglet/gl/__init__.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m    225\u001b[0m     \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    226\u001b[0m         \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0mcarbon\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mCarbonConfig\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mConfig\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 227\u001b[0;31m \u001b[0;32mdel\u001b[0m \u001b[0mbase\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    228\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    229\u001b[0m \u001b[0;31m# XXX remove\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mNameError\u001b[0m: name 'base' is not defined"
     ]
    }
   ],
   "source": [
    "UP_ACTION = 2\n",
    "DOWN_ACTION = 3\n",
    "\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "\n",
    "# This options dict was created from sys.argv[x] normally, but in ipynb this is redundant, hence a dictionary is made here.\n",
    "option_dict = {'random': False, 'train': False, 'oldnetwork': True} # Change dictionary values for decision making.\n",
    "model = 'DDQN' # Either DQN, DDQN or CDDQN\n",
    "\n",
    "# Environment and neural networks\n",
    "env = make_env('PongNoFrameskip-v4')\n",
    "net = DQN(env.observation_space.shape, env.action_space.n, learning_rate).to(device)\n",
    "target_net = DQN(env.observation_space.shape, env.action_space.n, learning_rate).to(device)\n",
    "\n",
    "# Agent and memory handling\n",
    "memory = Memory(REPLAY_SIZE)\n",
    "agent = Agent(env, memory)\n",
    "\n",
    "initial_observation = env.reset()\n",
    "\n",
    "\n",
    "\n",
    "if 'cuda' in str(device):\n",
    "    print('The GPU is being used')\n",
    "else:\n",
    "    print('The CPU is being used')\n",
    "\n",
    "if option_dict['random']:\n",
    "    play_random(env, UP_ACTION, DOWN_ACTION, seconds=5)\n",
    "\n",
    "if option_dict['train']:\n",
    "    print(\"Training\")\n",
    "    print(\"ReplayMemory will require {}gb of GPU RAM\".format(round(REPLAY_SIZE * 32 * 84 * 84 / 1e+9, 2)))\n",
    "    agent.reset_environtment()\n",
    "    train(env, net, target_net, epsilon_data, agent, memory, GAMMA, device,\n",
    "                DELAY_LEARNING, TARGET_UPDATE_FREQ, BATCH_SIZE, model)\n",
    "\n",
    "if option_dict['oldnetwork']:\n",
    "    file_path = './pull/Best_performing_model/DDQN_10_6-reward_7.97.dat'\n",
    "    seconds = 30\n",
    "    test_old_network(env, net, file_path, seconds, device)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
