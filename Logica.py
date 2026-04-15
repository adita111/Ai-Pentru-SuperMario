import numpy as np
from gym import Wrapper
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

import torch
from torch import nn
from  tensordict import TensorDict
from torch.ao.quantization.backend_config.onednn import observation_type
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from torchrl.data import LazyTensorStorage






#### AgentNeuronal Network

class AgentNeuronalNetwork(nn.Module):
    def __init__(self,input_shape, n_actions , freeze=False):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0],32,kernel_size=8,stride=4),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=4,stride=2),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.network = nn.Sequential(
            self.conv,
            nn.Flatten(),
            nn.Linear(conv_out_size,256),
            nn.ReLU(),
            nn.Linear(256,n_actions)
        )

        if freeze:
            self._freeze()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(torch.cuda.is_available(), torch.__version__)

        self.to(self.device)
    def _get_conv_out(self,shape):
        o = self.conv(torch.zeros(1,*shape))
        return int(np.prod(o.size()))

    def _freeze(self):
        for param in self.network.parameters():
            param.requires_grad = False

    def forward(self,x):
        return self.network(x)

#### End of AgentNeuronal Network


####Wrapper to skip frames
class SkipFrame(Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        for _ in range(self._skip):
            next_state, reward, done,trunc,info = self.env.step(action)
            total_reward += reward
            if done or trunc:
                break
        return next_state, total_reward, done,trunc , info

def apply_wrappers(env):
    env = SkipFrame(env, skip=6)
    env = ResizeObservation(env,shape=(84,84))
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=4, lz4_compress=True)
    return env
### End of Wrapper

#### Agent

class Agent:
    def __init__(self,input_dims, num_actions):
        self.num_actions = num_actions
        self.learn_step_counter = 0

        self.lr=0.00025
        self.gamma=0.99
        self.epsilon=1.0
        self.eps_min=0.1
        self.eps_dec=0.99999975
        self.batch_size=16
        self.sync_network_rate=10000

        self.online_network = AgentNeuronalNetwork(input_dims, num_actions)
        self.target_network = AgentNeuronalNetwork(input_dims, num_actions)

        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.lr)
        self.loss = torch.nn.MSELoss()

        replay_buffer_capacity=10000
        storage = LazyTensorStorage( replay_buffer_capacity)
        self.replay_buffer= TensorDictReplayBuffer(storage=storage)

    def choose_action(self,observation):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        observation = torch.as_tensor(observation, dtype=torch.float32)\
            .unsqueeze(0)\
            .to(self.online_network.device)
        return self.online_network(observation).argmax().item()

    def decay_epsilon(self):
        self.epsilon=max(self.epsilon*self.eps_dec,self.eps_min)

    def store_in_memory(self,state,action,reward,next_state,done):
        self.replay_buffer.add(
            TensorDict(
                {
                    "state": torch.as_tensor(state,dtype=torch.float32),
                    "action": torch.tensor(action),
                    "reward": torch.tensor(reward),
                    "next_state": torch.as_tensor(next_state,dtype=torch.float32),
                    "done": torch.tensor(done)
                },
                batch_size=[]
            )
        )

    def sync_networks(self):
        if self.learn_step_counter % self.sync_network_rate == 0 and self.learn_step_counter >0:
            self.target_network.load_state_dict(self.online_network.state_dict())


    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        self.sync_networks()
        self.optimizer.zero_grad()

        samples = self.replay_buffer.sample(self.batch_size).to(self.online_network.device)

        keys=("state","action","reward","next_state","done")

        states,actions,rewards,next_states,dones=[samples[key] for key in keys]

        predicted_q_values = self.online_network(states)
        predicted_q_values = predicted_q_values[np.arange(self.batch_size),actions.squeeze()]

        target_q_values = self.target_network(next_states).max(dim=1)[0]
        target_q_values = rewards + self.gamma * target_q_values * (1-dones.float())

        loss = self.loss(predicted_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter +=1
        self.decay_epsilon()
#### End of Agent





#### Main LoopF
env = gym_super_mario_bros.make(
    "SuperMarioBros-v1",
    apply_api_compatibility=True,
    render_mode="human"
)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
NUM_OF_EPISODES=50000
LEARN_EVERY = 4
global_step = 0
env=apply_wrappers(env)

agent=Agent(
    input_dims=env.observation_space.shape,
    num_actions=env.action_space.n
)

for i in range(NUM_OF_EPISODES):
    done=False
    state,_=env.reset()
    while not done:
        action = agent.choose_action(state)

        new_state,reward,done,truncated,info=env.step(action)
        agent.store_in_memory(state,action,reward,new_state,done)

        global_step += 1
        if global_step % LEARN_EVERY == 0:
            agent.learn()

        state = new_state

env.close()