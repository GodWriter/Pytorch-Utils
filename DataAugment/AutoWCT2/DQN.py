import argparse
import numpy as np

import torch
import torch.nn as nn

from tensorboardX import SummaryWriter
from VGG_ENV import VGG, VGGRapper, make_layers


env = VGGRapper()
num_action = env.n_actions


class DQN():
    def __init__(self, args):
        super(DQN, self).__init__()
        self.args = args

        self.target_net = VGG(make_layers(cfg[args.vgg_type], batch_norm=True), 
                              num_class=num_action).to(args.device).eval()
        self.eval_net = VGG(make_layers(cfg[args.vgg_type], batch_norm=True),
                            num_class=num_action).to(args.device).train()
        
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), self.args.lr)

        self.writer = SummaryWriter(self.args.logs)

    def choose_action(self, state):
        value = self.eval_net(state)

        _, idx = torch.max(value, 1)
        action = idx.item()

        if np.random.rand(1) >= 0.9:
            action = np.random.choice(range(num_action), 1).item()
        
        return action

    def learn(self, state, action, reward, next_state):
        pass


def train(args):
    agent = DQN()

    for n_ep in range(args.n_episodes):
        for batch_i, images in enumerate(train_loader):
            images = images.to(args.device)

            state = images[:, :, :, 0: args.stride]
            for idx in range(args.stride, args.img_size, args.stride):
                next_state = images[:, :, :, idx: idx + args.stride]

                action = agent.choose_action(state)
                reward = env.step(state, action)

                agent.learn(state, action, reward, next_state)
                state = next_state
    
    print("Training is Done!!!")