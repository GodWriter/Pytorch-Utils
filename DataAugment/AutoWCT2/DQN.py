import argparse
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image
from Data import DataLoader
from VGG_ENV import VGGRapper, VGG, make_layers

from tensorboardX import SummaryWriter


env = VGGRapper()
num_action = env.n_actions


cfg = {11 : [16,     'M', 32,       'M', 64,  64,            'M', 128, 128,           'M', 128, 128,           'M'],
       13 : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
       16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
       19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}


class DQN():
    def __init__(self, args):
        super(DQN, self).__init__()
        self.args = args

        self.target_net = VGG(make_layers(cfg[args.vgg_type], batch_norm=True), 
                              num_class=num_action).to(args.device).eval()
        self.eval_net = VGG(make_layers(cfg[args.vgg_type], batch_norm=True),
                            num_class=num_action).to(args.device).train()
        
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), args.lr)

        self.writer = SummaryWriter(args.logs)

    def choose_action(self, state):
        value = self.eval_net(state)

        _, idx = torch.max(value, 1)
        shape = idx.size(0) # 由于数据总量不可能总是整除batch_size，故需要每次得到当前的batch_size

        action = np.zeros(shape)
        for i in range(shape):
            action[i] = idx[i].item()

        if np.random.rand(1) >= 0.1:
            for i in range(shape):
                action[i] = np.random.choice(range(num_action), 1)
        
        action = torch.LongTensor([t for t in action]).view(-1, 1).long()
        action = action.to(args.device)

        return action

    def learn(self, state, action, reward, next_state):
        pass


def train(args):
    agent = DQN(args)
    writer = SummaryWriter(args.logs)

    transform = transforms.Compose([transforms.Resize((args.img_size, args.img_size), Image.BICUBIC),
                                    transforms.ToTensor()])

    train_set = DataLoader(args.train_path, args.stride, transform)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.n_cpu,
                                               pin_memory=False)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_episodes', type=int, default=100)
    parser.add_argument("--img_size", type=int, default=320, help="size of each image dimension")
    parser.add_argument("--vgg_type", type=int, default=13, help="you can choose from 11, 13, 16, 19")
    parser.add_argument("--stride", type=int, default=32, help="size of stride")
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument("--n_cpu", type=int, default=2, help="dataloader threads number")
    parser.add_argument('--logs', type=str, default='logs/20201130')
    parser.add_argument('--train_path', type=str, default='data/train.txt')
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    print(args)

    train(args)