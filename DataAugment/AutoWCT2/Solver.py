import argparse
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image
from DQN import DQN
from Data import DataLoader
from VGG_ENV import VGGRapper, VGG, make_layers
from tensorboardX import SummaryWriter


env = VGGRapper()
num_action = env.n_actions


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
                                               pin_memory=False,
                                               drop_last=True)

    for n_ep in range(args.n_episodes):  
        for batch_i, images in enumerate(train_loader):
            images = images.to(args.device)
            ep_reward = 0.0

            state = images[:, :, :, 0: args.stride]
            for idx in range(args.stride, args.img_size, args.stride):
                next_state = images[:, :, :, idx: idx + args.stride]

                action, value = agent.choose_action(state)
                reward = env.step(state, value)
                ep_reward += reward

                agent.learn(state, action, reward, next_state)
                state = next_state
            
            ep_reward = ep_reward.mean()
            print("n_ep:{}, batch_i:{}, update_count:{}, ep_reward:{}".format(n_ep, batch_i, agent.update_count, ep_reward))

            if batch_i % 2 == 0:
                n_step = n_ep*len(train_set) + batch_i*args.batch_size
                agent.writer.add_scalar('live/ep_reward', ep_reward, global_step=n_step)

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
    parser.add_argument("--n_cpu", type=int, default=4, help="dataloader threads number")
    parser.add_argument('--logs', type=str, default='logs/20201202')
    parser.add_argument('--train_path', type=str, default='data/train.txt')
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    print(args)

    train(args)