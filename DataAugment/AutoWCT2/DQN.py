import torch

from VGG_ENV import VGG, VGGRapper


env = VGGRapper()

num_channel = 3
num_action = env.n_actions


class DQN():
    def __init__(self, args):
        super(DQN, self).__init__()
        self.args = args
    
    def choose_action(self, state):
        pass

    def learn(self):
        pass


def train(args):
    env.reset()
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