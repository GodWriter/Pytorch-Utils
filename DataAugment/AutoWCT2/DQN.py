import numpy as np

import torch
import torch.nn as nn

from PIL import Image
from tensorboardX import SummaryWriter
from VGG_ENV import VGGRapper, VGG, make_layers


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
        
        # 用于通过动作选择相关的权值, 一定要加入np.float32，否则会报错expected device cuda:0 等问题
        self.actions = torch.from_numpy(np.asarray([env.action_space for _ in range(args.batch_size)], dtype=np.float32)).to(args.device)
        
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), args.lr)

        self.update_count = 0
        self.writer = SummaryWriter(args.logs)

    def choose_action(self, state):
        value = self.eval_net(state)

        _, idx = torch.max(value, 1)
        shape = idx.size(0) # 由于数据总量不可能总是整除batch_size，故需要每次得到当前的batch_size

        action = np.zeros(shape)
        for i in range(shape):
            action[i] = idx[i].item()

        if np.random.rand(1) >= 0.9:
            for i in range(shape):
                action[i] = np.random.choice(range(num_action), 1)
        
        action = torch.LongTensor([t for t in action]).view(-1, 1).long()
        action = action.to(self.args.device)

        value = self.actions.gather(1, action)
        return action, value

    def learn(self, state, action, reward, next_state):
        with torch.no_grad():
            target_v = reward + self.args.gamma * self.target_net(next_state).max(1)[0]
            target_v = target_v.unsqueeze(1)

        eval_v = (self.eval_net(state).gather(1, action))
        loss = self.loss_func(target_v, eval_v)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.writer.add_scalar('loss/value_loss', loss, self.update_count)
        self.update_count += 1

        if self.update_count % 100 == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        
        if self.update_count % 2000 == 0:
            torch.save(self.eval_net.state_dict(), "checkpoints/autowct_{}.pth".format(self.update_count))