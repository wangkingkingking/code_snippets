import torch.nn as nn
import time
import torch


class BasicModule(nn.Module):
    """
    Make nn.Module easier to use
    """
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def save(self, epoch=None, iter=None, name=None):
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            if epoch:
                prefix += 'epoch_{:03d}_'.format(int(epoch))
            if iter:
                prefix += 'iter_{:06d}_'.format(int(iter))
            name = time.strftime(prefix + '%m%d_%H:%M.pth')
        print('Saving state_dict to {:s}...'.format(name))
        torch.save(self.state_dict(), name)
        
        return name

    def load(self, path):
        print('Loading state_dict from {:s}...'.format(path))
        self.load_state_dict(torch.load(path))

    
