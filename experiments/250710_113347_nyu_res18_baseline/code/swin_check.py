# from model.backbone import swin
# net = swin.swin_large_naive_l4w722422k()
# print('Swin loaded OK, #params =', sum(p.numel() for p in net.parameters())/1e6,'M')

# import distutils, inspect, sys
# print('imported from:', distutils.__file__)

# from distutils.version import LooseVersion
# print('LooseVersion OK ->', LooseVersion('1.0') < LooseVersion('2.0'))

from torch.utils.tensorboard import SummaryWriter
print('TensorBoard OK, SummaryWriter =', SummaryWriter)