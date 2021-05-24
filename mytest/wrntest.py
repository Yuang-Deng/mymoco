from collections import OrderedDict
import torch

from moco.wrn import Network

model_config = OrderedDict([
    ('arch', 'wrn'),
    ('depth', 28),
    ('base_channels', 16),
    ('widening_factor', 2),
    ('drop_rate', 0.3),
    ('input_shape', (1, 3, 32, 32)),
    ('n_classes', 10),
])
wrnnet = Network(model_config)
out = wrnnet.forward_conv(
                torch.zeros((32,3,32,32)))
out = wrnnet.fc(out.view(out.size(0), -1))
print(out)
