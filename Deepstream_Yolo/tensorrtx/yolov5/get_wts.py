import torch
import struct
from utils.torch_utils import select_device
import sys
# Initialize
device = select_device('cpu')
# Load model
model = torch.load(sys.argv[1], map_location=device)['model'].float()  # load to FP32
model.to(device).eval()

f = open(sys.argv[2], 'w')
f.write('{}\n'.format(len(model.state_dict().keys())))
for k, v in model.state_dict().items():
    vr = v.reshape(-1).cpu().numpy()
    f.write('{} {} '.format(k, len(vr)))
    for vv in vr:
        f.write(' ')
        f.write(struct.pack('>f',float(vv)).hex())
    f.write('\n')
