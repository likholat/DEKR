import numpy as np
import torch
from openvino.inference_engine import IECore

inp = torch.load('image.pt').numpy()
ref = torch.load('ref_res.pt')

ie = IECore()
net = ie.read_network('model.xml', 'model.bin')
exec_net = ie.load_network(net, 'CPU')

ov_out = exec_net.infer({'input': inp})

ref_h = ref[0].detach().numpy()
ref_o = ref[1].detach().numpy()

heatmap_loss = np.max(np.abs(ref_h - ov_out['heatmap']))
offset_loss = np.max(np.abs(ref_o - ov_out['offset']))

print(f"Heatmap range: [{str(np.min(ref_h))}, {str(np.max(ref_h))}]")
print(f"Offset range: [{str(np.min(ref_o))}, {str(np.max(ref_o))}]\n")

print("Heatmap loss: " + str(heatmap_loss))
print("Offset loss: " + str(offset_loss))
