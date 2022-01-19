import torch.utils.data.distributed
import torch.utils.data
import torch.optim
import torch.nn.parallel
import torch
import io
import argparse
import sys
sys.path.append("../lib")
import _init_paths
import models
import numpy as np

from config import update_config
from config import cfg
from openvino.inference_engine import IECore


CTX = torch.device('cpu')


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(
            cfg.TEST.MODEL_FILE, map_location=CTX), strict=False)
    else:
        raise ValueError('expected model defined in config at TEST.MODEL_FILE')

    pose_model.to(CTX)
    pose_model.eval()

    with torch.no_grad():
        inp = torch.rand(1, 3, 512, 960)
        heatmap, offset = pose_model(inp)

        torch.onnx.export(pose_model, inp, 'model.onnx',
                          input_names=['input'],
                          output_names=['heatmap', 'offset'],
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
                          opset_version=12
                          )

        with open('model.onnx', 'rb') as f:
            buf = io.BytesIO(f.read())

        ie = IECore()
        net = ie.read_network(buf.getvalue(), b"", init_from_buffer=True)
        exec_net = ie.load_network(net, "CPU")

        ov_out = exec_net.infer({'input': inp})

        ref_h = heatmap.detach().numpy()
        ref_o = offset.detach().numpy()

        heatmap_loss = np.max(np.abs(ref_h - ov_out['heatmap']))
        offset_loss = np.max(np.abs(ref_o - ov_out['offset']))

        print(f"Heatmap range: [{str(np.min(ref_h))}, {str(np.max(ref_h))}]")
        print(f"Offset range: [{str(np.min(ref_o))}, {str(np.max(ref_o))}]\n")

        print("Heatmap loss: " + str(heatmap_loss))
        print("Offset loss: " + str(offset_loss))
        

if __name__ == '__main__':
    main()
