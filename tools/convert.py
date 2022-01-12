import torch.utils.data.distributed
import torch.utils.data
import torch.optim
import torch.nn.parallel
import torch
import argparse
import sys
sys.path.append("../lib")
import _init_paths
import models

from config import update_config
from config import cfg


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

    print(cfg.MODEL.NAME)

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
        res = pose_model(inp)

        # to compare results
        torch.save(inp, 'image.pt')
        torch.save(res, 'ref_res.pt')

        torch.onnx.export(pose_model, inp, 'model.onnx',
                          input_names=['input'],
                          output_names=['heatmap', 'offset'],
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
                          opset_version=12
                          )


if __name__ == '__main__':
    main()
