import argparse
import warnings
import os
warnings.filterwarnings('ignore')
from ultralytics import YOLO


def get_args(add_help=True):
    parser = argparse.ArgumentParser(description="SPAFPN Training", add_help=add_help)
    parser.add_argument('--data', default='coco.yaml', type=str, help='path of dataset')
    parser.add_argument('--yaml', default='SPAFPNn-C2f.yaml', type=str, help='experiments yaml')
    parser.add_argument('--epochs', default=500, type=int, help='number of total epochs to run')
    parser.add_argument('--imgsz', default=640, type=int, help='train, val image size (pixels)')
    parser.add_argument('--batch', default=128, type=int, help='batch size')
    parser.add_argument('--device', default=[0, 1],
                        help='device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu')
    parser.add_argument('--weight', default=None, type=str, help='pretrained weight')
    parser.add_argument('--resume', default=None, type=str, help='resume train')
    parser.add_argument('--amp', default=True, type=bool, help='Automatic Mixed Precision (AMP) training')
    parser.add_argument('--freeze', default=None,
                        help='freeze first n layers, or freeze list of layer indices during training')
    # parser.add_argument('--local-rank', default=0, type=int,
    #                     help='for higher version of torch to use python -m torch.distributed.launch')
    # more args can be modified at ultralytics/cfg/default.yaml
    return parser


def main(args):
    if args.weight is None:
        model = YOLO(args.yaml)
    else:
        model = YOLO(args.weight)
    if args.resume is None:
        model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                    device=args.device, amp=args.amp, freeze=args.freeze)
    else:
        model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                    device=args.device, amp=args.amp, freeze=args.freeze, resume=args.resume)


if __name__ == '__main__':
    args = get_args().parse_args()
    main(args)
