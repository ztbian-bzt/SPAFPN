import argparse
import warnings
import os
warnings.filterwarnings('ignore')
from ultralytics import YOLO


def get_args(add_help=True):
    parser = argparse.ArgumentParser(description="SPAFPN Validation", add_help=add_help)
    parser.add_argument('--data', default='coco.yaml', type=str, help='path of dataset')
    parser.add_argument('--imgsz', default=640, type=int, help='train, val image size (pixels)')
    parser.add_argument('--batch', default=32, type=int, help='batch size')
    parser.add_argument('--weight', default='weights/SPAFPNn-C2f.pt', type=str, help='pretrained weight')
    parser.add_argument('--save_json', default=False, type=bool, help='save results to JSON file')
    parser.add_argument('--iou', default=0.7, type=float, help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--conf', default=0.001, type=float, help='object confidence threshold for detection')
    parser.add_argument('--half', default=False, type=bool, help='use half precision (FP16)')
    parser.add_argument('--augment', default=False, type=bool, help='apply image augmentation to val sources')
    parser.add_argument('--softnms', default=False, type=bool, help='use soft-nms for val')
    parser.add_argument('--split', default='val', type=str,
                        help="dataset split to use for validation, i.e. 'val', 'test' or 'train'")
    # more args can be modified at ultralytics/cfg/default.yaml
    return parser


def main(args):
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if visible is None:
        visible = 0
    model = YOLO(args.weight)
    model.val(data=args.data, split=args.split, imgsz=args.imgsz, batch=args.batch, save_json=args.save_json,
              device=visible, iou=args.iou, conf=args.conf, half=args.half, augment=args.augment, softnms=args.softnms)


if __name__ == '__main__':
    args = get_args().parse_args()
    main(args)
