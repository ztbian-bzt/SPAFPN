import argparse
import warnings
import os
warnings.filterwarnings('ignore')
from ultralytics import YOLO


def get_args(add_help=True):
    parser = argparse.ArgumentParser(description="SPAFPN Prediction", add_help=add_help)
    parser.add_argument('--source', help='path of image')
    parser.add_argument('--weight', default='weights/SPAFPNn-C2f.pt', type=str, help='pretrained weight')
    parser.add_argument('--iou', default=0.7, type=float, help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--conf', default=0.3, type=float, help='object confidence threshold for detection')
    parser.add_argument('--augment', default=False, type=bool, help='apply image augmentation to val sources')
    parser.add_argument('--softnms', default=False, type=bool, help='use soft-nms for val')
    parser.add_argument('--save', default=True, type=bool, help='save predict results')
    parser.add_argument('--show_labels', default=True, type=bool, help='show object labels in plots')
    parser.add_argument('--show_conf', default=True, type=bool, help='show object confidence scores in plots')
    # more args can be modified at ultralytics/cfg/default.yaml
    return parser


def main(args):
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if visible is None:
        visible = 0
    model = YOLO(args.weight)
    model.predict(source=args.source, device=visible, iou=args.iou, conf=args.conf, augment=args.augment,
                  softnms=args.softnms, save=args.save, show_labels=args.show_labels, show_conf=args.show_conf)


if __name__ == '__main__':
    args = get_args().parse_args()
    main(args)
