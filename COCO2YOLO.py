from ultralytics.data.converter import convert_coco
# for coco    (det)
convert_coco(labels_dir='datasets/coco/annotations/')
# for coco-seg
convert_coco(labels_dir='datasets/coco/annotations/', use_segments=True)
