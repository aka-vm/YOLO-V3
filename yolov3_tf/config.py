import os
import pathlib

# env var dataset_dir
BASE_DATASET_DIR = "/Users/username/path_to_datasets"
BASE_DATASET_DIR = os.environ.get('dataset_dir')
BASE_DATASET_DIR_PATH = pathlib.Path(BASE_DATASET_DIR)

NUM_CLASSES_DICT = {
    "COCO": 80,
    "PASCAL_VOC": 20,
}

# DATASET = 'COCO'
DATASET = 'PASCAL_VOC'

BATCH_SIZE = 32
IMAGE_SIZE = 416
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
CONF_THRESHOLD = 0.05
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45

NUM_CLASSES = NUM_CLASSES_DICT[DATASET]
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]

LOAD_MODEL = True
SAVE_MODEL = True

ANCHORS = [
    [(0.279,0.216), (0.375,0.476), (0.897,0.784)],      # 116,90,  156,198,  373,326
    [(0.072,0.147), (0.149,0.108), (0.142,0.286)],      # 30,61,  62,45,  59,119
    [(0.024,0.031), (0.038,0.072), (0.079,0.055)],      # 10,13,  16,30,  33,23
]

CLASS_LABELS_DICT = {
    "COCO": [
        'person',
        'bicycle',
        'car',
        'motorcycle',
        'airplane',
        'bus',
        'train',
        'truck',
        'boat',
        'traffic light',
        'fire hydrant',
        'stop sign',
        'parking meter',
        'bench',
        'bird',
        'cat',
        'dog',
        'horse',
        'sheep',
        'cow',
        'elephant',
        'bear',
        'zebra',
        'giraffe',
        'backpack',
        'umbrella',
        'handbag',
        'tie',
        'suitcase',
        'frisbee',
        'skis',
        'snowboard',
        'sports ball',
        'kite',
        'baseball bat',
        'baseball glove',
        'skateboard',
        'surfboard',
        'tennis racket',
        'bottle',
        'wine glass',
        'cup',
        'fork',
        'knife',
        'spoon',
        'bowl',
        'banana',
        'apple',
        'sandwich',
        'orange',
        'broccoli',
        'carrot',
        'hot dog',
        'pizza',
        'donut',
        'cake',
        'chair',
        'couch',
        'potted plant',
        'bed',
        'dining table',
        'toilet',
        'tv',
        'laptop',
        'mouse',
        'remote',
        'keyboard',
        'cell phone',
        'microwave',
        'oven',
        'toaster',
        'sink',
        'refrigerator',
        'book',
        'clock',
        'vase',
        'scissors',
        'teddy bear',
        'hair drier',
        'toothbrush'
    ],
    "PASCAL_VOC": [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor"
    ]
}

CLASS_LABELS = CLASS_LABELS_DICT[DATASET]