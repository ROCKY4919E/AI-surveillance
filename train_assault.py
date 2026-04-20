# train_assault.py
from ultralytics import YOLO
import os

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model = YOLO('yolov8m-cls.pt')

    model.train(
        data     = os.path.join(BASE_DIR, 'data/assault'),
        epochs   = 50,
        imgsz    = 224,
        batch    = 16,
        name     = 'assault_classifier',
        project  = os.path.join(BASE_DIR, 'runs/classify'),
        device   = 0,
        patience = 10,
        workers  = 2,
        plots    = True,
    )