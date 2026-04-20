# train_fighting.py
from ultralytics import YOLO
import os

if __name__ == '__main__':
    model = YOLO('yolov8m-cls.pt')

    # Get the directory where this script lives
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model.train(
        data     = os.path.join(BASE_DIR, 'data/fighting'),
        epochs   = 50,
        imgsz    = 224,
        batch    = 16,
        name     = 'fight_classifier',
        project  = os.path.join(BASE_DIR, 'runs/classify'),  # absolute path
        device   = 0,
        patience = 10,
        workers  = 2,
        plots    = True,
    )