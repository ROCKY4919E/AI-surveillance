from pathlib import Path
from ultralytics import YOLO
import torch

if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent
    project_dir = base_dir / "runs" / "dfire"
    project_dir.mkdir(parents=True, exist_ok=True)

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Project dir: {project_dir}")

    model_path = base_dir / "yolov8m.pt"
    data_path = base_dir / "data" / "dfire" / "data.yaml"

    model = YOLO(str(model_path))

    results = model.train(
        data=str(data_path),
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
        patience=10,
        save=True,
        project=str(project_dir),
        name="yolov8m_dfire",
        exist_ok=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.1,
        warmup_epochs=3,
        mosaic=1.0,
        close_mosaic=10,
        augment=True,
        flipud=0.5,
        fliplr=0.5,
    )

    print("Training complete.")
    print(f"Best model saved at: {results.save_dir}")