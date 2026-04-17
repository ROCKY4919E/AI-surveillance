from pathlib import Path
from ultralytics import YOLO
import torch

if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent
    project_dir = base_dir / "runs" / "visdrone"
    project_dir.mkdir(parents=True, exist_ok=True)

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Project dir: {project_dir}")

    model_path = base_dir / "yolov8m.pt"
    data_path = base_dir / "data" / "visdrone" / "data.yaml"

    model = YOLO(str(model_path))

    results = model.train(
        data=str(data_path),
        epochs=50,
        imgsz=640,
        batch=4,        # reduced from 8
        device=0,
        workers=2,      # reduced from 4
        patience=10,
        save=True,
        save_period=5,  # saves last.pt every 5 epochs so crashes don't lose everything
        project=str(project_dir),
        name="yolov8m_visdrone",
        exist_ok=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        cache=False,
    )

    print("Training complete.")
    print(f"Best model saved at: {results.save_dir}")