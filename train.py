from pathlib import Path
import argparse
import yaml
from ultralytics import YOLO
import torch


def load_config(config_path: Path):
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    base_dir = Path(__file__).resolve().parent
    config_path = base_dir / "datasets.yaml"

    parser = argparse.ArgumentParser(description="Train a YOLOv8 model for a selected dataset.")
    parser.add_argument("--dataset", required=True, choices=["visdrone", "dfire"], help="Dataset key from datasets.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Override image size")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", default="0", help="CUDA device or cpu")
    parser.add_argument("--workers", type=int, default=None, help="Override number of workers")
    parser.add_argument("--name", default=None, help="Override the training run name")
    args = parser.parse_args()

    cfg = load_config(config_path)
    dataset_cfg = cfg["datasets"][args.dataset]

    project_dir = base_dir / dataset_cfg["project"]
    project_dir.mkdir(parents=True, exist_ok=True)

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset: {args.dataset}")
    print(f"Project dir: {project_dir}")

    model_path = base_dir / "yolov8m.pt"
    data_path = base_dir / dataset_cfg["data"]

    model = YOLO(str(model_path))

    train_kwargs = {
        "data": str(data_path),
        "epochs": args.epochs if args.epochs is not None else dataset_cfg.get("epochs", 50),
        "imgsz": args.imgsz,
        "batch": args.batch if args.batch is not None else dataset_cfg.get("batch", 8),
        "device": args.device,
        "workers": args.workers if args.workers is not None else dataset_cfg.get("workers", 4),
        "patience": dataset_cfg.get("patience", 10),
        "save": True,
        "save_period": dataset_cfg.get("save_period", 5),
        "project": str(project_dir),
        "name": args.name if args.name is not None else dataset_cfg.get("name"),
        "exist_ok": True,
        "optimizer": dataset_cfg.get("optimizer", "AdamW"),
        "lr0": dataset_cfg.get("lr0", 0.001),
        "lrf": dataset_cfg.get("lrf", 0.01),
        "warmup_epochs": dataset_cfg.get("warmup_epochs", 3),
        "cache": dataset_cfg.get("cache", False),
    }

    for extra in ["mosaic", "close_mosaic"]:
        if extra in dataset_cfg:
            train_kwargs[extra] = dataset_cfg[extra]

    results = model.train(**train_kwargs)

    print("Training complete.")
    print(f"Best model saved at: {results.save_dir}")


if __name__ == "__main__":
    main()
