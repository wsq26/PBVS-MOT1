#!/usr/bin/env python
"""
Train YOLOv11 detector for thermal object detection.

This script trains a YOLOv11 model on thermal imagery using the provided dataset.
"""
import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv11 detector for thermal MOT')
    parser.add_argument('--data', type=str, required=True, help='path to dataset YAML file')
    parser.add_argument('--weights', type=str, default='yolov11s.pt', help='initial weights path')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--device', type=str, default='0', help='device id (0, 1, or cpu)')
    parser.add_argument('--project', type=str, default='./runs/detect', help='project directory')
    parser.add_argument('--name', type=str, default='thermal_detector', help='experiment name')
    parser.add_argument('--lr0', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01, help='final learning rate')
    parser.add_argument('--patience', type=int, default=50, help='early stopping patience')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load model
    model = YOLO(args.weights)

    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        lr0=args.lr0,
        lrf=args.lrf,
        patience=args.patience,
        save=True,
        verbose=True
    )

    print(f"Training completed. Best model saved to {results.save_dir}")

if __name__ == '__main__':
    main()
