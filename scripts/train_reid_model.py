#!/usr/bin/env python
"""
Fine-tune ReID model for thermal person re-identification.

This script fine-tunes a pre-trained ReID model on thermal imagery.
"""
import argparse
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune ReID model for thermal MOT')
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='path to pre-trained checkpoint')
    parser.add_argument('--work-dir', type=str, default='./work_dirs/reid', help='work directory')
    parser.add_argument('--gpus', type=int, default=1, help='number of GPUs')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    return parser.parse_args()

def main():
    args = parse_args()

    try:
        from mmtrack.apis import train_model
        from mmcv import Config
    except ImportError:
        print("Error: mmtrack not installed. Please install mmtrack first:")
        print("  pip install mmtrack")
        sys.exit(1)

    # Load config
    cfg = Config.fromfile(args.config)

    # Set checkpoint if provided
    if args.checkpoint:
        cfg.load_from = args.checkpoint

    # Set work directory
    cfg.work_dir = args.work_dir
    os.makedirs(args.work_dir, exist_ok=True)

    # Train
    print(f"Training ReID model with config: {args.config}")
    print(f"Work directory: {args.work_dir}")

    try:
        train_model(cfg, distributed=False, validate=True)
        print(f"Training completed. Model saved to {args.work_dir}")
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
