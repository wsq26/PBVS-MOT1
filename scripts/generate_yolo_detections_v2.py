import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from tqdm import tqdm
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='path to model weights')
    parser.add_argument('--source', type=str, required=True, help='path to images root')
    parser.add_argument('--output-dir', type=str, required=True, help='directory to save detection results')
    parser.add_argument('--conf-thres', type=float, default=0.01, help='confidence threshold')
    parser.add_argument('--imgsz', type=int, default=640, help='inference size')
    parser.add_argument('--device', type=str, default='0', help='device id (0, 1, or cpu)')
    args = parser.parse_args()

    # Load model
    print(f"Loading YOLOv11 model from {args.weights}...")
    model = YOLO(args.weights)

    # Sequences
    seqs = ["seq2", "seq17", "seq22", "seq47", "seq54", "seq66"]
    
    os.makedirs(args.output_dir, exist_ok=True)

    for seq in seqs:
        print(f"Processing sequence: {seq}")
        seq_dir = os.path.join(args.source, seq, 'thermal')
        
        det_file = os.path.join(args.output_dir, f"{seq}.txt")
        
        img_files = sorted(glob.glob(os.path.join(seq_dir, '*.png')))
        if not img_files:
            print(f"Warning: No images found for {seq} in {seq_dir}")
            continue
            
        print(f"Found {len(img_files)} images.")
        
        with open(det_file, 'w') as f:
            # Process one by one to avoid OOM
            for i, img_path in enumerate(tqdm(img_files)):
                frame_id = i + 1
                
                # Run inference on single image
                results = model.predict(source=img_path, conf=args.conf_thres, imgsz=args.imgsz, device=args.device, verbose=False)
                
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        
                        if cls != 0:
                            continue
                            
                        w = x2 - x1
                        h = y2 - y1
                        
                        line = f"{frame_id},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n"
                        f.write(line)
                    
        print(f"Saved detections to {det_file}")
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
