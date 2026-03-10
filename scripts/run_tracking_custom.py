
import os
import argparse
import numpy as np
import mmcv
import torch
from mmtrack.apis import init_model
from mmcv import Config
from mmtrack.models import build_model
import glob
import json
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Run Tracking with Pre-computed Detections + ReID')
    parser.add_argument('--config', required=True, help='config file path')
    parser.add_argument('--reid-checkpoint', required=True, help='checkpoint file for reid')
    parser.add_argument('--detections-dir', required=True, help='Directory containing detection txt files (seq.txt)')
    parser.add_argument('--img-root', required=True, help='Image root directory')
    parser.add_argument('--output-dir', required=True, help='output directory')
    parser.add_argument('--device', default='cuda:0', help='device used for inference')
    parser.add_argument('--seqs', nargs='+', default=["seq2", "seq17", "seq22", "seq47", "seq54", "seq66"], help='sequences to process')
    
    args = parser.parse_args()
    return args

def load_detections(det_file):
    dets = {}
    if not os.path.exists(det_file):
        print(f"Warning: Detection file {det_file} not found.")
        return dets
    
    print(f"Loading detections from {det_file}...")
    count = 0
    with open(det_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame_id = int(parts[0])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6])
            
            x2 = x + w
            y2 = y + h
            
            if frame_id not in dets:
                dets[frame_id] = []
            dets[frame_id].append([x, y, x2, y2, conf])
            count += 1
            
    for fid in dets:
        dets[fid] = np.array(dets[fid], dtype=np.float32)
        
    print(f"Loaded {count} detections for {len(dets)} frames.")
    return dets

def main():
    args = parse_args()
    
    cfg = Config.fromfile(args.config)
    
    if hasattr(cfg.model, 'reid'):
        cfg.model.reid.init_cfg = dict(type='Pretrained', checkpoint=args.reid_checkpoint)
        
    # Build model
    print("Initializing model...")
    model = init_model(cfg, device=args.device)
    print("Model initialized.")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for seq in args.seqs:
        print(f"Processing {seq}...")
        
        det_file = os.path.join(args.detections_dir, f"{seq}.txt")
        print(f"Reading detections from: {det_file}")
        all_dets = load_detections(det_file)
        
        if not all_dets:
             print(f"No detections loaded for {seq}!")
        
        img_dir = os.path.join(args.img_root, seq, 'thermal')
        img_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))
        
        if not img_files:
            print(f"No images for {seq}")
            continue
            
        # Reset tracker
        if hasattr(model, 'tracker'):
             if hasattr(model.tracker, 'reset'):
                 model.tracker.reset()
             if hasattr(model.tracker, 'tracks'):
                 model.tracker.tracks = {}
        
        out_file = os.path.join(args.output_dir, f"{seq}.txt")
        
        results = []
        
        for i, img_path in enumerate(img_files):
            frame_id = i + 1
            if i % 50 == 0:
                print(f"Processing frame {frame_id}/{len(img_files)}", end='\r')
            
            img = mmcv.imread(img_path)
            
            if frame_id in all_dets:
                bboxes = all_dets[frame_id] # [x1, y1, x2, y2, score]
                labels = np.zeros(len(bboxes), dtype=np.int64)
            else:
                bboxes = np.zeros((0, 5), dtype=np.float32)
                labels = np.zeros((0,), dtype=np.int64)
            
            # Convert to tensors
            img_tensor = torch.from_numpy(img).to(args.device).permute(2, 0, 1).unsqueeze(0).float()
            
            # Create dummy img_norm_cfg (since we didn't normalize, but the tracker expects this key)
            img_norm_cfg = dict(
                mean=np.array([0., 0., 0.], dtype=np.float32),
                std=np.array([1., 1., 1.], dtype=np.float32),
                to_rgb=True)
            
            img_metas = [{'img_shape': img.shape, 'scale_factor': 1.0, 'flip': False, 'img_norm_cfg': img_norm_cfg}]
            
            # Convert bboxes to tensor
            bboxes_tensor = torch.from_numpy(bboxes).to(args.device)
            labels_tensor = torch.from_numpy(labels).to(args.device)
            
            with torch.no_grad():
                # Let tracker handle ReID extraction if needed
                tracker_output = model.tracker.track(
                    img=img_tensor,
                    img_metas=img_metas,
                    model=model,
                    feats=None, # Let tracker extract feats
                    bboxes=bboxes_tensor,
                    labels=labels_tensor,
                    frame_id=frame_id,
                    rescaled=True
                )
                
                # Check return
                if isinstance(tracker_output, tuple):
                    if len(tracker_output) == 3:
                        bboxes_out, labels_out, ids_out = tracker_output
                        if bboxes_out is not None and bboxes_out.size(0) > 0:
                            for j in range(bboxes_out.size(0)):
                                x1, y1, x2, y2, score = bboxes_out[j].cpu().numpy()
                                obj_id = ids_out[j].item() + 1 # 1-based ID
                                w = x2 - x1
                                h = y2 - y1
                                results.append([frame_id, obj_id, x1, y1, w, h, score, -1, -1, -1])
                    else:
                         # Handle older versions or different return types
                         pass
                elif isinstance(tracker_output, dict):
                     # Handle dict return
                     pass

        # Save results
        print(f"Saved {len(results)} tracks to {out_file}")
        with open(out_file, 'w') as f:
            for res in results:
                f.write(','.join(map(str, res)) + '\n')
                
    print("Tracking finished.")

if __name__ == "__main__":
    main()
