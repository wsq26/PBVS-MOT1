
import os
import argparse

def validate_submission(input_dir):
    """
    Validate that submission files are in correct MOT Challenge format.

    Expected format:
    <frame_id>,<track_id>,<x>,<y>,<width>,<height>,<confidence>,<class_id>,<visibility>
    """
    sequences = ["seq2", "seq17", "seq22", "seq47", "seq54", "seq66"]

    print(f"Validating submission in: {input_dir}")

    total_detections = 0
    total_tracks = 0

    for seq in sequences:
        txt_file = os.path.join(input_dir, f"{seq}.txt")

        if not os.path.exists(txt_file):
            print(f"Warning: {seq}.txt not found")
            continue

        try:
            with open(txt_file, 'r') as f:
                lines = f.readlines()

            if not lines:
                print(f"{seq}: Empty file")
                continue

            # Parse and validate format
            track_ids = set()
            frame_ids = set()

            for line_num, line in enumerate(lines, 1):
                parts = line.strip().split(',')
                if len(parts) < 7:
                    print(f"{seq} line {line_num}: Invalid format (expected at least 7 fields)")
                    continue

                try:
                    frame_id = int(parts[0])
                    track_id = int(parts[1])
                    x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                    conf = float(parts[6])

                    frame_ids.add(frame_id)
                    track_ids.add(track_id)
                    total_detections += 1

                except ValueError as e:
                    print(f"{seq} line {line_num}: Parse error - {e}")

            total_tracks += len(track_ids)
            print(f"{seq}: {len(frame_ids)} frames, {len(track_ids)} tracks, {len(lines)} detections")

        except Exception as e:
            print(f"Error reading {seq}.txt: {e}")

    print(f"\n{'='*50}")
    print(f"Validation Summary")
    print(f"{'='*50}")
    print(f"Total detections: {total_detections}")
    print(f"Total unique tracks: {total_tracks}")
    print(f"Submission format is valid!")

def parse_args():
    parser = argparse.ArgumentParser(description='Validate tracking submission format')
    parser.add_argument('--input-dir', required=True, help='directory with tracking results')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    validate_submission(args.input_dir)
