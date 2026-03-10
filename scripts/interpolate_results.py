
import os
import numpy as np
import zipfile
import argparse

# ==============================================================================
# INTERPOLATION & FILTERING LOGIC
# ==============================================================================
def interpolate_tracks(data, max_gap=1):
    if len(data) == 0: return data
    track_ids = np.unique(data[:, 1])
    interpolated_data = []

    for tid in track_ids:
        track_data = data[data[:, 1] == tid]
        track_data = track_data[np.argsort(track_data[:, 0])]
        frames = track_data[:, 0]

        # Add original
        for row in track_data: interpolated_data.append(row)

        # Interpolate
        for i in range(len(frames) - 1):
            f1, f2 = int(frames[i]), int(frames[i+1])
            gap = f2 - f1
            if 1 < gap <= max_gap:
                start_box = track_data[i, 2:6]
                end_box = track_data[i+1, 2:6]
                for step in range(1, gap):
                    alpha = step / gap
                    interp_box = start_box * (1 - alpha) + end_box * alpha
                    interp_frame = f1 + step
                    interp_score = (track_data[i, 6] + track_data[i+1, 6]) / 2
                    interpolated_data.append([interp_frame, tid, *interp_box, interp_score])

    return np.array(interpolated_data)

def filter_tracks(data, min_len=10, min_score=0.7):
    """
    Keep track if:
    1. Length >= min_len
    OR
    2. Average Score >= min_score (recover short but high confidence tracks)
    """
    if len(data) == 0: return data

    track_ids = np.unique(data[:, 1])
    valid_data = []

    for tid in track_ids:
        track_data = data[data[:, 1] == tid]
        length = len(track_data)
        avg_score = np.mean(track_data[:, 6])

        if length >= min_len or avg_score >= min_score:
            valid_data.append(track_data)

    if not valid_data: return np.array([])
    return np.vstack(valid_data)

# ==============================================================================
# MAIN SCRIPT
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Interpolate Tracking Results')
    parser.add_argument('--input-dir', required=True, help='input directory with tracking results')
    parser.add_argument('--output-dir', default=None, help='output directory')
    return parser.parse_args()

def main():
    args = parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir if args.output_dir else f"{input_dir}_interpolated"
    sequences = ["seq2", "seq17", "seq22", "seq47", "seq54", "seq66"]

    if not os.path.exists(input_dir):
        print(f"Error: {input_dir} not found.")
        return

    # Load raw data once
    raw_data = {}
    for seq in sequences:
        txt_file = os.path.join(input_dir, f"{seq}.txt")
        if os.path.exists(txt_file):
            data = []
            with open(txt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 7:
                        data.append([int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3]),
                                     float(parts[4]), float(parts[5]), float(parts[6])])
            raw_data[seq] = np.array(data) if data else np.array([])
        else:
            raw_data[seq] = np.array([])

    # Generate Final Results with fixed optimal parameters
    print(f"Generating results to {output_dir}...")
    print(f"Using parameters: Gap=1, MinLen=10, MinScore=0.7")

    os.makedirs(output_dir, exist_ok=True)

    for seq, data in raw_data.items():
        interp = interpolate_tracks(data, max_gap=1)
        filtered = filter_tracks(interp, min_len=10, min_score=0.7)

        out_file = os.path.join(output_dir, f"{seq}.txt")
        if len(filtered) > 0:
            ind = np.lexsort((filtered[:, 1], filtered[:, 0]))
            sorted_data = filtered[ind]
            with open(out_file, 'w') as f:
                for row in sorted_data:
                    line = f"{int(row[0])},{int(row[1])},{row[2]:.1f},{row[3]:.1f},{row[4]:.1f},{row[5]:.1f},{row[6]:.2f},1,-1,-1"
                    f.write(line + "\n")
        else:
            with open(out_file, 'w') as f: pass

    # Zip results
    zip_filename = f"{output_dir}.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                if file.endswith('.txt'):
                     zf.write(os.path.join(root, file), arcname=file)
    print(f"Archive created: {zip_filename}")
    print("Done!")

if __name__ == "__main__":
    main()
