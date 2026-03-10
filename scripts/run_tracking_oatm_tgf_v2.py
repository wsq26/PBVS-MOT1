#!/usr/bin/env python
"""
Run OATM+TGF v2 (Enhanced Thermal Gradient Flow) tracking pipeline.
"""
import os
import argparse
import subprocess


def parse_args():
    p = argparse.ArgumentParser(description='Run ByteTrack-ReID-OATM-TGF-v2 and interpolation')
    p.add_argument('--detections-dir', required=True)
    p.add_argument('--output-dir', default='submission_results_oatm_tgf_v2')
    p.add_argument('--reid-checkpoint', default=None, help='path to ReID checkpoint')
    p.add_argument('--python-bin', default='python', help='python executable path')
    p.add_argument('--workspace', default='.', help='workspace directory')
    return p.parse_args()


def run(cmd, env=None):
    print('RUN:', ' '.join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main():
    args = parse_args()
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env['PYTHONNOUSERSITE'] = '1'
    env['PYTHONPATH'] = os.path.join(args.workspace, 'mmtracking')

    track_cmd = [
        args.python_bin,
        os.path.join(args.workspace, 'run_tracking_custom.py'),
        '--config', os.path.join(args.workspace, 'mmtracking/configs/mot/bytetrack/bytetrack_reid_oatm_tgf_v2.py'),
        '--reid-checkpoint', args.reid_checkpoint,
        '--detections-dir', args.detections_dir,
        '--output-dir', os.path.join(args.workspace, args.output_dir)
    ]
    run(track_cmd, env=env)

    interp_cmd = [
        args.python_bin,
        os.path.join(args.workspace, 'scripts/interpolate_results.py'),
        '--input-dir', args.output_dir,
        '--output-dir', f"{args.output_dir}_interpolated"
    ]
    run(interp_cmd, env=env)


if __name__ == '__main__':
    main()
