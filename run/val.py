# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

"""
Evaluate on the benchmark of your choice. MOT16, 17 and 20 are donwloaded and unpackaged automatically when selected.
Mimic the structure of either of these datasets to evaluate on your custom one

Usage:

    $ python3 val.py --tracking-method strongsort --benchmark MOT16
                     --tracking-method ocsort     --benchmark MOT17
                     --tracking-method ocsort     --benchmark <your-custom-dataset>
"""

import argparse
from ultralytics.utils.checks import print_args
from pathlib import Path
from dettrack.tracking.evaluator import Evaluator

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='path to dataset')
    parser.add_argument('--yolo-model', type=str, default=Path('weights/yolov8n.pt'), help='model.pt path(s)')
    parser.add_argument('--reid-model', type=str, default=Path('weights/osnet_x0_25_msmt17.pt'))
    parser.add_argument('--tracking-method', type=str, default='deepocsort',
                        help='strongsort, ocsort')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--classes', nargs='+', type=str, default=['0'],
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=Path('runs/val'),
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--gsi', action='store_true',
                        help='apply gsi to results')
    parser.add_argument('--benchmark', type=str, default='MOT17-mini',
                        help='MOT16, MOT17, MOT20')
    parser.add_argument('--split', type=str, default='train',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--eval-existing', action='store_true',
                        help='evaluate existing results under project/name/mot')
    parser.add_argument('--conf', type=float, default=0.45,
                        help='confidence threshold')
    parser.add_argument('--imgsz', '--img-size', nargs='+', type=int, default=[1280],
                        help='inference size h,w')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save', action='store_true',
                        help='save video tracking results')
    parser.add_argument('--processes-per-device', type=int, default=2,
                        help='how many subprocesses can be invoked per GPU (to manage memory consumption)')
    opt = parser.parse_args()
    device = []

    for a in opt.device.split(','):
        try:
            a = int(a)
        except ValueError:
            pass
        device.append(a)
    opt.device = device

    print_args(vars(opt))
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    # check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    e = Evaluator(opt)
    e.run(opt)
