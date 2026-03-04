import optuna
import argparse
import yaml
import torch
from pathlib import Path
from train import train
from utils.general import increment_path
from optuna.integration import TensorBoardCallback


def objective(trial, base_opt):
    hyp = {
        "lr0":              trial.suggest_float("lr0", 1e-4, 1e-1, log=True),
        "lrf":              trial.suggest_float("lrf", 0.01, 0.2),
        "momentum":         trial.suggest_float("momentum", 0.8, 0.98),
        "weight_decay":     trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "warmup_epochs":    trial.suggest_float("warmup_epochs", 1.0, 5.0),
        "warmup_momentum":  0.8,
        "warmup_bias_lr":   0.1,
        "box":              trial.suggest_float("box", 0.02, 0.2),
        "cls":              trial.suggest_float("cls", 0.1, 2.0),
        "cls_pw":           1.0,
        "obj":              trial.suggest_float("obj", 0.2, 2.0),
        "obj_pw":           1.0,
        "iou_t":            0.2,
        "anchor_t":         4.0,
        "fl_gamma":         0.0,
        "hsv_h":            0.015,
        "hsv_s":            0.7,
        "hsv_v":            0.4,
        "degrees":          0.0,
        "translate":        0.2,
        "scale":            0.9,
        "shear":            0.0,
        "perspective":      0.0,
        "flipud":           0.0,
        "fliplr":           0.5,
        "mosaic":           1.0,
        "mixup":            0.15,
        "copy_paste":       0.0,
        "paste_in":         0.15,
        "loss_ota":         1,
        "label_smoothing":  0.0,
    }

    opt = argparse.Namespace(**vars(base_opt))
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = train(hyp, opt, device)

    map50 = results[2]
    return map50


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',       default="data/visdrone.yaml")
    parser.add_argument('--cfg',        default='cfg/training/yolov7.yaml')
    parser.add_argument('--weights',    default='yolov7.pt')
    parser.add_argument('--epochs',     type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--img-size',   nargs='+', type=int, default=[640, 640])
    parser.add_argument('--project',    default='runs/optuna')
    parser.add_argument('--name',       default='trial')
    parser.add_argument('--n-trials',   type=int, default=20)
    parser.add_argument('--device',     default='')
    parser.add_argument('--workers',     type=int, default=8)
    parser.add_argument('--single-cls', action='store_true')
    parser.add_argument('--adam',       action='store_true')
    parser.add_argument('--linear-lr',  action='store_true')
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument('--multi-scale', action='store_true')
    parser.add_argument('--evolve',     action='store_true')
    parser.add_argument('--resume',     nargs='?', const=True, default=False)
    parser.add_argument('--rect',       action='store_true')
    parser.add_argument('--notest',     action='store_true')
    parser.add_argument('--nosave',     action='store_true')
    parser.add_argument('--cache-images', action='store_true')
    parser.add_argument('--image-weights', action='store_true')
    parser.add_argument('--sync-bn',    action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--freeze',     nargs='+', type=int, default=[0])
    parser.add_argument('--v5-metric',  action='store_true')
    parser.add_argument('--quad',       action='store_true')
    parser.add_argument('--entity',     default=None)
    parser.add_argument('--upload_dataset', action='store_true')
    parser.add_argument('--bbox_interval', type=int, default=-1)
    parser.add_argument('--save_period', type=int, default=-1)
    parser.add_argument('--artifact_alias', default='latest')
    opt = parser.parse_args()

    opt.world_size = 1
    opt.global_rank = -1
    opt.total_batch_size = opt.batch_size
    opt.imgsz = opt.img_size[0]
    opt.imgsz_test = opt.img_size[-1]
    opt.hyp = 'data/hyp.scratch.p5.yaml'
    opt.noautoanchor = False
    opt.bucket = ''
    opt.exist_ok = False
    opt.yaml = ''

    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        storage='sqlite:///optuna_yolov7.db',
        study_name='yolov7_hpo',
        load_if_exists=True
    )
    tb_callback = TensorBoardCallback("runs/optuna_tb/", metric_name="mAP")
    study.optimize(
        lambda trial: objective(trial, opt),
        n_trials=opt.n_trials,
        callbacks=[tb_callback]
    )

    print('\nBest Trial:')
    print(f'    mAP@0.5: {study.best_value:.4f}')
    print(f'    Params: {study.best_params}')
