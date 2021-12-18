import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import argparse

import gc
import torch
import torch.nn as nn

from data import *
import networks
from utils import seed_everything, resize_module, str2bool

def predict(model, test_loader, device):
    test_value = []
    with torch.no_grad():
        for before_img, after_img in tqdm(iter(test_loader)):
            before_img, after_img = before_img.float().to(device), after_img.float().to(device)
            preds = model(before_img, after_img)
            preds = preds.squeeze(1).detach().cpu().float()

            test_value.extend(preds)
    return test_value

def inference_data(model, test_loader, tta, model_paths, device):
    test_values = []
    for model_path in model_paths:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        model.to(device)
        for _ in range(tta):
            test_value = predict(model, test_loader, device)
            test_values.append(test_value)
    
    return test_values


def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    df = make_test_df(args.test_path)

    if args.do_resize:
        for before_path, after_path in tqdm(zip(df['before_file_path'], df['after_file_path'])):
            resize_module(before_path, args.size)
            resize_module(after_path, args.size)

    test_loader = prepare_dataloader(df, 'test', args)
    test_values = []
    if args.swin:
        model = getattr(networks, 'swin')(pretrained=False)
        test_values.extend(inference_data(model, test_loader, args.tta_num, args.swin_paths, device))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if args.vit:
        model = getattr(networks, 'vit')(pretrained=False)
        test_values.extend(inference_data(model, test_loader, args.tta_num, args.vit_paths, device))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    submission = pd.read_csv(args.submission_path)
    test_values = np.array(test_values) + 1
    test_values = np.mean(test_values, axis=0)
    submission['time_delta'] = test_values
    submission.to_csv(args.save_path, index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--do_resize",
        default=True,
        type=str2bool
    )
    parser.add_argument(
        "--size",
        default=224,
        type=int
    )
    parser.add_argument(
        "--tta",
        default=True,
        type=str2bool
    )
    parser.add_argument(
        "--tta_num",
        default=3,
        type=int
    )
    parser.add_argument(
        "--swin",
        default=False,
        type=str2bool
    )
    parser.add_argument(
        "--swin_paths",
        default=[],
        type=str,
        nargs='*'
    )
    parser.add_argument(
        "--vit",
        default=False,
        type=str2bool
    )
    parser.add_argument(
        "--vit_paths",
        default=[],
        type=str,
        nargs='*'
    )
    parser.add_argument(
        "--test_path",
        default='./open/test_dataset',
        type=str
    )
    parser.add_argument(
        "--submission_path",
        default="./open/sample_submission.csv",
        type=str
    )
    parser.add_argument(
        "--save_path",
        default="./submission.csv",
        type=str
    )
    parser.add_argument(
        "--workers",
        default=4,
        type=int
    )
    
    args = parser.parse_args()
    main(args)
