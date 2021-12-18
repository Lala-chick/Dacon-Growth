import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import argparse

from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import math
import gc

from data import *
import networks
from utils import seed_everything, resize_module, str2bool
from schedulers import CustomCosineAnnealingWarmUpRestarts

def train_model(model, optimizer, train_loader, valid_loader, scheduler, device, fold, args):
    model.to(device)
    criterion = nn.MSELoss().to(device)
    best_rmse = 100

    for epoch in range(1, args.epoch+1):
        model.train()
        train_metric = []
        valid_metric = []
        for before_img, after_img, target in tqdm(iter(train_loader)):
            before_img, after_img, target = before_img.float().to(device), after_img.float().to(device), target.float().to(device)
            target -= 1.
            optimizer.zero_grad()

            with autocast():
                preds = model(before_img, after_img)
                loss = criterion(preds.squeeze(1), target)
                loss = torch.sqrt(loss)

            loss.backward()
            optimizer.step()

            train_metric.append(loss.item())
        
        train_rmse = np.mean(train_metric)

        if scheduler is not None:
            scheduler.step()

        model.eval()
        with torch.no_grad():
            for before_img, after_img, target in tqdm(iter(valid_loader)):
                before_img, after_img, target = before_img.float().to(device), after_img.float().to(device), target.float().to(device)
                target -= 1.
                preds = model(before_img, after_img)
                loss = criterion(preds.squeeze(1), target)
                loss = torch.sqrt(loss)

                valid_metric.append(loss.item())

        valid_rmse = np.mean(valid_metric)

        print(f"Epoch [{epoch}] Train RMSE: [{train_rmse: .5f}] Valid RMSE: [{valid_rmse: .5f}]\n")

        if best_rmse > valid_rmse:
            best_rmse = valid_rmse
            torch.save(model.state_dict(), f"{args.save_path}/{args.model}_{fold}.pth")
            print("Model is Saved!")

def main(args):
    seed_everything(args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    df = make_train_df(args.train_path)
    if args.do_resize:
        for before_path, after_path in tqdm(zip(df['before_file_path'], df['after_file_path'])):
            resize_module(before_path, args.size)
            resize_module(after_path, args.size)

    folds = KFold(shuffle=True, random_state=args.seed)
    fold = 1
    for trn_idx, val_idx in folds.split(df):
        print(f"Fold {fold} START!\n")
        train_df, valid_df = df.iloc[trn_idx], df.iloc[val_idx]

        train_loader = prepare_dataloader(train_df, 'train', args)
        valid_loader = prepare_dataloader(valid_df, 'valid', args)
        train_length = args.epoch*len(train_loader)
        model = getattr(networks, args.model)(pretrained=args.pretrained)
        optimizer = getattr(torch.optim, args.optimizer)(params=model.parameters(), lr=0, weight_decay=args.weight_decay)
        scheduler = CustomCosineAnnealingWarmUpRestarts(optimizer, T_0=train_length, T_mult=1, eta_max=args.lr, T_up=train_length//10)

        train_model(model, optimizer, train_loader, valid_loader, scheduler, device, fold, args)
        fold += 1
        del model, optimizer, train_loader, valid_loader, scheduler
        gc.collect()
        torch.cuda.empty_cache()
        if fold > args.fold:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        default=41,
        type=int
    )
    parser.add_argument(
        "--train_path",
        default="/content/open/train_dataset"
    )
    parser.add_argument(
        "--save_path",
        default='./trained_models',
        type=str
    )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int
    )
    parser.add_argument(
        "--workers",
        default=4,
        type=int
    )
    parser.add_argument(
        "--optimizer",
        default="Adam",
        type=str
    )
    parser.add_argument(
        "--model",
        default="vit",
        type=str
    )
    parser.add_argument(
        "--pretrained",
        default=True,
        type=str2bool
    )
    parser.add_argument(
        "--epoch",
        default=30,
        type=int
    )
    parser.add_argument(
        "--lr",
        default=1e-4,
        type=float
    )
    parser.add_argument(
        "--weight_decay",
        default=1e-5,
        type=float
    )
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
        "--fold",
        default=1,
        type=int
    )

    args = parser.parse_args()
    main(args)