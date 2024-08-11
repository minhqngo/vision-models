import argparse
import os
import torch
from tensorboardX import SummaryWriter
from torch import optim, nn
from tqdm import tqdm
from datetime import datetime

from tiny_imagenet_dataset import tiny_imagenet_loader
from torch_dataset_factory import torch_dataset_factory
from models import create_model, model_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='mobilenet_v3_large')
    parser.add_argument('--ds-name', type=str, default='tiny-imagenet', 
                        choices=['tiny-imagenet', 'cifar10', 'cifar100', 'svhn'])
    parser.add_argument('--ds-dir', type=str, default=None)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    if args.ds_name == 'tiny-imagenet':
        train_loader, val_loader, n_classes = tiny_imagenet_loader(args.ds_dir, args.bs)
    else:
        train_loader, val_loader, n_classes = torch_dataset_factory(args.ds_name, args.ds_dir, args.bs)
    
    model = create_model(args.arch, n_classes=n_classes)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu" and args.device != -1)
    model.to(device)
    model_summary(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epoch // 10, gamma=0.9)

    date_str = datetime.now().strftime('%Y%m%d-%H%M')
    save_dir = os.path.join('./weights', args.arch, date_str)
    os.makedirs(save_dir, exist_ok=True)

    log_dir = os.path.join('./logs', args.arch, date_str)
    os.makedirs(log_dir, exist_ok=True)
    logger = SummaryWriter(log_dir)

    # Train the model
    best_acc = float('-inf')
    for epoch in range(args.epoch):
        correct = 0
        total = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Train epoch {epoch + 1}/{args.epoch}")

        model.train()
        for i, (inputs, targets) in pbar:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Load tensor on device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total += targets.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == targets).sum().item()

            pbar.set_postfix(loss=loss.item(), train_acc=correct / total)
            logger.add_scalar('Training/Loss', loss.item(), epoch)
        logger.add_scalar('Training/Learning rate', optimizer.param_groups[0]['lr'], epoch)

        model.eval()
        correct = 0
        total = 0
        pbar = tqdm(val_loader, total=len(val_loader), desc=f"Test epoch {epoch + 1}/{args.epoch}")
        with torch.no_grad():
            for inputs, targets in pbar:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                acc = correct / total
                pbar.set_postfix(acc=acc)
            acc = correct / total
            logger.add_scalar('Test/Accuracy', acc, epoch)

            if acc > best_acc:
                print(f"Test accuracy improved from {best_acc:.3f} to {acc:.3f}")
                best_acc = acc
                torch.save(model.state_dict(), os.path.join(save_dir, 'best.pth'))
