from torch import nn
import torch
import os
from config.config import args
import os
from tqdm import tqdm
from dataset.load_data import *
from utils.metric_train import create_metrics

if args.USE_BASELINE:
    from model.net import MoireCNN
else:
    from model.net_shapeconv import MoireCNN
    # args.EXP_NAME = 'ShapeMoire'


def train(model, train_loader, criterion, epoch, lr, device, iters, optimizer):
    model.train()
    tbar = tqdm(train_loader)
    total_loss = 0.0
    for batch_idx, data in enumerate(tbar):
        img_train = data['in_img'].to(device)
        label = data['label'].to(device)

        if not args.USE_BASELINE:
            base_img_train = torch.mean(img_train, dim=[2, 3], keepdim=True)
            shape_img_train = img_train - base_img_train
            img_train = torch.cat((img_train, shape_img_train), dim=0)
            base_label = torch.mean(label, dim=[2, 3], keepdim=True)
            shape_label = label - base_label
            label = torch.cat((label, shape_label), dim=0)

        optimizer.zero_grad()
        output = model(img_train)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        iters += 1

        total_loss += loss.item()
        avg_train_loss = total_loss / (batch_idx + 1)

        desc = 'Training  : Epoch %d, lr %.7f, Avg. Loss = %.5f' % (epoch, lr, avg_train_loss)
        tbar.set_description(desc)
        tbar.update()

    return iters, avg_train_loss


def val(model, val_loader, epoch, device, compute_metrics):
    model.eval()

    tbar = tqdm(val_loader)
    idx, loss_sum, total_psnr = 0, 0.0, 0
    criterion = nn.MSELoss()

    for batch_idx, data in enumerate(tbar):
        in_img = data['in_img'].to(device)
        target = data['label'].to(device)

        with torch.no_grad():
            output = model(in_img)
            loss = criterion(output, target)

        loss_sum += loss.item()
        idx += 1

        cur_psnr = compute_metrics.compute(output, target)
        total_psnr += cur_psnr

    loss_sum /= idx
    total_psnr /= (batch_idx + 1)

    return total_psnr, loss_sum


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.fill_(0)


if __name__ == '__main__':
    args.LOGS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'logs')
    args.NETS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'net_checkpoints')
    os.makedirs(args.LOGS_DIR, exist_ok=True)
    os.makedirs(args.NETS_DIR, exist_ok=True)

    # random seed
    random.seed(args.SEED)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed_all(args.SEED)
    if args.SEED == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # dataloader
    train_path = args.TRAIN_DATASET
    train_loader = create_dataset(args, data_path=train_path, mode='train')
    Validate_path = args.TEST_DATASET
    args.BATCH_SIZE = 1
    val_loader = create_dataset(args, data_path=Validate_path, mode='test')

    compute_metrics = create_metrics(args, device=device, use_fast=True)

    print('loaded dataset successfully!')
    model = MoireCNN().to(device)

    if args.LOAD_EPOCH:
        load_path = " "
        model = torch.load(load_path)
    else:
        model.apply(weights_init)

    criterion = nn.MSELoss()
    lr = 0.0001
    best_loss, last_loss = 100.0, 100.0
    best_psnr = 0
    iters = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.00001)

    for epoch in range(args.LOAD_EPOCH + 1, args.EPOCHS + 1):
        iters, avg_train_loss = train(model, train_loader, criterion, epoch, lr, device, iters, optimizer)

        cur_psnr, current_loss = val(model, val_loader, epoch, device, compute_metrics)

        if current_loss < best_loss:
            best_loss = current_loss

        if current_loss > last_loss and lr > 1e-6:
            lr *= 0.1
        last_loss = current_loss

        # save best epoch
        bestfilename = args.NETS_DIR + f'/best_epoch{epoch}_{cur_psnr:.4f}.tar'

        if best_psnr <= cur_psnr:
            torch.save({
                'learning_rate': lr,
                'iters': iters,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict()
            }, bestfilename)

            best_psnr = cur_psnr

