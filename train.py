import os

from model import CSRNet

from utils import save_checkpoint

from make_lists import make_lists

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms

import argparse
import dataset
import time
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch CSRNet')

parser.add_argument('train_path', metavar='TRAIN',
                    help='path to train folder')

## argument added to support malaria dataset
parser.add_argument('--json','-j', metavar='JSON_FILE',
                    help='specifies whether the train_path is a json file or an image folder (y or n)')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None,type=str,
                    help='path to the pretrained model')

parser.add_argument('gpu',metavar='GPU', type=str,
                    help='GPU id to use.')

parser.add_argument('task',metavar='TASK', type=str,
                    help='task id to use.')

def main():
    
    global args,best_prec1
    
    best_prec1 = 1e6
    
    args = parser.parse_args()
    args.original_lr = 1e-7
    args.lr = 1e-7
    args.batch_size    = 1
    args.momentum      = 0.95
    args.decay         = 5*1e-4
    args.start_epoch   = 0
    args.epochs = 400
    args.workers = 4
    args.steps = [-1,1,100,150]
    args.scales = [1,1,1,1]
    args.seed = time.time()
    args.print_freq = 30
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(args.seed)
    model = CSRNet()
    
    model = model.cuda()
    
    criterion = nn.MSELoss(size_average=False).cuda()
    
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.decay)

    train_list, val_list = make_lists(args.train_path, args.json=='y')

    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))
            
    loss_values = []
    mae_values = []
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        loss_avg = train(train_list, model, criterion, optimizer, epoch)
        prec1 = validate(val_list, model, criterion)
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best,args.task)

        loss_values.append(loss_avg)
        mae_values.append(prec1)
        if epoch % 10 == 0:
            print('saving current loss curve..')
            plt.figure()
            plt.plot(loss_values)
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.savefig('loss_curve.png')
            print('saving current mae curve..')
            plt.figure()
            plt.plot(mae_values)
            plt.xlabel('epochs')
            plt.ylabel('average MAE')
            plt.savefig('mae_curve.png')
            print('done.')

    ## plot latest loss values
    plt.figure()
    plt.plot(loss_values)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig('loss_curve_last.png')
    ## plot latest mae values
    plt.figure()
    plt.plot(mae_values)
    plt.xlabel('epochs')
    plt.ylabel('average MAE')
    plt.savefig('mae_curve_latest')

def train(train_list, model, criterion, optimizer, epoch):
    
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                       shuffle=True,
                       transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]), 
                       train=True, 
                       seen=model.seen,
                       batch_size=args.batch_size,
                       num_workers=args.workers),
        batch_size=args.batch_size)
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))
    
    model.train()
    end = time.time()

    for i,(img, target)in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        forward_start = time.perf_counter()
        img = img.cuda()
        img = Variable(img)
        forward_end = time.perf_counter()
        output = model(img)

        target_start = time.perf_counter()        
        target = target.type(torch.FloatTensor).unsqueeze(0).cuda()
        target = Variable(target)
        target_end = time.perf_counter()

        
        loss_start = time.perf_counter()
        loss = criterion(output, target)
        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss_end = time.perf_counter()

        backward_start = time.perf_counter()
        loss.backward()
        optimizer.step()    
        batch_time.update(time.time() - end)
        backward_end = time.perf_counter()
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Forward {forward_time}\t'
                  'Target {target_time}\t'
                  'Loss* {loss_time}\t'
                  'Backward {backward_time}\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, forward_time=forward_end-forward_start,
                   target_time=target_end-target_start, loss_time=loss_end-loss_start,
                   backward_time=backward_end-backward_start))
    return losses.avg

def validate(val_list, model, criterion):
    print ('begin test')
    test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(val_list,
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),  train=False),
    batch_size=args.batch_size)    
    
    model.eval()
    
    mae = 0
    
    for i,(img, target) in enumerate(test_loader):
        img = img.cuda()
        img = Variable(img)
        output = model(img)
        
        mae += abs(output.data.sum()-target.sum().type(torch.FloatTensor).cuda())
        
    mae = mae/len(test_loader)    
    print(' * MAE {mae:.3f} '
              .format(mae=mae))

    return mae    

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    
    
    args.lr = args.original_lr
    
    for i in range(len(args.steps)):
        
        scale = args.scales[i] if i < len(args.scales) else 1
        
        
        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    
    
if __name__ == '__main__':
    main()        
