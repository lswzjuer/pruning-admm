import os
import sys
import time
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import yaml
import random
import math
from base_pruner import AdmmPruner



sys.path.append("../")
from  utils import *
from datasets.classification import getDataloader
from models import get_models


def getArgs():
    parser=argparse.ArgumentParser("Train network in cifar10/100/svhn/mnist/tiny_imagenet")
    # model and train setting
    parser.add_argument('--model',default="resnet18",help=" resnet models")
    parser.add_argument('--init_type',default="kaiming",help="weight init func")
    # datasets
    parser.add_argument('--datasets',type=str,default="cifar10",help="datset name")
    parser.add_argument('--root',type=str,default="./datasets",help="datset path")
    parser.add_argument('--class_num',type=int,default=10,help="datasets class name")
    parser.add_argument('--flag',type=str,default="train",help="train or eval")
    # lr and train setting
    parser.add_argument('--pretrain_epochs', default=100, type=int, metavar='N',
                            help='number of total epochs to run')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                            help='number of total epochs to run')
    parser.add_argument('--iters', default=100, type=int, metavar='N',
                            help='number of iterations of ADMM training process')
    parser.add_argument('--fintune_epochs', default=100, type=int, metavar='N',
                            help='number of total epochs to fintune')

    parser.add_argument('--batch_size',type=int,default=128,help="batch size")
    parser.add_argument('--lr', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay 0 for bireal network')
    parser.add_argument('--workers',type=int,default=2,help="dataloader num_workers")
    parser.add_argument("--pin_memory",type=bool,default=True,help="dataloader cache ")
    parser.add_argument('--cutout',default=False, action='store_true')
    parser.add_argument('--eval', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--resume',action='store_true',default=False,help="traing resume")
    parser.add_argument('--resume_path',type=str,default="./checkpoint/model_xx",help="traing resume_path")
    parser.add_argument('--num_best_scores',type=int,default=5,help="num_best_scores")
    parser.add_argument('--optimizer',type=str,default="adam",choices=["adam","sgd","radam"],help="optimizer")
    parser.add_argument('--scheduler',type=str,default="mstep",choices=["warm_up_cos","cos","step","mstep"],help="scheduler")
    parser.add_argument('--step_size',type=int,default=100,help="steplr's step size")
    parser.add_argument('--gamma',type=float,default=0.1,help="learning rate decay")
    # recorder and logging
    parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the trained models',
                        default='../checkpoints/', type=str)
    parser.add_argument('--postfix',
                        help='model folder postfix',
                        default='1', type=str)
    parser.add_argument('--save_freq',type=int,default=1,help="how many epoch to save model")
    parser.add_argument('--print_freq',type=int,default=150,help="print_freq")
    # gpus
    parser.add_argument('--gpus',type=int,default=1,help="gpu number")
    parser.add_argument('--manualSeed',type=int,default=0,help="default init seed")

    args = parser.parse_args()
    return args


def main():

    args = getArgs()
    if args.model in ["resnet18", "mobilenetv2", "mobilenetv1", "ghostnet"]:
        # args.steplist = [30,60,90]
        args.steplist = [50, 75]
    else:
        args.steplist = [150, 220]

    # logging
    projectName = "{}_{}_{}_{}_{}_{}".format(args.model.lower(), args.datasets,
                                             args.epochs, args.batch_size,
                                             args.lr, args.postfix)
    modelDir = os.path.join(args.save_dir, projectName)
    logger = get_logger(modelDir)
    with open(os.path.join(modelDir, "args.yaml"), "w") as yaml_file:  # dump experiment config
        yaml.dump(args, yaml_file)

    # dataloader
    trainLoader = getDataloader(args.datasets, "train", args.batch_size,
                                args.workers, args.pin_memory, args.cutout)
    valLoader = getDataloader(args.datasets, "val", args.batch_size,
                              args.workers, args.pin_memory, args.cutout)

    # device init
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    args.use_cuda = args.gpus > 0 and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.use_cuda else 'cpu')
    if args.use_cuda:
        torch.cuda.manual_seed(args.manualSeed)
        cudnn.benchmark = True

    # model
    model = get_models(args.model,init_type=args.init_type,num_classes=args.class_num)
    logger.info("model is:{} \n".format(model))

    if torch.cuda.device_count() > 1 and args.use_cuda:
        logger.info('use: %d gpus', torch.cuda.device_count())
        model = nn.DataParallel(model)

    # loss and optimazer
    criterion = nn.CrossEntropyLoss()

    if args.use_cuda:
        logger.info("load model and criterion to gpu !")
        model = model.to(args.device)
        criterion = criterion.to(args.device)

    if args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    elif args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     betas=(0.9, 0.999),
                                     weight_decay=args.weight_decay)

    elif args.optimizer.lower() == 'radam':
        optimizer = RAdam(model.parameters(), lr=args.lr,
                          betas=(0.9, 0.999),
                          weight_decay=args.weight_decay)
    else:
        NotImplementedError()
        return

    if args.scheduler.lower() == 'warm_up_cos':
        warm_up_epochs = 5
        warm_up_with_adam = lambda epoch: (epoch + 1) / warm_up_epochs if epoch < warm_up_epochs \
            else 0.5 * (1 + math.cos(math.pi * epoch / args.epochs))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_adam)

    elif args.scheduler.lower() == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0, last_epoch=-1)

    elif args.scheduler.lower() == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args.step_size, gamma=0.1,
                                                    last_epoch=-1)

    elif args.scheduler.lower() == "mstep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.steplist, gamma=args.gamma)
    else:
        NotImplementedError()
        return

    if args.resume:
        model.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        test(model, args.device, criterion, valLoader)
    else:
        # traing
        best_acc = 0
        best_epoch = 0
        for epoch in range(args.pretrain_epochs):
            train(args, model, args.device, trainLoader, criterion, optimizer, epoch)
            scheduler.step()
            acc = test(model, args.device, criterion, valLoader)
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                state_dict = model.state_dict()
        model.load_state_dict(state_dict)
        logger.info('Best acc:', best_acc)
        logger.info('Best epoch:', best_epoch)

        if args.save_model:
            torch.save(state_dict, os.path.join(modelDir, 'model_trained_stage1.pth'))
            logger.info('Model trained saved to %s' % modelDir)


    def trainer(model, optimizer, criterion, epoch, callback):
        return train(args, model, args.device, trainLoader, criterion, optimizer, epoch=epoch, callback=callback)

    def evaluator(model):
        return test(model, args.device, criterion, valLoader)

    # admm pruning
    pattern_config = [{
        'pattern_num': 8,
        'op_types': ['Conv2d'],
        'op_names': []
    }]

    connect_config = [
        {
            'sparsity': 0.5,
            'op_types': ['Conv2d'],
            'op_names': []
        }
    ]
    # Pruner.compress() returns the masked model
    # but for AutoCompressPruner, Pruner.compress() returns directly the pruned model
    pruner = AdmmPruner(model,pattern_config=pattern_config,connectivity_config=connect_config,
                        trainer=trainer, num_iterations=args.iters, training_epochs=args.epochs)
    pruner.compress()
    evaluation_result = evaluator(pruner.bound_model)
    logger.info('Evaluation result (masked model): %s' % evaluation_result)


    # finetuning
    model = pruner.wrapper_model()
    # optimer

    # scheduler

    best_acc = 0
    for epoch in range(args.fintune_epochs):
        train(args, model, args.device, trainLoader, criterion, optimizer, epoch)
        scheduler.step()
        acc = evaluator(model)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(args.experiment_data_dir, 'model_fine_tuned.pth'))

    logger.info('Evaluation result (fine tuned): %s' % best_acc)
    logger.info('Fined tuned model saved to %s' % modelDir)




def train(args, model, device, train_loader, criterion, optimizer, epoch, callback=None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        # callback should be inserted between loss.backward() and optimizer.step()
        if callback:
            callback()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, criterion, val_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset), 100. * accuracy))

    return accuracy





if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Eval time: %ds.', duration)

