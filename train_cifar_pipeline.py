import os
import sys
import time
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
from torch.optim import optimizer
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import yaml
import random
import math
from base_pruner import AdmmPruner


sys.path.append("../")
from utils import *
from datasets.classification import getDataloader
from models import get_models

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def getArgs():
    parser=argparse.ArgumentParser("Train network in cifar10/100/svhn/mnist/tiny_imagenet")
    # model and train setting
    parser.add_argument('--model',default="resnet20",help=" resnet models")
    parser.add_argument('--init_type',default="kaiming",help="weight init func")
    # datasets
    parser.add_argument('--datasets',type=str,default="cifar10",help="datset name")
    parser.add_argument('--root',type=str,default="./datasets",help="datset path")
    parser.add_argument('--class_num',type=int,default=10,help="datasets class name")
    parser.add_argument('--flag',type=str,default="train",help="train or eval")
    # lr and train setting
    parser.add_argument('--pretrain_epochs', default=300, type=int, metavar='N',
                            help='number of total epochs to run')
    parser.add_argument('--epochs', default=3, type=int, metavar='N',
                            help='number of total epochs to run')
    parser.add_argument('--iters', default=50, type=int, metavar='N',
                            help='number of iterations of ADMM training process')
    parser.add_argument('--fintune_epochs', default=100, type=int, metavar='N',
                            help='number of total epochs to fintune')

    parser.add_argument('--batch_size',type=int,default=128,help="batch size")
    parser.add_argument('--lr', default=0.01, type=float,
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
    parser.add_argument('--optimizer',type=str,default="sgd",choices=["adam","sgd","radam"],help="optimizer")
    parser.add_argument('--scheduler',type=str,default="warm_up_cos",choices=["warm_up_cos","cos","step","mstep"],help="scheduler")
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

    parser.add_argument('--pattern_num',type=int,default=8,help="default pattern num ")
    parser.add_argument('--sp',type=float,default=0.5,help="sparse rate  ")
    parser.add_argument('--row1',type=int,default=5e-4,help="default pattern num ")
    parser.add_argument('--row2',type=int,default=5e-4,help="default pattern num ")
    
    args = parser.parse_args()
    return args



def get_optimizer_scheduler(args,model):

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
    return optimizer,scheduler


def main():

    args = getArgs()
    if args.model in ["resnet18", "mobilenetv2", "mobilenetv1", "ghostnet"]:
        # args.steplist = [30,60,90]
        args.steplist = [50, 75]
    else:
        args.steplist = [150, 220]
    # logging
    projectName = "{}_{}_{}_{}_{}_{}_{}_{}_p{}_sp{}_{}".format(args.model.lower(), args.datasets,
                                             args.epochs, args.batch_size,
                                             args.lr, args.epochs,args.iters,
                                             args.fintune_epochs,args.pattern_num,
                                             args.sp,
                                             args.postfix)
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

    if args.gpus > 1 and args.use_cuda:
        logger.info('use: %d gpus', torch.cuda.device_count())
        model = nn.DataParallel(model)

    # loss and optimazer
    criterion = nn.CrossEntropyLoss()

    if args.use_cuda:
        logger.info("load model and criterion to gpu !")
        model = model.to(args.device)
        criterion = criterion.to(args.device)

    optimizer,scheduler = get_optimizer_scheduler(args,model)
    if args.resume:
        model.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        test(model, args.device, criterion, valLoader,logger=logger)
    else:
        # traing
        best_acc = 0
        best_epoch = 0
        best_state_dict = None
        for epoch in range(args.pretrain_epochs):
            train(args, model, args.device, trainLoader, criterion, optimizer, epoch, callback=None,logger=logger)
            scheduler.step()
            acc = test(model, args.device, criterion, valLoader,logger=logger)
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                best_state_dict = model.state_dict()
                torch.save(best_state_dict, os.path.join(modelDir, 'model_trained_stage1.pth'))
        assert best_state_dict is not None,"state dict must be exists !"
        model.load_state_dict(best_state_dict)
        logger.info('Best acc:{}  Best epoch:{}'.format(best_acc,best_epoch))
        logger.info('Model trained saved to %s' % modelDir)
        return 


    def trainer(model, optimizer, criterion, epoch, callback):
        return train(args, model, args.device, trainLoader, criterion, optimizer, epoch=epoch, callback=callback,
                    logger = logger)

    def evaluator(model):
        return test(model, args.device, criterion, valLoader,logger = logger)

    # admm pruning
    pattern_config = [{
            'pattern_num': args.pattern_num,
            'op_types': ['Conv2d'],
            'op_names': []
        },
        {
            "exclude" : True,
            'op_types': ['Conv2d'],
            'op_names': ["features.0.0"]
        }
    ]

    connect_config = [
        {
            'sparsity': args.sp,
            'op_types': ['Conv2d'],
            'op_names': []
        },
        {
            "exclude" : True,
            'op_types': ['Conv2d'],
            'op_names': ["features.0.0"]
        }
    ]
    # Pruner.compress() returns the masked model
    # but for AutoCompressPruner, Pruner.compress() returns directly the pruned model
    pruner1 = AdmmPruner(model,pattern_config=pattern_config,connectivity_config=connect_config,
                        trainer=trainer, num_iterations=args.iters, training_epochs=args.epochs,
                        row1=args.row1, row2=args.row2, base_algo='l2',logger = logger
                        )
    pruner1.compress_pattern()
    #pruner1.compress_connect()
    # visual changerate and loss
    pruner1.visiual_changerate(modelDir)
    pruner1.visiual_lossrate(modelDir)
    evaluation_result = evaluator(pruner1.bound_model)
    logger.info('Evaluation result (connect admm model): %s' % evaluation_result)
    # do mask 
    pruner1.wrapper_model()
    evaluation_result = evaluator(pruner1.bound_model)
    logger.info('Evaluation result (connect masked model): %s' % evaluation_result) 
    # save model
    maskmodeldir = os.path.join(modelDir,"model_masked1.pth")
    maskdir = os.path.join(modelDir,"mask1.pth")
    pruner1.export_model(model_path=maskmodeldir,mask_path=maskdir)
    # finetune
    # optimizer1,scheduler1 = get_optimizer_scheduler(args,pruner1.bound_model)

    # best_acc = 0
    # for epoch in range(args.fintune_epochs):
    #     train(args, pruner1.bound_model, args.device, trainLoader, criterion, optimizer1, epoch, logger=logger)
    #     scheduler1.step()
    #     acc = evaluator(pruner1.bound_model)
    #     if acc > best_acc:
    #         best_acc = acc
    #         torch.save(pruner1.bound_model.state_dict(), os.path.join(modelDir, 'model_masked1_finetune.pth'))
    # pruner1.bound_model.load_state_dict(torch.load(os.path.join(modelDir, 'model_masked1_finetune.pth')))
    # logger.info('Evaluation result (fine tuned): {} and {}'.format(best_acc,evaluator(pruner1.bound_model)))
    # logger.info('Fined tuned model saved to %s' % modelDir)
    # pruner1.check_masked_weight()


    # Pruner.compress() returns the masked model
    # but for AutoCompressPruner, Pruner.compress() returns directly the pruned model
    modelDir = os.path.join(modelDir,"stage_two")
    if not os.path.exists(modelDir):
        os.mkdir(modelDir)
    pruner2 = AdmmPruner(pruner1.bound_model,pattern_config=pattern_config,connectivity_config=connect_config,
                        trainer=trainer, num_iterations=args.iters, training_epochs=args.epochs,
                        row1=5e-4, row2=5e-4, base_algo='l2',logger = logger
                        )
    pruner2.compress_connect()
    #pruner2.compress_pattern()
    # visual changerate and loss
    pruner2.visiual_changerate(modelDir)
    pruner2.visiual_lossrate(modelDir)
    evaluation_result = evaluator(pruner2.bound_model)
    logger.info('Evaluation result (pattern admm model): %s' % evaluation_result)
    # do mask
    pruner2.wrapper_model()
    evaluation_result = evaluator(pruner2.bound_model)
    logger.info('Evaluation result (pattern masked model): %s' % evaluation_result) 
    # save model
    maskmodeldir = os.path.join(modelDir,"model_masked.pth")
    maskdir = os.path.join(modelDir,"mask.pth")
    pruner2.export_model(model_path=maskmodeldir,mask_path=maskdir)
    # finetune
    optimizer2,scheduler2 = get_optimizer_scheduler(args,pruner2.bound_model)
    best_acc = 0
    for epoch in range(args.fintune_epochs):
        train(args, pruner2.bound_model, args.device, trainLoader, criterion, optimizer2, epoch, logger=logger)
        scheduler2.step()
        acc = evaluator(pruner2.bound_model)
        if acc > best_acc:
            best_acc = acc
            torch.save(pruner2.bound_model.state_dict(), os.path.join(modelDir, 'model_masked_finetune.pth'))
    pruner2.bound_model.load_state_dict(torch.load(os.path.join(modelDir, 'model_masked_finetune.pth')))
    logger.info('Evaluation result (fine tuned): {} and {}'.format(best_acc,evaluator(pruner2.bound_model)))
    logger.info('Fined tuned model saved to %s' % modelDir)
    pruner2.check_masked_weight()



def train(args, model, device, train_loader, criterion, optimizer, epoch, logger, callback=None):
    model.train()
    loss_meter = AverageMeter()
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
        n = data.size(0)
        loss_meter.update(loss.data.item(), n)
        if batch_idx % args.print_freq == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return loss_meter.avg

def test(model, device, criterion, val_loader,logger):
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

    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset), 100. * accuracy))

    return accuracy





if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Eval time: %ds.', duration)

