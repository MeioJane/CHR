import argparse

import torch
import torch.nn as nn

from CHR.engine import MultiLabelMAPEngine
from CHR.models import resnet101_CHR
from CHR.ray import XrayClassification
from torch.nn.modules.loss import _WeightedLoss

parser = argparse.ArgumentParser(description='CHR Training')
parser.add_argument('--data', metavar='DIR',default='./dataset/',
                    help='path to dataset (e.g. ../data/')
parser.add_argument('--image-size', '-i', default=224, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=15, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--k', default=1, type=float,
                    metavar='N', help='number of regions (default: 1)')
parser.add_argument('--alpha', default=1, type=float,
                    metavar='N', help='weight for the min regions (default: 1)')
parser.add_argument('--maps', default=1, type=int,
                    metavar='N', help='number of maps per class (default: 1)')

def binary_cross_entropy(input, target, eps=1e-10):
    '''if not (target.size() == input.size()):
        warnings.warn("Using a target size ({}) that is different to the input size ({}) is deprecated. "
                      "Please ensure they have the same size.".format(target.size(), input.size()))
    if input.nelement() != target.nelement():
        raise ValueError("Target and input must have the same number of elements. target nelement ({}) "
                         "!= input nelement ({})".format(target.nelement(), input.nelement()))

    if weight is not None:
        new_size = _infer_size(target.size(), weight.size())
        weight = weight.expand(new_size)
        if torch.is_tensor(weight):
            weight = Variable(weight)'''
    input=torch.sigmoid(input)
    return -(target*torch.log(input+eps)+(1-target)*torch.log(1-input+eps))



class MultiLabelSoftMarginLoss(_WeightedLoss):

    def forward(self, input, target):
        return binary_cross_entropy(input, target)




def main_ray():
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    args.data='/DATA/disk1/mcj/dataset/'
    args.resume = './CHR/models-/checkpoint.pth.tar'



    use_gpu = torch.cuda.is_available()

    # define dataset
    train_dataset = XrayClassification(args.data, 'train')
    val_dataset = XrayClassification(args.data, 'test_new')
    num_classes = 5

    # load model
    model = resnet101_CHR(num_classes, pretrained=True)

    # define loss function (criterion)
    criterion = MultiLabelSoftMarginLoss()

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
             'evaluate': args.evaluate, 'resume': args.resume}
    state['difficult_examples'] = True
    state['save_model_path'] = './CHR/models-'
    state['epoch_step']={20}

    engine = MultiLabelMAPEngine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)


if __name__ == '__main__':
    main_ray()
