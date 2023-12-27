import argparse
import os
import time

# PyTorch includes
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader

# Custom includes
from util.utils import *
from src.sod_layer11 import Deeplabv3plus

# Dataloaders includes
import util.custom_transforms as trforms
from util.data_loader import SalObjDataset


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-gpu', type=str, default='cuda:0')

    ## Model settings
    parser.add_argument('-criterion', type=str, default='BCE')  # cross_entropy
    parser.add_argument('-input_size', type=int, default=352)
    parser.add_argument('-output_stride', type=int, default=16)

    ## Train settings
    parser.add_argument('-batch_size', type=int, default=10)
    parser.add_argument('-nepochs', type=int, default=40)
    parser.add_argument('-resume_epoch', type=int, default=0)
    parser.add_argument('-load_pretrain', type=str, default=None)
    parser.add_argument('-save_tar', type=str, default="")

    ## Optimizer settings
    parser.add_argument('-lr', type=float, default=2e-8)
    parser.add_argument('-update_lr_every', type=int, default=10)

    ## Visualization settings
    parser.add_argument('-save_every', type=int, default=5)
    parser.add_argument('-log_every', type=int, default=300)
    parser.add_argument('-use_test', type=int, default=1)
    return parser.parse_args()


def main(args):
    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    
    save_dir = os.path.join(save_dir_root, 'runs', args.save_tar)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    net = Deeplabv3plus(os=args.output_stride)

    if args.load_pretrain is not None:
        net.load_state_dict(torch.load(args.load_pretrain))

    if args.resume_epoch !=0 :
        load_path = os.path.join(save_dir, 'model_epoch_' + str(args.resume_epoch - 1) + '.pth')
        print('Initializing weights from: {}...'.format(load_path))
        net.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage))
    
    device_ids = []
    if len(device_ids) > 0:
        net = nn.DataParallel(net, device_ids=device_ids)
    net.to(args.gpu)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    if args.criterion == 'BCE':
        criterion = BCE_2d
    elif args.criterion == 'cross_entropy':
        criterion = cross_entropy2d
    else:
        raise NotImplementedError

    composed_transforms_tr = transforms.Compose([
        trforms.FixedResize(size=(args.input_size+50, args.input_size+50)),
        trforms.RandomCrop(size=(args.input_size, args.input_size)),
        trforms.RandomHorizontalFlip(),
        trforms.Normalize(mean=(0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()
    ])

    composed_transforms_ts = transforms.Compose([
        trforms.FixedResize(size=(args.input_size, args.input_size)),
        trforms.Normalize(mean=(0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()
    ])

    train_data = SalObjDataset(image_root="datasets/refined/image/",
                               gt_root="datasets/refined/mask",
								transform=composed_transforms_tr)
    
    val_data = SalObjDataset(image_root="datasets/DUTS/DUTS-TE/DUTS-TE-Image/",
                             gt_root="datasets/DUTS/DUTS-TE/DUTS-TE-Mask/",
                             transform=composed_transforms_ts)

    trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8)
    testloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=8)

    num_iter_tr = len(trainloader)
    num_iter_ts = len(testloader)
    
    nitrs = args.resume_epoch * num_iter_tr
    nsamples = args.resume_epoch * len(train_data)
    print('nitrs: %d num_iter_tr: %d' % (nitrs, num_iter_tr))
    print('nsamples: %d tot_num_samples: %d' % (nsamples, len(train_data)))

    recent_losses = []
    start_t = time.time()

    size_rates = [0.75, 1, 1.25]
    best_f, cur_f = 0.0, 0.0
    for epoch in range(args.resume_epoch, args.nepochs):

        net.train()
        epoch_losses = []
        for step, sample_batched in enumerate(trainloader):
            for rate in size_rates:
                optimizer.zero_grad()

                inputs, labels = sample_batched['image'], sample_batched['label']
                inputs, labels = Variable(inputs, requires_grad=True).to(args.gpu), Variable(labels).to(args.gpu)

                trainsize = int(round(args.input_size*rate/32)*32)
                if rate != 1:
                    inputs = F.interpolate(inputs, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    labels = F.interpolate(labels, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

                sal_pred = net.forward(inputs)
                loss = criterion(sal_pred, labels, size_average=False, batch_average=True)
                trainloss = loss.item()
                epoch_losses.append(trainloss)
                
                if len(recent_losses) < args.log_every:
                    recent_losses.append(trainloss)
                else:
                    recent_losses[nitrs % len(recent_losses)] = trainloss

                # Backward the averaged gradient
                loss.backward()
                optimizer.step()
                
                nitrs += 1
                nsamples += args.batch_size

            if nitrs % args.log_every == 0:
                now = str(time.strftime("%Y-%m-%d %H:%M:%S"))
                meanloss = sum(recent_losses) / len(recent_losses)

                print('%s epoch: %d, step: %d, trainloss: %.2f, timecost:%.2f secs' %
                    (now, epoch, step, meanloss, time.time() - start_t))

        meanloss = sum(epoch_losses) / len(epoch_losses)
        print('epoch: %d meanloss: %.2f' % (epoch, meanloss))

        if args.use_test == 1:
            cnt = 0     
            total_mae = 0.0
            sum_testloss = 0.0
            prec_lists = []
            recall_lists = []
            net.eval()
            for ii, sample_batched in enumerate(testloader):
                inputs, labels = sample_batched['image'], sample_batched['label']

                # Forward pass of the mini-batch
                inputs, labels = Variable(inputs,requires_grad=True), Variable(labels)
                
                inputs, labels = inputs.to(args.gpu), labels.to(args.gpu)

                with torch.no_grad():
                    outputs = net.forward(inputs)

                loss = criterion(outputs, labels, size_average=False, batch_average=True)
                sum_testloss += loss.item()

                # predictions = torch.max(outputs, 1)[1]
                predictions = torch.tensor([1])
                if args.criterion == 'cross_entropy':
                    tmp = torch.nn.Softmax2d()(outputs)
                    predictions = tmp.narrow(1, 1, 1)
                if args.criterion == 'BCE':
                    predictions = torch.nn.Sigmoid()(outputs)

                total_mae += get_mae(predictions, labels) * predictions.size(0)
                prec_list, recall_list = get_prec_recall(predictions, labels)
                prec_lists.extend(prec_list)
                recall_lists.extend(recall_list)
                cnt += predictions.size(0)

                if ii % num_iter_ts == num_iter_ts - 1:
                    mmae = total_mae / cnt
                    mean_testloss = sum_testloss / num_iter_ts
                    mean_prec = sum(prec_lists) / len(prec_lists)
                    mean_recall = sum(recall_lists) / len(recall_lists)
                    fbeta = 1.3 * mean_prec * mean_recall / (0.3 * mean_prec +
                                                             mean_recall)
                    print('Validation:epoch: %d, numImages: %d testloss: %.2f mmae: %.4f fbeta: %.4f'
                        % (epoch, cnt, mean_testloss, mmae, fbeta))
                   
                    cur_f = fbeta
                    if cur_f > best_f:
                        save_path = os.path.join(save_dir, 'model_best.pth')
                        torch.save(net.state_dict(), save_path)
                        print("Save the best model at {} epoch\n".format(str(epoch)))
                        best_f = cur_f
        # If you want to start training with any epoch, turn it on
        # if epoch % args.save_every == args.save_every - 1:
        #     save_path = os.path.join(save_dir, 'model_epoch_' + str(epoch) + '.pth')
        #     torch.save(net.state_dict(), save_path)
        #     print("Save model at {}\n".format(save_path))

        if epoch % args.update_lr_every == args.update_lr_every - 1:
            lr_ = lr_poly(args.lr, epoch, args.nepochs, 0.9)
            print('(poly lr policy) learning rate: ', lr_)
            optimizer = optim.SGD(net.parameters(),lr=lr_, momentum=0.9,weight_decay=5e-4)


if __name__ == '__main__':
    args = get_arguments()
    main(args)
