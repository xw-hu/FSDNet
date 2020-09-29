import datetime
import os

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

import joint_transforms
from nets import FSDNet, basic, basic_DSC
from config import train_cuhkshadow_path, val_cuhkshadow_path
from dataset import ImageFolder
from misc import AvgMeter, check_mkdir
from models.sync_batchnorm.replicate import patch_replication_callback
from models.deeplab import *
from KL_divergence import KL_divergence

# torch.cuda.set_device(0)

cudnn.benchmark = True

ckpt_path = './ckpt'
exp_name = 'FSDNet'

args = {
    'iter_num': 50000,
    'train_batch_size': 6,
    'last_iter': 0,
    'lr': 0.005,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,

    'momentum': 0.9,
    'nesterov': False,
    'resume_snapshot': '',
    'val_freq': 10000,
    'img_size_h': 512,
	'img_size_w': 512,
	'crop_size': 512,
    'snapshot_epochs': 10000,
    'backbone': 'mobilenet', # 'resnet', 'xception', 'drn', 'mobilenet'],
    'out_stride': 16, # 8 or 16
    'sync_bn': None, # whether to use sync bn (default: auto)
    'freeze_bn': False,
    'pre_train': True

}

transform = transforms.Compose([
    transforms.ToTensor()
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_pil = transforms.ToPILImage()

joint_transform = joint_transforms.Compose([
    joint_transforms.Resize((args['img_size_h'], args['img_size_w'])),
    #joint_transforms.RandomCrop(args['crop_size']),
    joint_transforms.RandomHorizontallyFlip()

])


joint_transform_val = joint_transforms.Compose([
    joint_transforms.Resize((args['img_size_h'], args['img_size_w'])),
])


train_set = ImageFolder(train_cuhkshadow_path, transform=transform, target_transform=transform, joint_transform=joint_transform, is_train=True, batch_size=args['train_batch_size'])
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=8, shuffle=True)

test1_set = ImageFolder(val_cuhkshadow_path, transform=transform, target_transform=transform, joint_transform=joint_transform_val, is_train=False, batch_size=args['train_batch_size'])
test1_loader = DataLoader(test1_set, batch_size=args['train_batch_size'], num_workers=8)


criterion = nn.L1Loss()
criterion_depth = nn.L1Loss()
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')
log_path_val = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '_val.txt')

criterion_distilation = nn.MSELoss()


def main():

    # Define network
    net = FSDNet(num_classes=1,
                       backbone=args['backbone'],
                       output_stride=args['out_stride'],
                       sync_bn=args['sync_bn'],
                       freeze_bn=args['freeze_bn'])

    net.cuda().train()

    # if args['pre_train']:
    #     print("load the model trained on all data.")
    #     net.load_state_dict(torch.load(os.path.join(ckpt_path, 'shadow_cuhk', '50000_9.54' + '.pth')))
	#
    #     train_params = [{'params': net.get_1x_lr_params(), 'lr': args['lr']},
    #                     {'params': net.get_10x_lr_params(), 'lr': args['lr']}]
    # else:

    train_params = [{'params': net.get_1x_lr_params(), 'lr': args['lr']},
                    {'params': net.get_10x_lr_params(), 'lr': args['lr'] * 10}]

    # Define Optimizer
    optimizer = torch.optim.SGD(train_params, momentum=args['momentum'],
                                weight_decay=args['weight_decay'], nesterov=args['nesterov'])


    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    open(log_path_val, 'w').write(str(args) + '\n\n')
    train(net, optimizer)


def train(net, optimizer):

    curr_iter = args['last_iter']
    train_loss_record = AvgMeter()
    train_net_loss_record = AvgMeter()
    train_uncertainty_loss_record = AvgMeter()


    while True:
        for i, data in enumerate(train_loader):

            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                            ) ** args['lr_decay']

            inputs, gts = data

            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            gts = Variable(gts).cuda()

            optimizer.zero_grad()

            result = net(inputs)

            nhw = inputs.size(0)*inputs.size(2)*inputs.size(3)

            loss_net = criterion(result, gts)
            loss = loss_net

            loss.backward()

            optimizer.step()

            #print(uncertainty)

            train_loss_record.update(loss.data, batch_size)
            train_net_loss_record.update(loss_net.data, batch_size)
            #train_uncertainty_loss_record.update(loss_uncertainty_regular.data, batch_size)


            curr_iter += 1

            # log = '[iter %d], [train loss %.5f], [lr %.8f], [loss_net %.5f], [w1 %.8f], [loss_uncertainty %.5f]' % \
            #       (curr_iter, train_loss_record.avg, optimizer.param_groups[1]['lr'],
            #        train_net_loss_record.avg, w1, train_uncertainty_loss_record.avg)

            log = '[iter %d], [train loss %.5f], [lr %.8f], [loss_net %.5f]' % \
                  (curr_iter, train_loss_record.avg, optimizer.param_groups[1]['lr'],
                   train_net_loss_record.avg)
            print(log)
            open(log_path, 'a').write(log + '\n')

            if (curr_iter + 1) % args['val_freq'] == 0:
                validate(net, curr_iter, optimizer)

            if (curr_iter + 1) % args['snapshot_epochs'] == 0:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, ('%d.pth' % (curr_iter + 1) )))
                torch.save(optimizer.state_dict(), os.path.join(ckpt_path, exp_name, ('%d_optim.pth' % (curr_iter + 1) )))

            if curr_iter > args['iter_num']:
                return


def validate(net, curr_iter, optimizer):
    print('validating...')
    net.eval()

    loss_record1 = AvgMeter()
    iter_num1 = len(test1_loader)

    with torch.no_grad():
        for i, data in enumerate(test1_loader):
            inputs, gts = data
            inputs = Variable(inputs).cuda()
            gts = Variable(gts).cuda()

            res = net(inputs)

            loss = criterion(res, gts)
            loss_record1.update(loss.data, inputs.size(0))

            print('processed test1 %d / %d' % (i + 1, iter_num1))


    # snapshot_name = 'iter_%d_loss1_%.5f_lr_%.6f' % (curr_iter + 1, loss_record1.avg,
    #                                                            optimizer.param_groups[1]['lr'])

    log_val = '[validate]: [iter %d], [loss1 %.5f]' % (curr_iter + 1, loss_record1.avg)
    print(log_val)
    open(log_path_val, 'a').write(log_val + '\n')

    # torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + '.pth'))
    # torch.save(optimizer.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + '_optim.pth'))

    net.train()


if __name__ == '__main__':
    main()
