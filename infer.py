import os
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from nets import FSDNet
from misc import check_mkdir
from models.deeplab import *


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.manual_seed(2019)
torch.cuda.set_device(0)

ckpt_path = './ckpt'
exp_name = 'FSDNet'
args = {
    'snapshot': '50000',
    'backbone': 'mobilenet',  # 'resnet', 'xception', 'drn', 'mobilenet'],
    'out_stride': 16,  # 8 or 16
    'sync_bn': None,  # whether to use sync bn (default: auto)
    'freeze_bn': False
}

transform = transforms.Compose([
    transforms.Resize([512,512]),
    transforms.ToTensor() ])


to_pil = transforms.ToPILImage()

to_test = {'CUHKShadow': '../../dataset/CUHKshadow'}


def main():

    net = FSDNet(num_classes=1,
                     backbone=args['backbone'],
                     output_stride=args['out_stride'],
                     sync_bn=args['sync_bn'],
                     freeze_bn=args['freeze_bn']).cuda()

    if len(args['snapshot']) > 0:
        print('load snapshot \'%s\' for testing' % args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'),
                                       map_location=lambda storage, loc: storage.cuda(0)))

    net.eval()
    total_time = 0
    with torch.no_grad():
        for name, root in to_test.items():

            img_txt = open(os.path.join(root, 'val.txt'))
            img_name = []

            for img_list in img_txt:
                x = img_list.split()
                img_name.append(os.path.join(root, x[0]))

            for idx, image_name in enumerate(img_name):
                #print('predicting for %s: %d / %d' % (name, idx + 1, len(img_name)))

                check_mkdir(
                    os.path.join(ckpt_path, exp_name, '(%s) %s_prediction_%s' % (exp_name, name, args['snapshot'])))

                img = Image.open(os.path.join(root, image_name))
                #img = Image.open('/home/xwhu/1.jpg')
                w, h = img.size
                img_var = Variable(transform(img).unsqueeze(0)).cuda()

                start_time = time.time()

                #res, res0, res1, res2, res3, res4 = net(img_var)
                res = net(img_var)

                torch.cuda.synchronize()

                total_time = total_time + time.time() - start_time

                print('predicting: %d / %d, avg_time: %.5f' % (idx + 1, len(img_name), total_time / (idx + 1)))

                result = transforms.Resize((h, w))(to_pil(res.data.squeeze(0).cpu()))



                sub_name = image_name.split('/')

                check_mkdir(
                    os.path.join(ckpt_path, exp_name, '(%s) %s_prediction_%s' % (exp_name, name, args['snapshot']),
                                 sub_name[-3]))

                check_mkdir(
                    os.path.join(ckpt_path, exp_name, '(%s) %s_prediction_%s' % (exp_name, name, args['snapshot']),
                                 sub_name[-3], sub_name[-2]))

                #result.save('/home/xwhu/2.jpg')

                result.save(
                    os.path.join(ckpt_path, exp_name, '(%s) %s_prediction_%s' % (
                        exp_name, name, args['snapshot']), sub_name[-3], sub_name[-2], sub_name[-1]))




if __name__ == '__main__':
    main()
