import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class RPPModle(nn.Module):
    def __init__(self):
        super(RPPModle, self).__init__()

        self.pool1 = nn.Sequential(nn.AvgPool2d(kernel_size=(2, 2), stride=2),
                                   nn.Conv2d(48, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(),
                                   nn.Conv2d(48, 48, 3, groups=48, bias=False), nn.BatchNorm2d(48), nn.ReLU(),
                                   nn.Conv2d(48, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU()
                                   )

        self.pool2 = nn.Sequential(nn.AvgPool2d(kernel_size=(4, 4), stride=4),
                                   nn.Conv2d(48, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(),
                                   nn.Conv2d(48, 48, 3, groups=48, bias=False), nn.BatchNorm2d(48), nn.ReLU(),
                                   nn.Conv2d(48, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU()
                                   )

        self.pool3 = nn.Sequential(nn.AvgPool2d(kernel_size=(8, 8), stride=8),
                                   nn.Conv2d(48, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(),
                                   nn.Conv2d(48, 48, 3, groups=48, bias=False), nn.BatchNorm2d(48), nn.ReLU(),
                                   nn.Conv2d(48, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU()
                                   )

        self.pool4 = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                   nn.Conv2d(48, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU()
                                   )

        self.merge = nn.Sequential(nn.Conv2d(192, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU())
        self.merge2 = nn.Sequential(nn.Conv2d(96, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU())


    def forward(self, x):
        pool1 = F.interpolate(self.pool1(x), size=x.size()[2:], mode='bilinear', align_corners=True)
        pool2 = F.interpolate(self.pool2(x), size=x.size()[2:], mode='bilinear', align_corners=True)
        pool3 = F.interpolate(self.pool3(x), size=x.size()[2:], mode='bilinear', align_corners=True)
        pool4 = F.interpolate(self.pool4(x), size=x.size()[2:], mode='bilinear', align_corners=True)

        x = self.merge2(torch.cat((self.merge(torch.cat((pool1, pool2, pool3, pool4), 1)), x), 1))
        return x



class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()

        self.last_conv = nn.Sequential(nn.Conv2d(312, 256, kernel_size=3, stride=1, padding=2, bias=False,dilation=2),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       # nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False,dilation=1),
                                       BatchNorm(256),
                                       nn.ReLU())
        self.class_conv = nn.Sequential(nn.Conv2d(256, num_classes, kernel_size=1, stride=1))


        self.reduce_low_dsc = nn.Sequential(nn.Conv2d(256, 24, kernel_size=1, stride=1))

        self.beta_low = nn.Parameter(torch.tensor([[[1.0]]]*24),requires_grad=True)

        self._init_weight()





    def forward(self, dsc, low_level_feat, middle_level_feat,high_level_feat):


        low_dsc = self.reduce_low_dsc(dsc)
        low_dsc = F.interpolate(low_dsc, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)


        low_dis = torch.pow(low_dsc-low_level_feat,2)

        low_distance = torch.log(1+low_dis)

        local_gated_coeffeicent_low = low_distance

        low_level_feat = local_gated_coeffeicent_low * self.beta_low * low_level_feat


        x = F.interpolate(high_level_feat, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        middle_level_feat = F.interpolate(middle_level_feat, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        final_feat = torch.cat((x,middle_level_feat, low_level_feat), dim=1)


        x = self.last_conv(final_feat)
        x = self.class_conv(x)


        return x


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)




class Decoder1(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder1, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       # nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU())
        self.class_conv = nn.Sequential(nn.Conv2d(256, num_classes, kernel_size=1, stride=1))


        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        final_feat = torch.cat((x, low_level_feat), dim=1)

        x = self.last_conv(final_feat)
        x = self.class_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



def build_decoder1(num_classes, backbone, BatchNorm):
    return Decoder1(num_classes, backbone, BatchNorm)

class sub_decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(sub_decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()

        self.down = nn.Sequential(
            nn.Conv2d(304, 64, 1, bias=False), nn.BatchNorm2d(64)
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, dilation=2, padding=2, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 1, bias=False), nn.BatchNorm2d(64)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, dilation=3, padding=3, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 1, bias=False), nn.BatchNorm2d(64)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, dilation=4, padding=4, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 1, bias=False), nn.BatchNorm2d(64)
        )

        self.up = nn.Sequential(
            nn.Conv2d(64, 128, 1, bias=False), nn.BatchNorm2d(128)
        )

        self.last_conv = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       # nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       # nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)

        x = self.down(x)
        block1 = F.relu(self.block1(x) + x, True)
        block2 = F.relu(self.block2(block1) + block1, True)
        block3 = F.relu(self.block3(block2) + block2, True)

        x = self.last_conv(self.up(block3))

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

