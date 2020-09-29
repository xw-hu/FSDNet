import torch
from torch import nn
import torch.nn.functional as F
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.aspp import build_aspp
from models.decoder import build_decoder
from torch.autograd import Variable
from models.backbone import build_backbone
from irnn import irnn

#from torchsummary import summary


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class Spacial_IRNN(nn.Module):
    def __init__(self, in_channels, alpha=1.0):
        super(Spacial_IRNN, self).__init__()
        self.left_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.right_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.up_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.down_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.left_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]] * in_channels))
        self.right_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]] * in_channels))
        self.up_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]] * in_channels))
        self.down_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]] * in_channels))

    def forward(self, input):
        return irnn()(input, self.up_weight.weight, self.right_weight.weight, self.down_weight.weight,
                      self.left_weight.weight, self.up_weight.bias, self.right_weight.bias, self.down_weight.bias,
                      self.left_weight.bias)


class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.out_channels = int(in_channels / 2)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(self.out_channels, 4, kernel_size=1, padding=0, stride=1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.sigmod(out)
        return out


class DSC_Module(nn.Module):
    def __init__(self, in_channels, out_channels, attention=1, alpha=1.0):
        super(DSC_Module, self).__init__()
        self.out_channels = out_channels
        self.irnn1 = Spacial_IRNN(self.out_channels, alpha)
        self.irnn2 = Spacial_IRNN(self.out_channels, alpha)
        self.conv_in = conv1x1(in_channels, in_channels)
        self.conv2 = conv1x1(in_channels * 4, in_channels)
        self.conv3 = conv1x1(in_channels * 4, in_channels)
        self.relu2 = nn.ReLU(True)
        self.attention = attention
        if self.attention:
            self.attention_layer = Attention(in_channels)

    def forward(self, x):
        if self.attention:
            weight = self.attention_layer(x)
        out = self.conv_in(x)
        top_up, top_right, top_down, top_left = self.irnn1(out)

        # direction attention
        if self.attention:
            top_up.mul(weight[:, 0:1, :, :])
            top_right.mul(weight[:, 1:2, :, :])
            top_down.mul(weight[:, 2:3, :, :])
            top_left.mul(weight[:, 3:4, :, :])
        out = torch.cat([top_up, top_right, top_down, top_left], dim=1)
        out = self.conv2(out)
        top_up, top_right, top_down, top_left = self.irnn2(out)

        # direction attention
        if self.attention:
            top_up.mul(weight[:, 0:1, :, :])
            top_right.mul(weight[:, 1:2, :, :])
            top_down.mul(weight[:, 2:3, :, :])
            top_left.mul(weight[:, 3:4, :, :])

        out = torch.cat([top_up, top_right, top_down, top_left], dim=1)
        out = self.conv3(out)
        out = self.relu2(out)

        return out

class LayerConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, relu):
        super(LayerConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)

        return x



class ShadowNetUncertaintyGuide(nn.Module):
    def __init__(self, backbone='mobilenet', output_stride=8, num_classes=1,
                 sync_bn=True, freeze_bn=False):
        super(ShadowNetUncertaintyGuide, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)

        self.temp_predict = nn.Sequential(nn.Conv2d(320, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
                                       )
        self.temp_uncertainty = nn.Sequential(nn.Conv2d(320, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                          BatchNorm(256),
                                          nn.ReLU(),
                                          nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                          BatchNorm(256),
                                          nn.ReLU(),
                                          nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
                                          )


        self.aspp = build_aspp(backbone, output_stride, BatchNorm)

        self.reduce1 = LayerConv(320, 256, 1, 1, 0, False)

        self.dsc = DSC_Module(256, 256)

        self.reduce2 = LayerConv(512, 256, 1, 1, 0, False)

        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        # self.last_conv = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
        #                                BatchNorm(256),
        #                                nn.ReLU(),
        #                                # nn.Dropout(0.5),
        #                                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
        #                                BatchNorm(256),
        #                                nn.ReLU(),
        #                                nn.Conv2d(256, num_classes, kernel_size=1, stride=1))


        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        high_level_feat, low_level_feat = self.backbone(input)


        temp_predict = self.temp_predict(high_level_feat.detach())
        temp_uncertatinty = self.temp_uncertainty(high_level_feat.detach())


        x = self.reduce1(high_level_feat)
        dsc = self.dsc(x)
        x = self.reduce2(torch.cat((self.aspp(high_level_feat), dsc), 1)) * (1+F.sigmoid(temp_uncertatinty).detach())
        x, u = self.decoder(x, low_level_feat)

        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        x = F.sigmoid(x)

        u = F.interpolate(u, size=input.size()[2:], mode='bilinear', align_corners=True)
        u = F.sigmoid(u)

        temp_predict = F.interpolate(temp_predict, size=input.size()[2:], mode='bilinear', align_corners=True)
        temp_predict = F.sigmoid(temp_predict)
        temp_uncertatinty = F.sigmoid(F.interpolate(temp_uncertatinty, size=input.size()[2:], mode='bilinear', align_corners=True))

        if self.training:
            return x, temp_predict, temp_uncertatinty, u
        else:
            return x, u

        # u = F.interpolate(u, size=input.size()[2:], mode='bilinear', align_corners=True)
        # u = F.sigmoid(u)


        # #### consine similarity
		#
        # u_down_feat = F.max_pool2d(u_feat, kernel_size=4, stride=4)
        # n, c, h, w = u_down_feat.size()
		#
        # ### relation map
        # # [n, hw, c]
        # theta = u_down_feat.view(n, c, -1).transpose(1, 2)
        # # [n, c, hw]
        # phi = u_down_feat.view(n, c, -1)
		#
        # norm = torch.sqrt(torch.sum(u_down_feat*u_down_feat, 1)).view(n, 1, -1)
		#
        # # [n, hw, hw]
        # R = torch.bmm(theta, phi) / torch.bmm(norm.transpose(1, 2), norm)



    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.reduce1, self.reduce2, self.dsc, self.decoder, self.temp_uncertainty, self.temp_predict]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p




class basic_ASPP_DSC(nn.Module):
    def __init__(self, backbone='mobilenet', output_stride=8, num_classes=1,
                 sync_bn=True, freeze_bn=False):
        super(basic_ASPP_DSC, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)

        self.aspp = build_aspp(backbone, output_stride, BatchNorm)

        self.reduce1 = LayerConv(320, 256, 1, 1, 0, False)

        self.dsc = DSC_Module(256, 256)

        self.reduce2 = LayerConv(512, 256, 1, 1, 0, False)


        self.last_conv = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       # nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))


        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        high_level_feat = self.backbone(input)

        x = self.reduce1(high_level_feat)
        dsc = self.dsc(x)
        x = self.reduce2(torch.cat((self.aspp(high_level_feat), dsc), 1))

        #x = self.aspp(high_level_feat)

        x = self.last_conv(x)



        #x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        x = F.sigmoid(x)

        if self.training:
            return x
        else:
            return x

        # u = F.interpolate(u, size=input.size()[2:], mode='bilinear', align_corners=True)
        # u = F.sigmoid(u)


        # #### consine similarity
		#
        # u_down_feat = F.max_pool2d(u_feat, kernel_size=4, stride=4)
        # n, c, h, w = u_down_feat.size()
		#
        # ### relation map
        # # [n, hw, c]
        # theta = u_down_feat.view(n, c, -1).transpose(1, 2)
        # # [n, c, hw]
        # phi = u_down_feat.view(n, c, -1)
		#
        # norm = torch.sqrt(torch.sum(u_down_feat*u_down_feat, 1)).view(n, 1, -1)
		#
        # # [n, hw, hw]
        # R = torch.bmm(theta, phi) / torch.bmm(norm.transpose(1, 2), norm)



    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.last_conv, self.dsc, self.aspp, self.reduce1, self.reduce2]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p




class basic_ASPP(nn.Module):
    def __init__(self, backbone='mobilenet', output_stride=8, num_classes=1,
                 sync_bn=True, freeze_bn=False):
        super(basic_ASPP, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)

        self.aspp = build_aspp(backbone, output_stride, BatchNorm)


        self.last_conv = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       # nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))


        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        high_level_feat = self.backbone(input)

        # x = self.reduce1(high_level_feat)
        # dsc = self.dsc(x)
        # x = self.reduce2(torch.cat((self.aspp(high_level_feat), dsc), 1))

        x = self.aspp(high_level_feat)

        x = self.last_conv(x)



        #x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        x = F.sigmoid(x)

        if self.training:
            return x
        else:
            return x

        # u = F.interpolate(u, size=input.size()[2:], mode='bilinear', align_corners=True)
        # u = F.sigmoid(u)


        # #### consine similarity
		#
        # u_down_feat = F.max_pool2d(u_feat, kernel_size=4, stride=4)
        # n, c, h, w = u_down_feat.size()
		#
        # ### relation map
        # # [n, hw, c]
        # theta = u_down_feat.view(n, c, -1).transpose(1, 2)
        # # [n, c, hw]
        # phi = u_down_feat.view(n, c, -1)
		#
        # norm = torch.sqrt(torch.sum(u_down_feat*u_down_feat, 1)).view(n, 1, -1)
		#
        # # [n, hw, hw]
        # R = torch.bmm(theta, phi) / torch.bmm(norm.transpose(1, 2), norm)



    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.last_conv, self.aspp]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p



class FSDNet(nn.Module):
    def __init__(self, backbone='mobilenet', output_stride=8, num_classes=1,
                 sync_bn=True, freeze_bn=False):
        super(FSDNet, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)

        self.reduce1 = LayerConv(320, 256, 1, 1, 0, False)

        self.dsc = DSC_Module(256, 256)

        self.reduce2 = LayerConv(576, 256, 1, 1, 0, False)

        self.decoder = build_decoder(num_classes, backbone, BatchNorm)


        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        high_level_feat, low_level_feat, middle_level_feat = self.backbone(input)

        x = self.reduce1(high_level_feat)
        dsc = self.dsc(x)
        x = self.reduce2(torch.cat((high_level_feat, dsc), 1))

        x = self.decoder(dsc, low_level_feat, middle_level_feat,x) # 256,256,256,256
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        x = torch.sigmoid(x)

        if self.training:
            return x
        else:
            return x


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.reduce1, self.reduce2, self.dsc, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

class basic_DSC(nn.Module):
    def __init__(self, backbone='mobilenet', output_stride=8, num_classes=1,
                 sync_bn=True, freeze_bn=False):
        super(basic_DSC, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)

        self.reduce1 = LayerConv(320, 256, 1, 1, 0, False)

        self.dsc = DSC_Module(256, 256)

        self.reduce2 = LayerConv(576, 256, 1, 1, 0, False)

        self.last_conv = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       # nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))


        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        high_level_feat = self.backbone(input)

        x = self.reduce1(high_level_feat)
        dsc = self.dsc(x)
        x = self.reduce2(torch.cat((high_level_feat, dsc), 1))

        x = self.last_conv(x)

        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        x = F.sigmoid(x)

        if self.training:
            return x
        else:
            return x


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.last_conv, self.reduce1, self.reduce2, self.dsc]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p



class basic(nn.Module):
    def __init__(self, backbone='mobilenet', output_stride=8, num_classes=1,
                 sync_bn=True, freeze_bn=False):
        super(basic, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)


        self.last_conv = nn.Sequential(nn.Conv2d(320, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       # nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))


        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        high_level_feat = self.backbone(input)


        x = self.last_conv(high_level_feat)

        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        x = F.sigmoid(x)

        if self.training:
            return x
        else:
            return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.last_conv]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


class ShadowNet(nn.Module):
    def __init__(self, backbone='mobilenet', output_stride=8, num_classes=1,
                 sync_bn=True, freeze_bn=False):
        super(ShadowNet, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)


        self.reduce1 = LayerConv(320, 256, 1, 1, 0, False)

        self.dsc = DSC_Module(256, 256)

        self.reduce2 = LayerConv(512, 256, 1, 1, 0, False)

        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        # self.last_conv = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
        #                                BatchNorm(256),
        #                                nn.ReLU(),
        #                                # nn.Dropout(0.5),
        #                                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
        #                                BatchNorm(256),
        #                                nn.ReLU(),
        #                                nn.Conv2d(256, num_classes, kernel_size=1, stride=1))


        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        high_level_feat, low_level_feat = self.backbone(input)

        x = self.reduce1(high_level_feat)
        dsc = self.dsc(x)
        x = self.reduce2(torch.cat((self.aspp(high_level_feat), dsc), 1))

        #x = self.last_conv(x)


        #x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        x = F.sigmoid(x)

        if self.training:
            return x
        else:
            return x

        # u = F.interpolate(u, size=input.size()[2:], mode='bilinear', align_corners=True)
        # u = F.sigmoid(u)


        # #### consine similarity
		#
        # u_down_feat = F.max_pool2d(u_feat, kernel_size=4, stride=4)
        # n, c, h, w = u_down_feat.size()
		#
        # ### relation map
        # # [n, hw, c]
        # theta = u_down_feat.view(n, c, -1).transpose(1, 2)
        # # [n, c, hw]
        # phi = u_down_feat.view(n, c, -1)
		#
        # norm = torch.sqrt(torch.sum(u_down_feat*u_down_feat, 1)).view(n, 1, -1)
		#
        # # [n, hw, hw]
        # R = torch.bmm(theta, phi) / torch.bmm(norm.transpose(1, 2), norm)



    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.reduce1, self.reduce2, self.dsc, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p




class ShadowNet2(nn.Module):
    def __init__(self, backbone='mobilenet', output_stride=8, num_classes=1,
                 sync_bn=True, freeze_bn=False):
        super(ShadowNet2, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)


        self.reduce1 = LayerConv(320, 256, 1, 1, 0, False)

        self.dsc = DSC_Module(256, 256)

        self.reduce2 = LayerConv(512, 256, 1, 1, 0, False)

        self.decoder = build_decoder(num_classes, backbone, BatchNorm)


        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        high_level_feat, low_level_feat, middle_level_feat = self.backbone(input)

        x = self.reduce1(high_level_feat)
        dsc = self.dsc(x)
        x = self.reduce2(torch.cat((self.aspp(high_level_feat), dsc), 1))

        #x = self.last_conv(x)


        #x = self.aspp(x)
        x = self.decoder(x, low_level_feat, middle_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        x = F.sigmoid(x)

        if self.training:
            return x
        else:
            return x

        # u = F.interpolate(u, size=input.size()[2:], mode='bilinear', align_corners=True)
        # u = F.sigmoid(u)


        # #### consine similarity
		#
        # u_down_feat = F.max_pool2d(u_feat, kernel_size=4, stride=4)
        # n, c, h, w = u_down_feat.size()
		#
        # ### relation map
        # # [n, hw, c]
        # theta = u_down_feat.view(n, c, -1).transpose(1, 2)
        # # [n, c, hw]
        # phi = u_down_feat.view(n, c, -1)
		#
        # norm = torch.sqrt(torch.sum(u_down_feat*u_down_feat, 1)).view(n, 1, -1)
		#
        # # [n, hw, hw]
        # R = torch.bmm(theta, phi) / torch.bmm(norm.transpose(1, 2), norm)



    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.reduce1, self.reduce2, self.dsc, self.decoder, self.aspp]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    model = ShadowNet3(backbone='mobilenet', output_stride=16)
    model.cuda().eval()
    input = torch.rand(1, 3, 512, 512)
    input = Variable(input).cuda()
    output  = model(input)
    print(output.size())


# device = torch.device("cuda:0")
#
# dp = depth_predciton().to(device)
# summary(dp, (3,512,1024))
#
# gt_net = generator().to(device)
# summary(gt_net, (3,512,1024))

