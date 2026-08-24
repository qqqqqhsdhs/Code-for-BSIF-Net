# coding:utf-8

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.models as models

class ConvBNReLU(nn.Sequential):
    """Convolution, batch normalization, and LeakyReLU."""
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilation=1, groups=1, bn=True, relu=True):
        padding = ((kernel_size - 1) * dilation + 1) // 2
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                              bias=False if bn else True)
        self.bn = bn
        if bn:
            self.bnop = nn.BatchNorm2d(out_planes)
        self.relu = relu
        if relu:
            self.reluop = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bnop(x)
        if self.relu:
            x = self.reluop(x)
        return x

class Atttion_avg_pool(nn.Module):
    def __init__(self, dim, reduction):
        super(Atttion_avg_pool, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.GELU(),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SIF(nn.Module):
    def __init__(self, C):
        super().__init__()

        self.rel_r = nn.Conv2d(C, 1, kernel_size=1, bias=True)
        self.rel_t = nn.Conv2d(C, 1, kernel_size=1, bias=True)

        self.dis = nn.Sequential(
            ConvBNReLU(2*C, C),
            nn.Conv2d(C, 1, kernel_size=3, padding=1)
        )

        self.phi_r = ConvBNReLU(C, C)
        self.phi_t = ConvBNReLU(C, C)

        self.alpha1 = nn.Parameter(torch.tensor(0.5))
        self.alpha2 = nn.Parameter(torch.tensor(0.5))

    def forward(self, Fr, Ft, return_d2f_activations=False):
        rr = torch.sigmoid(self.rel_r(Fr))
        rt = torch.sigmoid(self.rel_t(Ft))

        diff = torch.abs(Fr - Ft)
        prod = torch.tanh(Fr * Ft)
        d = torch.sigmoid(self.dis(torch.cat([diff, prod], dim=1)))

        g_tr = d * rt * (1 - rr)
        g_rt = d * rr * (1 - rt)

        patch_tr = self.phi_t(Ft)
        patch_rt = self.phi_r(Fr)

        Fr2 = Fr + self.alpha1 * g_tr * patch_tr
        Ft2 = Ft + self.alpha2 * g_rt * patch_rt

        Ff = torch.cat([Fr2, Ft2], dim=1)
        fuse = Fr2+Ft2
        if return_d2f_activations:
            return Ff, fuse, Fr2, Ft2, rr, rt, d, g_tr, g_rt
        return Ff, fuse

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class AggDeep_v2(nn.Module):
    """Fuse three decoder scales and predict semantic and binary logits."""
    def __init__(self, channel, n_class=9):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.p4 = ConvBNReLU(channel, channel, 3, 1)
        self.p3 = ConvBNReLU(channel, channel, 3, 1)
        self.p2 = ConvBNReLU(channel, channel, 3, 1)

        self.w = nn.Sequential(
            ConvBNReLU(3*channel, channel, 1, 1),
            nn.Conv2d(channel, 3, kernel_size=1, bias=True)
        )

        self.refine =  ConvBNReLU(channel, channel, 3, 1)

        self.head_splat = nn.Conv2d(channel, 2, kernel_size=1, bias=True)
        self.head_out   = nn.Conv2d(channel, n_class, kernel_size=1, bias=True)

    def forward(self, x4, x3, x2):
        f4 = self.p4(self.upsample4(x4))
        f3 = self.p3(self.upsample(x3))
        f2 = self.p2(x2)

        logits = self.w(torch.cat([f4, f3, f2], dim=1))
        w = torch.softmax(logits, dim=1)

        fused = w[:,0:1]*f4 + w[:,1:2]*f3 + w[:,2:3]*f2
        fused = self.refine(fused)

        splat = self.head_splat(fused)
        out   = self.head_out(fused)
        return splat, out

class AggEdgeHF_v3(nn.Module):
    """Fuse shallow features for boundary prediction."""
    def __init__(self, C, edge_out_ch=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.p5 = ConvBNReLU(C, C, 3, 1)
        self.p1 = ConvBNReLU(C, C, 3, 1)
        self.p0 = ConvBNReLU(C, C, 3, 1)

        self.w = nn.Sequential(
            ConvBNReLU(3*C, C, 1, 1),
            nn.Conv2d(C, 3, kernel_size=1, bias=True)
        )

        self.refine = ConvBNReLU(C, C, 3, 1)

        self.dw = ConvBNReLU(C, C, groups=C)

        self.edge_head = nn.Conv2d(C, edge_out_ch, kernel_size=1, bias=True)

    def forward(self, x5, x1, x0):

        f5 = self.p5(self.upsample(x5))
        f1 = self.p1(x1)
        f0 = self.p0(x0)

        logits_w = self.w(torch.cat([f5, f1, f0], dim=1))
        w = torch.softmax(logits_w, dim=1)

        Ff = w[:, 0:1]*f5 + w[:, 1:2]*f1 + w[:, 2:3]*f0
        Ff = self.refine(Ff)

        HF = self.dw(Ff) + Ff

        edge_logits = self.edge_head(HF)

        return edge_logits


class Refine(nn.Module):
    def __init__(self):
        super(Refine,self).__init__()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, attention,x1,x2,x3):
        x1 = x1 + torch.mul(x1, self.upsample2(attention))
        x2 = x2 + torch.mul(x2, self.upsample2(attention))
        x3 = x3 + torch.mul(x3, attention)
        return x1, x2, x3

class DDCR(nn.Module):
    def __init__(self,channel):
        super().__init__()
        self.conv_6 = ConvBNReLU(channel, channel, kernel_size=3, stride=1, groups=channel, dilation=6)
        self.conv_12 = ConvBNReLU(channel, channel, kernel_size=3, stride=1, groups=channel, dilation=12)
        self.conv_18 = ConvBNReLU(channel, channel, kernel_size=3, stride=1, groups=channel, dilation=18)
    def forward(self, x):
        x1 = self.conv_6(x)
        x2 = self.conv_12(x)
        x3 = self.conv_18(x)
        feature_map = x1 + x2 + x3 + x
        return feature_map
class CCG(nn.Module):
    """Generate modality gates with a single-channel CCNN recurrence."""
    def __init__(self, channels=64, alpha=0.5, low_dim=64, num_iterations=4):
        super().__init__()
        
        self.channels = channels
        self.num_iterations = num_iterations
        
        self.down_r = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.down_t = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.conv_depthwise_rgb = nn.Conv2d(
            1, 1, kernel_size=3, stride=1, padding=1, 
            bias=False
        )
        self.conv_depthwise_t = nn.Conv2d(
            1, 1, kernel_size=3, stride=1, padding=1, 
            bias=False
        )
        
        self.a = nn.Parameter(torch.tensor(0.1))
        self.b = nn.Parameter(torch.tensor(1.0))

        self.e_coeff = nn.Parameter(torch.tensor(0.1))

        self.alpha_r = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.alpha_t = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
    
    def forward(self, x_r, x_t, f_r=None, e_r=None, y_r=None, f_t=None, e_t=None, y_t=None, return_activations=False):
        """Return RGB and thermal gates plus recurrent states."""
        x_r_low = self.down_r(x_r)
        x_t_low = self.down_t(x_t)

        if f_r is None:
            f_r_low = torch.zeros_like(x_r_low)
        else:
            f_r_low = f_r
        
        if e_r is None:
            e_r_low = torch.zeros_like(x_r_low)
        else:
            e_r_low = e_r
        
        if y_r is None:
            y_r_low = torch.zeros_like(x_r_low)
        else:
            y_r_low = y_r
        
        if f_t is None:
            f_t_low = torch.zeros_like(x_t_low)
        else:
            f_t_low = f_t
        
        if e_t is None:
            e_t_low = torch.zeros_like(x_t_low)
        else:
            e_t_low = e_t
        
        if y_t is None:
            y_t_low = torch.zeros_like(x_t_low)
        else:
            y_t_low = y_t
        
        y_r_iterations = [] if return_activations else None
        y_t_iterations = [] if return_activations else None
        y_r_sum = torch.zeros_like(y_r_low)
        y_t_sum = torch.zeros_like(y_t_low)

        af = torch.exp(-self.a)
        ae = torch.exp(-self.b)
        
        for iter in range(self.num_iterations):
            y_r_low_old = y_r_low
            y_t_low_old = y_t_low
            conv_y_r_low = self.conv_depthwise_rgb(y_r_low_old)
            f_r_low = af * f_r_low + x_r_low * (1 + conv_y_r_low)
            e_r_low = ae * e_r_low + self.e_coeff * y_r_low_old
            y_r_low = torch.sigmoid(f_r_low - e_r_low)
            conv_y_t_low = self.conv_depthwise_t(y_t_low_old)
            f_t_low = af * f_t_low + x_t_low * (1 + conv_y_t_low)
            e_t_low = ae * e_t_low + self.e_coeff * y_t_low_old
            y_t_low = torch.sigmoid(f_t_low - e_t_low)
            y_r_sum = y_r_sum + y_r_low
            y_t_sum = y_t_sum + y_t_low
            if return_activations:
                y_r_iterations.append((y_r_low).clone().detach())
                y_t_iterations.append((y_t_low).clone().detach())

        y_r_low = y_r_sum / self.num_iterations
        y_t_low = y_t_sum / self.num_iterations

        y_r_low = y_r_low**6
        y_t_low = y_t_low**6
        
        gate_r = y_r_low
        gate_t = y_t_low
        if return_activations:
            return gate_r, gate_t, f_r_low, e_r_low, y_r_low, f_t_low, e_t_low, y_t_low, y_r_iterations, y_t_iterations
        else:
            return gate_r, gate_t, f_r_low, e_r_low, y_r_low, f_t_low, e_t_low, y_t_low

class BSIFNet(nn.Module):

    def __init__(self, n_class):
        super(BSIFNet, self).__init__()

        self.num_resnet_layers = 50

        if self.num_resnet_layers == 18:
            resnet_raw_model1 = models.resnet18(pretrained=True)
            resnet_raw_model2 = models.resnet18(pretrained=True)
            self.inplanes = 512
        elif self.num_resnet_layers == 34:
            resnet_raw_model1 = models.resnet34(pretrained=True)
            resnet_raw_model2 = models.resnet34(pretrained=True)
            self.inplanes = 512
        elif self.num_resnet_layers == 50:
            resnet_raw_model1 = models.resnet50(pretrained=True)
            resnet_raw_model2 = models.resnet50(pretrained=True)
            self.inplanes = 2048
        elif self.num_resnet_layers == 101:
            resnet_raw_model1 = models.resnet101(pretrained=True)
            resnet_raw_model2 = models.resnet101(pretrained=True)
            self.inplanes = 2048
        elif self.num_resnet_layers == 152:
            resnet_raw_model1 = models.resnet152(pretrained=True)
            resnet_raw_model2 = models.resnet152(pretrained=True)
            self.inplanes = 2048

        self.encoder_thermal_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.encoder_thermal_conv1.weight.copy_(resnet_raw_model1.conv1.weight.mean(dim=1, keepdim=True))
        self.encoder_thermal_bn1 = resnet_raw_model1.bn1
        self.encoder_thermal_relu = resnet_raw_model1.relu
        self.encoder_thermal_maxpool = resnet_raw_model1.maxpool
        self.encoder_thermal_layer1 = resnet_raw_model1.layer1
        self.encoder_thermal_layer2 = resnet_raw_model1.layer2
        self.encoder_thermal_layer3 = resnet_raw_model1.layer3
        self.encoder_thermal_layer4 = resnet_raw_model1.layer4

        self.encoder_rgb_conv1 = resnet_raw_model2.conv1
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
        self.encoder_rgb_layer1 = resnet_raw_model2.layer1
        self.encoder_rgb_layer2 = resnet_raw_model2.layer2
        self.encoder_rgb_layer3 = resnet_raw_model2.layer3
        self.encoder_rgb_layer4 = resnet_raw_model2.layer4

        layer0_channels = 64
        if self.num_resnet_layers == 18 or self.num_resnet_layers == 34:
            layer1_channels = 64
            layer2_channels = 128
            layer3_channels = 256
            layer4_channels = 512
        else:  # 50, 101, 152
            layer1_channels = 256
            layer2_channels = 512
            layer3_channels = 1024
            layer4_channels = 2048
        
        self.coupled_ccnn_layer0 = CCG(channels=layer0_channels, num_iterations=6)

        self.d2f_layer0 = SIF(layer0_channels)
        self.d2f_layer1 = SIF(layer1_channels)
        self.d2f_layer2 = SIF(layer2_channels)
        self.d2f_layer3 = SIF(layer3_channels)
        self.d2f_layer4 = SIF(layer4_channels)

        self.Turbo_decoder = Turbo_decoder(n_class)
 
    def forward(self, input, return_activations=False):

        rgb = input[:,:3]
        thermal = input[:,3:]

        verbose = False

        f_r = e_r = y_r = None
        f_t = e_t = y_t = None
        all_activations = {} if return_activations else None

        if verbose: print("rgb.size() original: ", rgb.size())
        if verbose: print("thermal.size() original: ", thermal.size())

        rgb = self.encoder_rgb_conv1(rgb)
        if verbose: print("rgb.size() after conv1: ", rgb.size())
        rgb = self.encoder_rgb_bn1(rgb)
        if verbose: print("rgb.size() after bn1: ", rgb.size())
        rgb = self.encoder_rgb_relu(rgb)
        if verbose: print("rgb.size() after relu: ", rgb.size())

        thermal = self.encoder_thermal_conv1(thermal)
        if verbose: print("thermal.size() after conv1: ", thermal.size())
        thermal = self.encoder_thermal_bn1(thermal)
        if verbose: print("thermal.size() after bn1: ", thermal.size())
        thermal = self.encoder_thermal_relu(thermal)
        if verbose: print("thermal.size() after relu: ", thermal.size())

        rgb = self.encoder_rgb_maxpool(rgb)
        if verbose: print("rgb.size() after maxpool: ", rgb.size())

        thermal = self.encoder_thermal_maxpool(thermal)
        if verbose: print("thermal.size() after maxpool: ", thermal.size())

        if return_activations:
            rgb_input = rgb.clone()
            thermal_input = thermal.clone()
        
        result = self.coupled_ccnn_layer0(
            rgb, thermal, f_r, e_r, y_r, f_t, e_t, y_t, return_activations=return_activations
        )
        if return_activations:
            gate_r, gate_t, f_r, e_r, y_r, f_t, e_t, y_t, y_r_iter, y_t_iter = result
            all_activations['layer0'] = {
                'y_r': y_r_iter, 
                'y_t': y_t_iter,
                'rgb_input': rgb_input,
                'thermal_input': thermal_input
            }
        else:
            gate_r, gate_t, f_r, e_r, y_r, f_t, e_t, y_t = result
        
        rgb = rgb * (1 + self.coupled_ccnn_layer0.alpha_r * gate_t)
        thermal = thermal * (1 + self.coupled_ccnn_layer0.alpha_t * gate_r)

        out_rgbt_0 = self.d2f_layer0(rgb, thermal, return_d2f_activations=return_activations)
        if return_activations:
            Ff_0, fuse_0, Fr2_0, Ft2_0, rr0, rt0, d0, g_tr0, g_rt0 = out_rgbt_0
            all_activations['d2f_layer0'] = {
                'Fr': rgb, 'Ft': thermal,
                'Fr2': Fr2_0, 'Ft2': Ft2_0, 'fuse': fuse_0,
                'rr': rr0, 'rt': rt0, 'd': d0, 'g_tr': g_tr0, 'g_rt': g_rt0
            }
        else:
            Ff_0, fuse_0 = out_rgbt_0
        layer0_channels = Ff_0.shape[1] // 2
        rgb = Ff_0[:, 0:layer0_channels, :, :]
        thermal = Ff_0[:, layer0_channels:layer0_channels*2, :, :]

        rgb = self.encoder_rgb_layer1(rgb)
        if verbose: print("rgb.size() after layer1: ", rgb.size())
        thermal = self.encoder_thermal_layer1(thermal)
        if verbose: print("thermal.size() after layer1: ", thermal.size())
        out_rgbt_1 = self.d2f_layer1(rgb, thermal, return_d2f_activations=return_activations)
        if return_activations:
            Ff_1, fuse_1, Fr2_1, Ft2_1, rr1, rt1, d1, g_tr1, g_rt1 = out_rgbt_1
            all_activations['d2f_layer1'] = {
                'Fr': rgb, 'Ft': thermal,
                'Fr2': Fr2_1, 'Ft2': Ft2_1, 'fuse': fuse_1,
                'rr': rr1, 'rt': rt1, 'd': d1, 'g_tr': g_tr1, 'g_rt': g_rt1
            }
        else:
            Ff_1, fuse_1 = out_rgbt_1
        layer1_channels = Ff_1.shape[1] // 2
        rgb = Ff_1[:, 0:layer1_channels, :, :]
        thermal = Ff_1[:, layer1_channels:layer1_channels*2, :, :]

        rgb = self.encoder_rgb_layer2(rgb)
        if verbose: print("rgb.size() after layer2: ", rgb.size())
        thermal = self.encoder_thermal_layer2(thermal)
        if verbose: print("thermal.size() after layer2: ", thermal.size())
        out_rgbt_2 = self.d2f_layer2(rgb, thermal, return_d2f_activations=return_activations)
        if return_activations:
            Ff_2, fuse_2, Fr2_2, Ft2_2, rr2, rt2, d2, g_tr2, g_rt2 = out_rgbt_2
            all_activations['d2f_layer2'] = {
                'Fr': rgb, 'Ft': thermal,
                'Fr2': Fr2_2, 'Ft2': Ft2_2, 'fuse': fuse_2,
                'rr': rr2, 'rt': rt2, 'd': d2, 'g_tr': g_tr2, 'g_rt': g_rt2
            }
        else:
            Ff_2, fuse_2 = out_rgbt_2
        layer2_channels = Ff_2.shape[1] // 2
        rgb = Ff_2[:, 0:layer2_channels, :, :]
        thermal = Ff_2[:, layer2_channels:layer2_channels*2, :, :]
        rgb = self.encoder_rgb_layer3(rgb)
        if verbose: print("rgb.size() after layer3: ", rgb.size())
        thermal = self.encoder_thermal_layer3(thermal)
        if verbose: print("thermal.size() after layer3: ", thermal.size())
        out_rgbt_3 = self.d2f_layer3(rgb, thermal, return_d2f_activations=return_activations)
        if return_activations:
            Ff_3, fuse_3, Fr2_3, Ft2_3, rr3, rt3, d3, g_tr3, g_rt3 = out_rgbt_3
            all_activations['d2f_layer3'] = {
                'Fr': rgb, 'Ft': thermal,
                'Fr2': Fr2_3, 'Ft2': Ft2_3, 'fuse': fuse_3,
                'rr': rr3, 'rt': rt3, 'd': d3, 'g_tr': g_tr3, 'g_rt': g_rt3
            }
        else:
            Ff_3, fuse_3 = out_rgbt_3
        layer3_channels = Ff_3.shape[1] // 2
        rgb = Ff_3[:, 0:layer3_channels, :, :]
        thermal = Ff_3[:, layer3_channels:layer3_channels*2, :, :]

        rgb = self.encoder_rgb_layer4(rgb)
        if verbose: print("rgb.size() after layer4: ", rgb.size())
        thermal = self.encoder_thermal_layer4(thermal)
        if verbose: print("thermal.size() after layer4: ", thermal.size())
        out_rgbt_4 = self.d2f_layer4(rgb, thermal, return_d2f_activations=return_activations)
        if return_activations:
            Ff_4, fuse_4, Fr2_4, Ft2_4, rr4, rt4, d4, g_tr4, g_rt4 = out_rgbt_4
            all_activations['d2f_layer4'] = {
                'Fr': rgb, 'Ft': thermal,
                'Fr2': Fr2_4, 'Ft2': Ft2_4, 'fuse': fuse_4,
                'rr': rr4, 'rt': rt4, 'd': d4, 'g_tr': g_tr4, 'g_rt': g_rt4
            }
        else:
            Ff_4, fuse_4 = out_rgbt_4
        layer4_channels = Ff_4.shape[1] // 2
        rgb = Ff_4[:, 0:layer4_channels, :, :]
        thermal = Ff_4[:, layer4_channels:layer4_channels*2, :, :]

        encoder_output = [fuse_4, fuse_3, fuse_2, fuse_1, fuse_0]
        hight_output, out, bin_output, edge_output = self.Turbo_decoder(encoder_output)

        if return_activations:
            return hight_output, out, bin_output, edge_output, all_activations
        else:
            return hight_output, out, bin_output, edge_output
  
class TransBottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(TransBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)  
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)  
        self.bn2 = nn.BatchNorm2d(planes)

        if upsample is not None and stride != 1:
            self.conv3 = nn.ConvTranspose2d(planes, planes, kernel_size=2, stride=stride, padding=0, bias=False)  
        else:
            self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)  

        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride
 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out

class LSKA(nn.Module):
    def __init__(self, indim, dim, k_size=7):
        super().__init__()

        self.k_size = k_size

        self.cbl = ConvBNReLU(indim, dim)

        if k_size == 7:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,(3-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=((3-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,2), groups=dim, dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=(2,0), groups=dim, dilation=2)
        elif k_size == 11:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,(3-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=((3-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,4), groups=dim, dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=(4,0), groups=dim, dilation=2)
        elif k_size == 23:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 7), stride=(1,1), padding=(0,9), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(7, 1), stride=(1,1), padding=(9,0), groups=dim, dilation=3)
        elif k_size == 35:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 11), stride=(1,1), padding=(0,15), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(11, 1), stride=(1,1), padding=(15,0), groups=dim, dilation=3)
        elif k_size == 41:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 13), stride=(1,1), padding=(0,18), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(13, 1), stride=(1,1), padding=(18,0), groups=dim, dilation=3)
        elif k_size == 53:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 17), stride=(1,1), padding=(0,24), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(17, 1), stride=(1,1), padding=(24,0), groups=dim, dilation=3)

        self.pw = nn.Conv2d(dim, dim, 1, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        x = self.cbl(x)
        u = x

        a = self.conv0h(x)
        a = self.conv0v(a)
        a = self.conv_spatial_h(a)
        a = self.conv_spatial_v(a)
        a = self.bn(self.pw(a))
        g = self.gate(a)

        return u + u * g

class Turbo_decoder(nn.Module):
    def __init__(self,n_class=9, channel=64):
        super(Turbo_decoder, self).__init__()
        self.rfb2_1 = LSKA(512, channel, k_size=23)
        self.rfb3_1 = LSKA(1024, channel, k_size=23)
        self.rfb4_1 = LSKA(2048, channel, k_size=23)
        self.rfb0_2 = LSKA(64, channel)
        self.rfb1_2 = LSKA(256, channel)
        self.rfb5_2 = LSKA(512, channel)
        self.agg1 = AggDeep_v2(channel)
        self.agg2 = AggEdgeHF_v3(channel)
        self.aspp = DDCR(channel)
        self.HA = Refine()
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.inplanes = channel
        self.agant1 = self._make_agant_layer(channel, channel)
        self.deconv1 = self._make_transpose_layer(TransBottleneck, channel, 3, stride=2)
        self.inplanes = channel
        self.agant2 = self._make_agant_layer(channel, channel)
        self.deconv2 = self._make_transpose_layer(TransBottleneck, channel, 3, stride=2)
        self.out2_conv = nn.Conv2d(channel, n_class, kernel_size=1)
    def _make_transpose_layer(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes, kernel_size=2, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes))
        for m in upsample.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        layers = []
        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))
        layers.append(block(self.inplanes, planes, stride, upsample))

        self.inplanes = planes
        return nn.Sequential(*layers)

    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def forward(self, x):
        rgb, rgb_1, rgb_2_1, rgb_3_1, rgb_4_1 = x[4],x[3],x[2],x[1],x[0]
        x2_1 = self.rfb2_1(rgb_2_1)
        ux2_1 = self.upsample2(x2_1)
        x3_1 = self.rfb3_1(rgb_3_1)
        ux3_1 = self.upsample4(x3_1)
        x4_1 = self.rfb4_1(rgb_4_1)
        ux4_1 = self.upsample(x4_1)
        agg_splat, agg_out = self.agg1(x4_1, x3_1, x2_1)
        attention_gate = torch.softmax(agg_splat, dim=1)[:, 1:2]
        x, x1, x5 = self.HA(attention_gate, rgb, rgb_1, rgb_2_1)
        x0_2 = self.rfb0_2(x)
        ux0_2 = x0_2
        x1_2 = self.rfb1_2(x1)
        ux1_2 = x1_2
        x5_2 = self.rfb5_2(x5)
        ux5_2 = self.upsample2(x5_2)
        agg_out2 = self.agg2(x5_2, x1_2, x0_2)

        edge_gate = torch.softmax(agg_out2, dim=1)[:, 1:2]

        shallow = ux5_2 + ux1_2 + ux0_2
        deep    = ux2_1 + ux3_1 + ux4_1

        feature_map = (1 - edge_gate) * deep + (edge_gate) * shallow
        feature_map = self.aspp(feature_map)
        hight_output = self.upsample(agg_out)
        bin_output = self.upsample(agg_splat)
        edge_output = self.upsample4(agg_out2)
        y = feature_map
        y = self.agant1(y)
        y = self.deconv1(y)
        y = self.agant2(y)
        y = self.deconv2(y)
        y = self.out2_conv(y)
        return hight_output, y, bin_output, edge_output

def unit_test():
    num_minibatch = 2
    rgb = torch.randn(num_minibatch, 3, 480, 640).cuda(0)
    thermal = torch.randn(num_minibatch, 1, 480, 640).cuda(0)
    bsif_net = BSIFNet(9).cuda(0)
    input = torch.cat((rgb, thermal), dim=1)
    hight_output, out, bin_output, edge_output = bsif_net(input)

if __name__ == '__main__':
    unit_test()
