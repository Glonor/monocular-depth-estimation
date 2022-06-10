import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def depthwise(in_channels, kernel_size):
    padding = (kernel_size-1) // 2
    assert 2*padding == kernel_size-1, "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
    return nn.Sequential(
          nn.Conv2d(in_channels,in_channels,kernel_size,stride=1,padding=padding,bias=False,groups=in_channels),
          nn.BatchNorm2d(in_channels),
          nn.ReLU(inplace=True),
        )

def pointwise(in_channels, out_channels):
    return nn.Sequential(
          nn.Conv2d(in_channels,out_channels,1,1,0,bias=False),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True),
        )

class MobileNetV3SkipAdd(nn.Module):
    def __init__(self, pretrained=False):

        super(MobileNetV3SkipAdd, self).__init__()
        self.original_model = models.mobilenet_v3_large(pretrained)
        
        # Adding skip connections
        for module_pos, module in self.original_model.features._modules.items():
          if(module_pos == '2'):
            for m_pos, m in module.block._modules.items():
              if(m_pos == '0'):
                m[2].register_forward_hook(get_activation('skip1'))
          if(module_pos == '7'):
            for m_pos, m in module.block._modules.items():
              if(m_pos == '0'):  
                m[2].register_forward_hook(get_activation('skip2'))
          if(module_pos == '11'):
            for m_pos, m in module.block._modules.items():
              if(m_pos == '0'):
                m[2].register_forward_hook(get_activation('skip3'))

        kernel_size = 5

        self.decode_conv1 = nn.Sequential(
            depthwise(960, kernel_size),
            pointwise(960, 480))
        self.decode_conv2 = nn.Sequential(
            depthwise(480, kernel_size),
            pointwise(480, 240))
        self.decode_conv3 = nn.Sequential(
            depthwise(240, kernel_size),
            pointwise(240, 120))
        self.decode_conv4 = nn.Sequential(
            depthwise(120, kernel_size),
            pointwise(120, 64))
        self.decode_conv5 = nn.Sequential(
            depthwise(64, kernel_size),
            pointwise(64, 32))
        self.decode_conv6 = pointwise(32, 1)

    def forward(self, x):
        x = self.original_model.features(x)
        x1 = activation['skip1']
        x2 = activation['skip2']
        x3 = activation['skip3']   
        
        for i in range(1,6):
            layer = getattr(self, 'decode_conv{}'.format(i))
            x = layer(x)
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            
            if i==4:
                x = x + x1
            elif i==2:
                x = x + x2
            elif i==1:
                x = x + x3
            
        x = self.decode_conv6(x)
        return x