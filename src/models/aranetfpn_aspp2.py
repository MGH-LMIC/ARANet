import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class AdaptiveResolutionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdaptiveResolutionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.attention = AdaptiveAttention(out_channels)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.attention(x)
        return x

class AdaptiveAttention(nn.Module):
    def __init__(self, channels):
        super(AdaptiveAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // 16, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // 16, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.atrous_1 = nn.Conv2d(in_channels, out_channels, 1, 1)
        self.atrous_6 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=6, dilation=6)
        self.atrous_12 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=12, dilation=12)
        self.atrous_18 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=18, dilation=18)
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, 1)
        )
        self.out = nn.Conv2d(out_channels * 5, out_channels, 1, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        atrous_1 = self.atrous_1(x)
        atrous_6 = self.atrous_6(x)
        atrous_12 = self.atrous_12(x)
        atrous_18 = self.atrous_18(x)
        
        im_pool = self.image_pool(x)
        im_pool = F.interpolate(im_pool, size=x.size()[2:], mode='bilinear', align_corners=True)
        
        out = torch.cat([atrous_1, atrous_6, atrous_12, atrous_18, im_pool], dim=1)
        out = self.out(out)
        out = self.bn(out)
        return self.relu(out)

class ARANetFPN(nn.Module):
    def __init__(self, num_classes):
        super(ARANetFPN, self).__init__()
        
     
        
        # Load pretrained VGG16 as encoder
        vgg16 = models.vgg16(pretrained=True)
        self.encoder = nn.ModuleList(vgg16.features)
        
        # Adaptive Resolution Modules
        self.arm5 = AdaptiveResolutionModule(512, 512)
        self.arm4 = AdaptiveResolutionModule(512, 512)
        self.arm3 = AdaptiveResolutionModule(256, 256)
        self.arm2 = AdaptiveResolutionModule(128, 128)
        self.arm1 = AdaptiveResolutionModule(64, 64)
        
        # FPN layers
        self.fpn_conv1 = nn.Conv2d(512, 256, kernel_size=1)
        self.fpn_conv2 = nn.Conv2d(512, 256, kernel_size=1)
        self.fpn_conv3 = nn.Conv2d(256, 256, kernel_size=1)
        self.fpn_conv4 = nn.Conv2d(128, 256, kernel_size=1)
        self.fpn_conv5 = nn.Conv2d(64, 256, kernel_size=1)
        
        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        # ASPP modules for each FPN level
        self.aspp_p5 = ASPP(256, 256)
        self.aspp_p4 = ASPP(256, 256)
        self.aspp_p3 = ASPP(256, 256)
        self.aspp_p2 = ASPP(256, 256)
        self.aspp_p1 = ASPP(256, 256)
        
        # Final layers
        self.final_conv = nn.Conv2d(256*5, 64, kernel_size=3, padding=1)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
       
        
        # Encoder path with skip connections
        skip_connections = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in [4, 9, 16, 23, 30]:  # Collect skip connections
                skip_connections.append(x)
        
        # Apply Adaptive Resolution Modules
        c5 = self.arm5(x)
        c4 = self.arm4(skip_connections[3])
        c3 = self.arm3(skip_connections[2])
        c2 = self.arm2(skip_connections[1])
        c1 = self.arm1(skip_connections[0])
        
        # FPN top-down pathway and lateral connections
        p5 = self.fpn_conv1(c5)
        p4 = self.upsample(p5) + self.fpn_conv2(c4)
        p3 = self.upsample(p4) + self.fpn_conv3(c3)
        p2 = self.upsample(p3) + self.fpn_conv4(c2)
        p1 = self.upsample(p2) + self.fpn_conv5(c1)
        
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        p1 = self.smooth4(p1)
        
        #print(f"After smooth shapes: p4:{p4.shape}, p3:{p3.shape}, p2:{p2.shape}, p1:{p1.shape}")
        
        # Apply ASPP to each FPN level
        p5 = self.aspp_p5(p5)
        p4 = self.aspp_p4(p4)
        p3 = self.aspp_p3(p3)
        p2 = self.aspp_p2(p2)
        p1 = self.aspp_p1(p1)
        
        #print(f"After aspp shapes: p5:{p5.shape}, p4:{p4.shape}, p3:{p3.shape}, p2:{p2.shape}, p1:{p1.shape}")
        
        # Upsample all to the same size (p1 size)
        p5 = self.upsample(self.upsample(self.upsample(self.upsample(self.upsample(p5)))))
        p4 = self.upsample(self.upsample(self.upsample(self.upsample(p4))))
        p3 = self.upsample(self.upsample(self.upsample(p3)))
        p2 = self.upsample(self.upsample(p2))
        p1 = self.upsample(p1)
        
        #print(f"After upsample shapes: p5:{p5.shape}, p4:{p4.shape}, p3:{p3.shape}, p2:{p2.shape}, p1:{p1.shape}")
        
        # Concatenate all FPN outputs
        out = torch.cat([p1, p2, p3, p4, p5], dim=1)
        
        # Final convolutions
        out = self.final_conv(out)
        out = self.final(out)
        
        return out
