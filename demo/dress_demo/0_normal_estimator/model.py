import torch
import torch.nn as nn

VARIABLE_COUNTER = 0

NUM_CH = [64,128,256,512,1024]
KER_SZ = 3


class conv_block_1(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block_1, self).__init__()
#        
#        self.conv = nn.Sequential(
#            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
#            nn.BatchNorm2d(out_ch),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
#            nn.BatchNorm2d(out_ch),
#            nn.ReLU(inplace=True))
        
        self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x


class Generate(nn.Module):
    """
    """
    def __init__(self, in_ch=3, out_ch=3):
        super(Generate, self).__init__()
#        NUM_CH = [64,128,256,512,1024]
#        n1 = 32
#        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
#        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv0 = conv_block_1(in_ch, NUM_CH[0])
        self.Conv1 = conv_block_1(NUM_CH[0], NUM_CH[0])
        self.Conv2 = conv_block_1(NUM_CH[0], NUM_CH[0])
        
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv3 = conv_block_1(NUM_CH[0], NUM_CH[1])
        
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv4 = conv_block_1(NUM_CH[1], NUM_CH[2])
        
        self.Maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv5 = conv_block_1(NUM_CH[2], NUM_CH[3])
        
        self.Maxpool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv6 = conv_block_1(NUM_CH[3], NUM_CH[4])
        
        self.Conv7 = conv_block_1(NUM_CH[4], NUM_CH[4])
        self.Conv8 = conv_block_1(NUM_CH[4], NUM_CH[3])
        
        self.Conv9  = conv_block_1(NUM_CH[4], NUM_CH[3])
        self.Conv10 = conv_block_1(NUM_CH[3], NUM_CH[2])
        
        self.Conv11 = conv_block_1(NUM_CH[3], NUM_CH[2])
        self.Conv12 = conv_block_1(NUM_CH[2], NUM_CH[1])
        
        self.Conv13 = conv_block_1(NUM_CH[2], NUM_CH[1])
        self.Conv14 = conv_block_1(NUM_CH[1], NUM_CH[0])
        
        self.Conv15 = conv_block_1(NUM_CH[1], NUM_CH[0])
        self.Conv16 = conv_block_1(NUM_CH[0], NUM_CH[0])
#        self.Up5 = torch.nn.functional.upsample_bilinear()


#        self.Up_conv4 = conv_block(filters[3], filters[2])
#
#        self.Up3 = up_conv(filters[2], filters[1])
#        self.Up_conv3 = conv_block(filters[2], filters[1])
#
#        self.Up2 = up_conv(filters[1], filters[0])
#        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv_out = nn.Conv2d(NUM_CH[0], out_ch, kernel_size=1, stride=1, padding=0)
#
        self.active = torch.nn.Tanh()

    def forward(self, x):

        e1 = self.Conv0(x)
        e2 = self.Conv1(e1)
        e2 = self.Conv2(e2)
        
        e3 = self.Maxpool3(e2)
        e3_1 = self.Conv3(e3)

        e4 = self.Maxpool4(e3_1)
        e4_1 = self.Conv4(e4)

        e5 = self.Maxpool5(e4_1)
        e5_1 = self.Conv5(e5)
        
        e6 = self.Maxpool6(e5_1)
        e6_1 = self.Conv6(e6)
        
        e7 = self.Conv7(e6_1)
        e8 = self.Conv8(e7)
        
        d1 =  nn.functional.interpolate(e8, scale_factor=2)
        cat0 = torch.cat([d1,e5_1],1)
        
        e9  = self.Conv9(cat0)
        e10 = self.Conv10(e9)
        
        d2 =  nn.functional.interpolate(e10, scale_factor=2)
        cat1 = torch.cat([d2,e4_1],1)
        
        e11 = self.Conv11(cat1)
        e12 = self.Conv12(e11)
        
        d3 =  nn.functional.interpolate(e12, scale_factor=2)
        cat2 = torch.cat([d3,e3_1],1)
        
        e13 = self.Conv13(cat2)
        e14 = self.Conv14(e13)
        
        d4 =  nn.functional.interpolate(e14, scale_factor=2)
        cat3 = torch.cat([d4,e2],1)
        
        e15 = self.Conv15(cat3)
        e16 = self.Conv16(e15)
        
        out = self.Conv_out(e16)
        out = self.active(out)
        return out