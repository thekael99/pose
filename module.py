import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torchvision


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))

        x = self.relu(self.batch_norm2(self.conv2(x)))

        x = self.conv3(x)
        x = self.batch_norm3(x)

        # downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        # add identity
        # print(x.shape)
        # print(identity.shape)
        x += identity
        x = self.relu(x)

        return x


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        # print("x: ", x.shape)
        # print("id: ", identity.shape)
        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.transpose1 = nn.ConvTranspose2d(2048, 1024, 3, stride=2, padding=1, output_padding=1)

        self.upsample = self._upsample()

        # self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # self.fc = nn.Linear(512*ResBlock.expansion, num_classes)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)  # (1024,16,16)
        layer4 = self.layer4(layer3)  # (2048,8,8)

        transpose1 = self.transpose1(layer4)  # (1024,16,16)

        x = self.upsample(transpose1)

        return x, torch.cat((transpose1, layer3), dim=1)

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * ResBlock.expansion)
            )

        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)

    def _upsample(self):
        layers = [
            nn.ConvTranspose2d(1024, 128, 3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 15, kernel_size=3, stride=1, padding=1)]

        return nn.Sequential(*layers)


def Extractor(backbone="Resnet50", channels=3):
    if backbone == "Resnet50":
        layer = [3, 4, 6, 3]
    elif backbone == "Resnet101":
        layer = [3, 4, 23, 3]
    elif backbone == "Resnet152":
        layer = [3, 8, 36, 3]
    else:
        raise Exception("Not Implement")
    return ResNet(Bottleneck, layer, channels)


class Distance(nn.Module):
    def __init__(self):
        super(Distance, self).__init__()

        self.downsample1 = nn.Sequential(
            nn.Conv2d(2048, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128))
        self.block1 = Block(2048, 128, i_downsample=self.downsample1)

        self.pool1 = nn.MaxPool2d(2, padding=0)

        self.downsample2 = nn.Sequential(
            nn.Conv2d(128, 16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16))
        self.block2 = Block(128, 16, i_downsample=self.downsample2)

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1024, 15)
        self.relu = nn.ReLU()

    def forward(self, x):  # (2048,16,16)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        return x  # 15


class ConvEncoder(nn.Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv2d(15, 128, 3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, padding=0)
        self.conv2 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, padding=0)
        self.conv3 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, padding=0)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

    def forward(self, x):
        conv1 = self.conv1(x)
        relu1 = self.relu(conv1)
        pool1 = self.pool1(relu1)
        conv2 = self.conv2(pool1)
        relu2 = self.relu(conv2)
        pool2 = self.pool2(relu2)
        conv3 = self.conv3(pool2)
        relu3 = self.relu(conv3)
        pool3 = self.pool3(relu3)
        flatten = self.flatten(pool3)
        return flatten


class FCEncoder(nn.Module):
    def __init__(self):
        super(FCEncoder, self).__init__()
        self.linear1 = nn.Linear(8192, 2048)
        self.linear2 = nn.Linear(2048, 512)
        self.linear3 = nn.Linear(512, 20)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ln1 = self.linear1(x)
        relu1 = self.relu(ln1)
        ln2 = self.linear1(relu1)
        relu2 = self.relu(ln2)
        ln3 = self.linear3(relu2)
        sigmoid1 = self.sigmoid(ln3)
        return sigmoid1


class FC3DPose(nn.Module):
    def __init__(self):
        super(FC3DPose, self).__init__()
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(35, 40)
        self.linear2 = nn.Linear(40, 40)
        self.linear3 = nn.Linear(40, 45)

    def forward(self, latent, distance):
        x = torch.cat((latent, distance), dim=1)
        ln1 = self.linear1(x)
        relu1 = self.relu(ln1)
        ln2 = self.linear1(relu1)
        relu2 = self.relu(ln2)
        ln3 = self.linear3(relu2)
        relu3 = self.relu(ln3)
        return relu3


class FCDecoder(nn.Module):
    def __init__(self):
        super(FCDecoder, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.linear1 = nn.Linear(20, 512)
        self.linear2 = nn.Linear(512, 2048)
        self.linear3 = nn.Linear(2048, 8192)

    def forward(self, x):
        ln1 = self.linear1(x)
        relu1 = self.relu(ln1)
        ln2 = self.linear1(relu1)
        relu2 = self.relu(ln2)
        ln3 = self.linear3(relu2)
        sigmoid1 = self.sigmoid(ln3)
        unflatten = nn.Unflatten(1, (512, 4, 4))(sigmoid1)
        return unflatten


class ConvDecoder(nn.Module):
    def __init__(self):
        super(ConvDecoder, self).__init__()

        self.conv1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.transpose1 = nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1, output_padding=1)

        self.conv2 = nn.Conv2d(512, 128, 3, stride=1, padding=1)
        self.transpose2 = nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1)

        self.conv3 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.transpose3 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)

        self.conv4 = nn.Conv2d(64, 15, 3, stride=1, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        conv1 = self.conv1(x)
        relu1 = self.relu(conv1)
        transpose1 = self.transpose1(relu1)

        conv2 = self.conv2(transpose1)
        relu2 = self.relu(conv2)
        transpose2 = self.transpose2(relu2)

        conv3 = self.conv3(transpose2)
        relu3 = self.relu(conv3)
        transpose3 = self.transpose3(relu3)

        conv4 = self.conv4(transpose3)
        relu4 = self.relu(conv4)

        return relu4
