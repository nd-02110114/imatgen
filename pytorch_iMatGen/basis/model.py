import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, z_size=None, leak_value=0.2):
        super(Encoder, self).__init__()
        self.z_size = z_size
        self.leak_value = leak_value
        # in channel 1 out channel 64 filter 444
        kernel_size = (4, 4, 4)
        self.conv1 = nn.Conv3d(1, 64, kernel_size, stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(64, 64, kernel_size, stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv3 = nn.Conv3d(64, 64, kernel_size, stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv4 = nn.Conv3d(64, 64, kernel_size, stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv5 = nn.Conv3d(64, z_size, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0))

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, self.leak_value)
        x = self.conv2(x)
        x = F.leaky_relu(x, self.leak_value)
        x = self.conv3(x)
        x = F.leaky_relu(x, self.leak_value)
        x = self.conv4(x)
        x = F.leaky_relu(x, self.leak_value)
        x = self.conv5(x)
        x = torch.tanh(x)
        return x


class Decoder(nn.Module):
    def __init__(self, z_size=None, leak_value=0.4):
        super(Decoder, self).__init__()
        self.z_size = z_size
        self.leak_value = leak_value
        # in channel 1 out channel 64 filter 444
        kernel_size = (4, 4, 4)
        self.deconv1 = nn.ConvTranspose3d(z_size, 64, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0))
        self.deconv2 = nn.ConvTranspose3d(64, 64, kernel_size, stride=(2, 2, 2), padding=(1, 1, 1))
        self.deconv3 = nn.ConvTranspose3d(64, 64, kernel_size, stride=(2, 2, 2), padding=(1, 1, 1))
        self.deconv4 = nn.ConvTranspose3d(64, 64, kernel_size, stride=(2, 2, 2), padding=(1, 1, 1))
        self.deconv5 = nn.ConvTranspose3d(64, 1, kernel_size, stride=(2, 2, 2), padding=(1, 1, 1))

    def forward(self, x):
        x = x.view(-1, self.z_size, 1, 1, 1)
        x = self.deconv1(x)
        x = F.leaky_relu(x, self.leak_value)
        x = self.deconv2(x)
        x = F.leaky_relu(x, self.leak_value)
        x = self.deconv3(x)
        x = F.leaky_relu(x, self.leak_value)
        x = self.deconv4(x)
        x = F.leaky_relu(x, self.leak_value)
        x = self.deconv5(x)
        x = torch.sigmoid(x)
        return x


class BasisAutoEncoder(nn.Module):
    def __init__(self, z_size=200, leak_value=0.2):
        super(BasisAutoEncoder, self).__init__()
        self.encoder = Encoder(z_size, leak_value)
        self.decoder = Decoder(z_size, leak_value)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
