import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, z_size=None, leak_value=0.2):
        super(Encoder, self).__init__()
        self.z_size = z_size
        self.leak_value = leak_value
        self.conv1 = nn.Conv2d(6, 100, (1, 4), stride=(1, 2), padding=(0, 1))
        self.conv2 = nn.Conv2d(100, 100, (1, 4), stride=(1, 2), padding=(0, 1))
        self.conv3 = nn.Conv2d(100, 100, (1, 4), stride=(1, 2), padding=(0, 1))
        self.conv4 = nn.Conv2d(100, 50, (1, 1), stride=(1, 1), padding=(0, 0))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1250, self.z_size*2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, self.leak_value)
        x = self.conv2(x)
        x = F.leaky_relu(x, self.leak_value)
        x = self.conv3(x)
        x = F.leaky_relu(x, self.leak_value)
        x = self.conv4(x)
        x = F.leaky_relu(x, self.leak_value)
        x = self.flatten(x)
        x = self.fc(x)
        mean, log_var = torch.split(x, int(x.size(1)/2), dim=1)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, z_size=None, leak_value=0.2):
        super(Decoder, self).__init__()
        self.z_size = z_size
        self.leak_value = leak_value
        self.fc = nn.Linear(self.z_size, 1250)
        self.deconv1 = nn.ConvTranspose2d(50, 100, (1, 1), stride=(1, 1), padding=(0, 0))
        self.deconv2 = nn.ConvTranspose2d(100, 100, (1, 4), stride=(1, 2), padding=(0, 1))
        self.deconv3 = nn.ConvTranspose2d(100, 100, (1, 4), stride=(1, 2), padding=(0, 1))
        self.deconv4 = nn.ConvTranspose2d(100, 6, (1, 4), stride=(1, 2), padding=(0, 1))

    def forward(self, x):
        x = self.fc(x)
        x = F.leaky_relu(x, self.leak_value)
        # channel, height, width = (50, 1, 25)
        x = x.view(-1, 50, 1, 25)
        x = self.deconv1(x)
        x = F.leaky_relu(x, self.leak_value)
        x = self.deconv2(x)
        x = F.leaky_relu(x, self.leak_value)
        x = self.deconv3(x)
        x = F.leaky_relu(x, self.leak_value)
        x = self.deconv4(x)
        x = torch.tanh(x)
        return x


class FormationEnergyClassifier(nn.Module):
    def __init__(self, z_size=None):
        super(FormationEnergyClassifier, self).__init__()
        self.fc = nn.Linear(z_size, 500)
        self.out = nn.Linear(500, 1)

    def forward(self, x):
        x = self.fc(x)
        x = torch.tanh(x)
        logits = self.out(x)
        return logits


class MaterialGenerator(nn.Module):
    def __init__(self, z_size=500, leak_value=0.2):
        super(MaterialGenerator, self).__init__()
        self.z_size = z_size
        self.leak_value = leak_value
        self.encoder = Encoder(z_size, leak_value)
        self.decoder = Decoder(z_size, leak_value)
        self.classifier = FormationEnergyClassifier(z_size)

    def sampling(self, x):
        mean, log_var = self.encoder(x)
        epsilon = torch.randn(mean.shape).to(mean.device)
        z = mean + torch.exp(log_var / 2) * epsilon
        return z, mean, log_var

    def decode(self, z):
        y = self.decoder(z)
        return y

    def classify(self, z):
        logits = self.classifier(z)
        return logits
