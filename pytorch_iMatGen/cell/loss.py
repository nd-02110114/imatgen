import torch.nn as nn


from utils.postprocess import post_process


class Reconstruction(nn.Module):
    def __init__(self):
        super(Reconstruction, self).__init__()

    def forward(self, outputs, inputs):
        outputs = post_process(outputs)
        criterion = nn.MSELoss()
        return criterion(outputs, inputs)
