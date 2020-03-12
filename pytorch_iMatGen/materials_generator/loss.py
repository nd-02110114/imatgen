import torch
import torch.nn as nn


class VAELoss(nn.Module):
    def __init__(self, coef_kl=1e-6, coef_classify=0):
        super(VAELoss, self).__init__()
        self.coef_kl = coef_kl
        self.coef_classify = coef_classify

    def forward(self, outputs, inputs, mean, log_var, logits, labels):
        mse_loss = nn.MSELoss()
        kl_loss = -0.5 * torch.mean(torch.sum(1 + log_var - mean**2 - torch.exp(log_var)))
        classify_loss = nn.BCEWithLogitsLoss()
        loss = mse_loss(outputs, inputs) + self.coef_kl * kl_loss + \
            self.coef_classify * classify_loss(logits, labels)
        return loss
