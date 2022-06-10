import torch
import kornia as K

import torch.nn as nn
import torch.nn.functional as F

class DepthLoss(nn.Module):
    def __init__(self, alpha=0.5, k=3):
        super(DepthLoss, self).__init__()
        self.mse = torch.nn.L1Loss()
        self.grad = MultiScaleGradientLoss(alpha, k)
    
    def forward(self, pred, target):
        mse = self.mse(pred, target)
        grad = self.grad(pred, target)
        loss = mse + grad 
        self.last_mse = mse.item()
        self.last_grad = grad.item()
        return loss

class MultiScaleGradientLoss(nn.Module):
    def __init__(self, alpha, k):
        super(MultiScaleGradientLoss, self).__init__()
        self.alpha = alpha
        self.k = k

    def forward(self, prediction_d, d_gt): 
        N = torch.numel(d_gt)
        loss = 0.0
        a = self.alpha
        sc_prediction_d = prediction_d
        grads_gt = K.filters.spatial_gradient(d_gt, mode='diff', order=1, normalized=False)
        dx_true = grads_gt[:, :, 0]
        dy_true = grads_gt[:, :, 1] 
        for i in range(0, self.k):     

            grads_pred = K.filters.spatial_gradient(prediction_d, mode='diff', order=1, normalized=False)    
            dx_pred = grads_pred[:, :, 0]
            dy_pred = grads_pred[:, :, 1] 
            
            h_gradient = torch.abs(dy_pred - dy_true)
            v_gradient = torch.abs(dx_pred - dx_true)

            gradient_loss = torch.sum(h_gradient) + torch.sum(v_gradient)
            gradient_loss*= a
            a /= 2
            loss += gradient_loss

            if(i<self.k-1):
                sc_prediction_d = F.interpolate(sc_prediction_d, scale_factor=0.5, recompute_scale_factor=False)
                prediction_d = F.interpolate(sc_prediction_d, size=d_gt.shape[2:], mode="bicubic", align_corners=False)
            

        loss = loss / N

        return loss