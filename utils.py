import torch
import math
import os
import numpy as np
import cv2

def log10(x):
    return torch.log(x) / math.log(10)

class MetricAggregator(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0

        self.sum_irmse, self.sum_imae = 0, 0
        self.sum_mse, self.sum_rmse, self.sum_mae = 0, 0, 0
        self.sum_absrel, self.sum_rmse_log, self.sum_lg10 = 0, 0, 0
        self.sum_delta1, self.sum_delta2, self.sum_delta3 = 0, 0, 0
        self.sum_gpu_time = 0

    def evaluate(self, output, target, mask=None, gpu_time=0):
        if mask is None:
            mask = (target>0)
        else:
            output =  output[mask == 1]
            target =  target[mask == 1]
        
        mse, rmse, mae, rmse_log, lg10, absrel, a1, a2, a3, irmse, imae = self.compute_errors(output, target)

        self.sum_mse += mse
        self.sum_rmse += rmse
        self.sum_mae += mae  
        self.sum_rmse_log += rmse_log      
        self.sum_lg10 += lg10
        self.sum_absrel += absrel

        self.sum_delta1 += a1
        self.sum_delta2 += a2
        self.sum_delta3 += a3

        self.sum_irmse += irmse
        self.sum_imae += imae

        self.sum_gpu_time += gpu_time

        self.count += 1

    def compute_errors(self, output, target):

        abs_diff = (output - target).abs()

        mse = float((torch.pow(abs_diff, 2)).mean())
        rmse = math.sqrt(mse)
        mae = float(abs_diff.mean())
        rmse_log = (torch.log(output) - torch.log(target)) ** 2
        rmse_log = math.sqrt(rmse_log.mean())
        lg10 = float((log10(output) - log10(target)).abs().mean())
        absrel = float((abs_diff / target).mean())

        maxRatio = torch.max(output / target, target / output)
        delta1 = float((maxRatio < 1.25).float().mean())
        delta2 = float((maxRatio < 1.25 ** 2).float().mean())
        delta3 = float((maxRatio < 1.25 ** 3).float().mean())

        inv_output = 1 / output
        inv_target = 1 / target
        abs_inv_diff = (inv_output - inv_target).abs()
        irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        imae = float(abs_inv_diff.mean())

        return mse, rmse, mae, rmse_log, lg10, absrel, delta1, delta2, delta3, irmse, imae

    def average(self):
        return {
            "rmse": self.sum_rmse / self.count,
            "mae": self.sum_mae / self.count,
            "mse": self.sum_mse / self.count,
            "irmse": self.sum_irmse / self.count, 
            "imae": self.sum_imae / self.count,
            "absrel": self.sum_absrel / self.count,
            "rmse_log": self.sum_rmse_log/self.count,
            "log10": self.sum_lg10 / self.count,
            "delta1": self.sum_delta1 / self.count,
            "delta2": self.sum_delta2 / self.count,
            "delta3": self.sum_delta3 / self.count,
            "gpu_time": self.sum_gpu_time / self.count
        }

def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

def disparity_to_depth(prediction, target, mask):
    # transform predicted disparity to aligned depth
    target_disparity = torch.zeros_like(target)
    target_disparity[mask == 1] = 1.0 / target[mask == 1]

    scale, shift = compute_scale_and_shift(prediction, target_disparity, mask)
    prediction_aligned = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

    disparity_cap = 1.0 / 10
    prediction_aligned[prediction_aligned < disparity_cap] = disparity_cap

    prediction_depth = 1.0 / prediction_aligned

    return prediction_depth


def is_image_file(filename):
  IMG_EXTENSIONS = ['.jpg']
  return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_images(dir):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in os.walk(d):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, os.path.splitext(fname)[0])
                    images.append(path)
    return images

def write_depth(path, depth, bits=1):
    """Write depth map to png file.

    Args:
        path (str): filepath without extension
        depth (array): depth
    """

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape)
    
    if bits == 1:
        cv2.imwrite(path + ".png", out.astype("uint8"))
    elif bits == 2:
        cv2.imwrite(path + ".png", out.astype("uint16"))

    return