#!/usr/bin/env python3
# license removed for brevity
import pathlib
import sys

file_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(file_path))
sys.path.append(str(file_path.parent.parent))
print(sys.path)

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from wp4_msgs.msg import DepthFrame

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model import MobileNetV3SkipAdd

class DepthEstimation():
    def __init__(self):
        self.device = 'cpu'
        model_path = os.path.join(os.path.dirname(__file__), 'model_0.4494756457743855.pth')
        self.model = MobileNetV3SkipAdd()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)

        self.depth_pusblisher = rospy.Publisher("/vision_msgs/depth_estimation", DepthFrame, queue_size=10)
        self.head_front_image_subscriber = rospy.Subscriber("/head_front_camera/color/image_raw/compressed", CompressedImage, self.CallBack)

    
    def CallBack(self, data):
        nparr = np.frombuffer(data.data, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB) / 255.0

        #resize
        sample = cv2.resize(
            image,
            (256, 192),
            interpolation=cv2.INTER_AREA,
        )

        #normalize
        sample = (sample - [0.5, 0.5, 0.5]) / [0.5, 0.5, 0.5]

        #prepare for net
        sample = np.transpose(sample, (2, 0, 1))
        sample = np.ascontiguousarray(sample).astype(np.float32)
        
        input_batch = torch.from_numpy(sample).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_batch)
                
        depth = prediction.squeeze().cpu().numpy()
        depth_min = depth.min()
        depth_max = depth.max()

        max_val = (2**(8))-1

        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (depth - depth_min) / (depth_max - depth_min)
        else:
            out = np.zeros(depth.shape, dtype=depth.type)    

        msg = DepthFrame()
        msg.height = out.shape[0]
        msg.width = out.shape[1]
        out = out.flatten().tolist()
        msg.data = out     
        msg.header.stamp = data.header.stamp
        
        self.depth_pusblisher.publish(msg)

if __name__ == '__main__':
    rospy.init_node("monocular_depth_estimation", anonymous=False)
    depth_estimation = DepthEstimation()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass