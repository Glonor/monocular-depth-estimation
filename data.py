import cv2
import h5py
import numpy as np

from scipy.io import loadmat

from torch.utils.data import Dataset
from torchvision.transforms import Compose
from transforms import Resize, NormalizeImage, PrepareForNet

def read_image(filename):
    img = cv2.imread(filename)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    return img

class Places_Dataset(Dataset):

    def __init__(self, filenames):
        self.imgs = filenames
        self.transform = Compose(
            [
                NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                PrepareForNet(),
            ]
        )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        rgb = read_image(img_path + '.jpg')
        input_image = self.transform({"image": rgb})["image"]
        depth = cv2.imread(img_path + '.png', -1)        
        depth = depth.astype(np.float32) / 1000.0

        depth_min = depth.min()
        depth_max = depth.max()

        depth = (depth - depth_min) / (depth_max - depth_min)
        
        assert np.sum(np.isnan(depth)) == 0

        return input_image, depth

class NyuDepthV2_Dataset(Dataset):
    def __init__(self, datapath, splitpath, config, split="test"):

        self.__image_list = []
        self.__depth_list = []

        self.__transform = Compose(
            [
                Resize(
                    config['width'],
                    config['height'],
                    resize_target=False,
                    keep_aspect_ratio=False,
                    ensure_multiple_of=32,
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                PrepareForNet(),
            ]
        )

        mat = loadmat(splitpath)

        if split == "train":
            indices = [ind[0] - 1 for ind in mat["trainNdxs"]]
        elif split == "test":
            indices = [ind[0] - 1 for ind in mat["testNdxs"]]
        else:
            raise ValueError("Split {} not found.".format(split))

        with h5py.File(datapath, "r") as f:
            for ind in indices:
                self.__image_list.append(np.swapaxes(f["images"][ind], 0, 2))
                self.__depth_list.append(np.swapaxes(f["rawDepths"][ind], 0, 1))

        self.__length = len(self.__image_list)

    def __len__(self):
        return self.__length

    def __getitem__(self, index):
        # image
        image = self.__image_list[index]
        image = image / 255

        # depth
        depth = self.__depth_list[index]

        # mask; cf. project_depth_map.m in toolbox_nyu_depth_v2 (max depth = 10.0)
        mask = (depth > 0) & (depth < 10.0)
        eval_mask = np.zeros(mask.shape)
        eval_mask[45:471, 41:601] = 1
        valid_mask = np.logical_and(mask, eval_mask)

        # sample
        sample = {}
        sample["image"] = image
        sample["depth"] = depth
        sample["mask"] = valid_mask

        # transforms
        if self.__transform is not None:
            sample = self.__transform(sample)

        return sample