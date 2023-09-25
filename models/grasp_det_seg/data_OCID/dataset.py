from os import path
import cv2
import numpy as np
import torch.utils.data as data
import os
from PIL import Image


class OCIDDataset(data.Dataset):
    """OCID_grasp dataset for grasp detection and semantic segmentation
    """

    def __init__(self, data_path, root_dir, split_name, transform):
        super(OCIDDataset, self).__init__()
        self.data_path = data_path
        self.root_dir = root_dir
        self.split_name = split_name
        self.transform = transform

        self._images = self._load_split()

    def _load_split(self):
        with open(path.join(self.data_path, self.split_name + ".txt"), "r") as fid:
            images = [x.strip() for x in fid.readlines()]

        return images

    def _load_item(self, item):
        seq_path, im_name = item.split(',')
        sample_path = os.path.join(self.root_dir, seq_path)
        img_path = os.path.join(sample_path, 'rgb', im_name)
        mask_path = os.path.join(sample_path, 'seg_mask_labeled_combi', im_name)
        anno_path = os.path.join(sample_path, 'Annotations', im_name[:-4] + '.txt')
        img_bgr = cv2.imread(img_path)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        with open(anno_path, "r") as f:
            points_list = []
            boxes_list = []
            for count, line in enumerate(f):
                line = line.rstrip()
                [x, y] = line.split(' ')

                x = float(x)
                y = float(y)

                pt = (x, y)
                points_list.append(pt)

                if len(points_list) == 4:
                    boxes_list.append(points_list)
                    points_list = []

        msk = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        box_arry = np.asarray(boxes_list)
        return img, msk, box_arry

    @property
    def categories(self):
        """Category names"""
        return self._meta["categories"]

    @property
    def num_categories(self):
        """Number of categories"""
        return len(self.categories)

    @property
    def num_stuff(self):
        """Number of "stuff" categories"""
        return self._meta["num_stuff"]

    @property
    def num_thing(self):
        """Number of "thing" categories"""
        return self.num_categories - self.num_stuff

    @property
    def original_ids(self):
        """Original class id of each category"""
        return self._meta["original_ids"]

    @property
    def palette(self):
        """Default palette to be used when color-coding semantic labels"""
        return np.array(self._meta["palette"], dtype=np.uint8)

    @property
    def img_sizes(self):
        """Size of each image of the dataset"""
        return [img_desc["size"] for img_desc in self._images]

    @property
    def img_categories(self):
        """Categories present in each image of the dataset"""
        return [img_desc["cat"] for img_desc in self._images]

    @property
    def get_images(self):
        """Categories present in each image of the dataset"""
        return self._images

    def __len__(self):
        return len(self._images)

    def __getitem__(self, item):
        im_rgb, msk, bbox_infos = self._load_item(item)

        rec, im_size = self.transform(im_rgb, msk, bbox_infos)

        rec["abs_path"] = item
        rec["root_path"] = self.root_dir
        rec["im_size"] = im_size
        return rec

    def get_raw_image(self, idx):
        """Load a single, unmodified image with given id from the dataset"""
        img_file = path.join(self._img_dir, idx)
        if path.exists(img_file + ".png"):
            img_file = img_file + ".png"
        elif path.exists(img_file + ".jpg"):
            img_file = img_file + ".jpg"
        else:
            raise IOError("Cannot find any image for id {} in {}".format(idx, self._img_dir))

        return Image.open(img_file)

    def get_image_desc(self, idx):
        """Look up an image descriptor given the id"""
        matching = [img_desc for img_desc in self._images if img_desc["id"] == idx]
        if len(matching) == 1:
            return matching[0]
        else:
            raise ValueError("No image found with id %s" % idx)


class OCIDTestDataset(data.Dataset):

    def __init__(self, data_path, root_dir, split_name, transform):
        super(OCIDTestDataset, self).__init__()
        self.data_path = data_path
        self.root_dir = root_dir
        self.split_name = split_name
        self.transform = transform

        self._images = self._load_split()

    def _load_split(self):
        with open(path.join(self.data_path, self.split_name + ".txt"), "r") as fid:
            images = [x.strip() for x in fid.readlines()]
        return images

    @property
    def img_sizes(self):
        """Size of each image of the dataset"""
        return [img_desc["size"] for img_desc in self._images]

    @property
    def get_images(self):
        """Categories present in each image of the dataset"""
        return self._images

    def __len__(self):
        return len(self._images)

    def __getitem__(self, item):
        seq_path, im_name = item.split(',')
        sample_path = os.path.join(self.root_dir, seq_path)
        img_path = os.path.join(sample_path, 'rgb', im_name)
        img_bgr = cv2.imread(img_path)
        im_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        img_, im_size = self.transform(im_rgb)

        return {"img": img_,
                "root_path": self.root_dir,
                "abs_path": item,
                "im_size": im_size
                }
