import random
import scipy
import numpy as np
import torch
from PIL import Image
import cv2
from torchvision.transforms import functional as tfn


class OCIDTransform:
    """Transformer function for OCID_grasp dataset
    """

    def __init__(self,
                 shortest_size,
                 longest_max_size,
                 rgb_mean=None,
                 rgb_std=None,
                 random_flip=False,
                 random_scale=None,
                 rotate_and_scale=False):
        self.shortest_size = shortest_size
        self.longest_max_size = longest_max_size
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.random_flip = random_flip
        self.random_scale = random_scale
        self.rotate_and_scale = rotate_and_scale

    def _adjusted_scale(self, in_width, in_height, target_size):
        min_size = min(in_width, in_height)
        max_size = max(in_width, in_height)
        scale = target_size / min_size

        if int(max_size * scale) > self.longest_max_size:
            scale = self.longest_max_size / max_size

        return scale

    @staticmethod
    def _random_flip(img, msk):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            msk = [m.transpose(Image.FLIP_LEFT_RIGHT) for m in msk]
            return img, msk
        else:
            return img, msk

    def _random_target_size(self):
        if len(self.random_scale) == 2:
            target_size = random.uniform(self.shortest_size * self.random_scale[0],
                                         self.shortest_size * self.random_scale[1])
        else:
            target_sizes = [self.shortest_size * scale for scale in self.random_scale]
            target_size = random.choice(target_sizes)
        return int(target_size)

    def _normalize_image(self, img):
        if self.rgb_mean is not None:
            img.sub_(img.new(self.rgb_mean).view(-1, 1, 1))
        if self.rgb_std is not None:
            img.div_(img.new(self.rgb_std).view(-1, 1, 1))
        return img

    @staticmethod
    def _Rotate2D(pts, cnt, ang):
        ang = np.deg2rad(ang)
        return scipy.dot(pts - cnt,
                         scipy.array([[scipy.cos(ang), scipy.sin(ang)], [-scipy.sin(ang), scipy.cos(ang)]])) + cnt

    @staticmethod
    def _prepare_frcnn_format(boxes, im_size):
        A = boxes
        xy_ctr = np.sum(A, axis=2) / 4
        x_ctr = xy_ctr[:, 0]
        y_ctr = xy_ctr[:, 1]
        width = np.sqrt(np.sum((A[:, :, 0] - A[:, :, 1]) ** 2, axis=1))
        height = np.sqrt(np.sum((A[:, :, 1] - A[:, :, 2]) ** 2, axis=1))

        theta = np.zeros((A.shape[0]), dtype=np.int)

        theta = np.arctan((A[:, 1, 1] - A[:, 1, 0]) / (A[:, 0, 0] - A[:, 0, 1]))
        b = np.arctan((A[:, 1, 0] - A[:, 1, 1]) / (A[:, 0, 1] - A[:, 0, 0]))
        theta[np.where(A[:, 0, 0] <= A[:, 0, 1])] = b[np.where(A[:, 0, 0] <= A[:, 0, 1])]

        # used for fasterrcnn loss
        x_min = x_ctr - width / 2
        x_max = x_ctr + width / 2
        y_min = y_ctr - height / 2
        y_max = y_ctr + height / 2

        x_coords = np.vstack((x_min, x_max))
        y_coords = np.vstack((y_min, y_max))

        mat = np.asarray((np.all(x_coords > im_size[1], axis=0), np.all(x_coords < 0, axis=0),
                          np.all(y_coords > im_size[0], axis=0), np.all(y_coords < 0, axis=0)))

        fail = np.any(mat, axis=0)
        correct_idx = np.where(fail == False)
        theta_deg = np.rad2deg(theta) + 90
        cls = (np.round((theta_deg) / (180 / 18))).astype(int)
        cls[np.where(cls == 18)] = 0

        if np.any(cls) > 17:
            assert False

        ret_value = (
        x_min[correct_idx], y_min[correct_idx], theta_deg[correct_idx], x_max[correct_idx], y_max[correct_idx],
        cls[correct_idx])
        return ret_value

    def _rotateAndScale(self, img, msk, all_boxes):
        im_size = [self.shortest_size, self.longest_max_size]

        img_pad = cv2.copyMakeBorder(img, 200, 200, 200, 200, borderType=cv2.BORDER_REPLICATE)
        msk_pad = cv2.copyMakeBorder(msk, 200, 200, 200, 200, borderType=cv2.BORDER_CONSTANT)

        (oldY, oldX, chan) = img_pad.shape  # note: numpy uses (y,x) convention but most OpenCV functions use (x,y)

        theta = float(np.random.randint(360) - 1)
        dx = np.random.randint(101) - 51
        dy = np.random.randint(101) - 51

        M = cv2.getRotationMatrix2D(center=(oldX / 2, oldY / 2), angle=theta,
                                    scale=1.0)  # rotate about center of image.

        # choose a new image size.
        newX, newY = oldX, oldY
        # include this if you want to prevent corners being cut off
        r = np.deg2rad(theta)
        newX, newY = (abs(np.sin(r) * newY) + abs(np.cos(r) * newX), abs(np.sin(r) * newX) + abs(np.cos(r) * newY))

        # Find the translation that moves the result to the center of that region.
        (tx, ty) = ((newX - oldX) / 2, (newY - oldY) / 2)
        M[0, 2] += tx
        M[1, 2] += ty

        imgRotate = cv2.warpAffine(img_pad, M, dsize=(int(newX), int(newY)))
        mskRotate = cv2.warpAffine(msk_pad, M, dsize=(int(newX), int(newY)))

        imgRotateCrop = imgRotate[
                        int(imgRotate.shape[0] / 2 - (im_size[0] / 2)) - dx:int(
                            imgRotate.shape[0] / 2 + (im_size[0] / 2)) - dx,
                        int(imgRotate.shape[1] / 2 - (im_size[1] / 2)) - dy:int(
                            imgRotate.shape[1] / 2 + (im_size[1] / 2)) - dy, :]
        mskRotateCrop = mskRotate[
                        int(mskRotate.shape[0] / 2 - (im_size[0] / 2)) - dx:int(
                            mskRotate.shape[0] / 2 + (im_size[0] / 2)) - dx,
                        int(mskRotate.shape[1] / 2 - (im_size[1] / 2)) - dy:int(
                            mskRotate.shape[1] / 2 + (im_size[1] / 2)) - dy]

        bbsInShift = np.zeros_like(all_boxes)
        bbsInShift[:, 0, :] = all_boxes[:, 0, :] - (im_size[1] / 2)
        bbsInShift[:, 1, :] = all_boxes[:, 1, :] - (im_size[0] / 2)
        R = np.array([[np.cos(theta / 180 * np.pi), -np.sin(theta / 180 * np.pi)],
                      [np.sin(theta / 180 * np.pi), np.cos(theta / 180 * np.pi)]])
        R_all = np.expand_dims(R, axis=0)  #
        R_all = np.repeat(R_all, all_boxes.shape[0], axis=0)
        bbsInShift = np.swapaxes(bbsInShift, 1, 2)

        bbsRotated = np.dot(bbsInShift, R_all.T)
        bbsRotated = bbsRotated[:, :, :, 0]
        bbsRotated = np.swapaxes(bbsRotated, 1, 2)
        bbsInShiftBack = np.asarray(bbsRotated)
        bbsInShiftBack[:, 0, :] = (bbsRotated[:, 0, :] + (im_size[1] / 2) + dy)
        bbsInShiftBack[:, 1, :] = (bbsRotated[:, 1, :] + (im_size[0] / 2) + dx)

        return imgRotateCrop, mskRotateCrop, bbsInShiftBack

    def __call__(self, img_, msk_, bbox_infos_):
        im_size = [self.shortest_size, self.longest_max_size]
        bbox_infos_ = np.swapaxes(bbox_infos_, 1, 2)

        x_min = int(img_.shape[0] / 2 - int(im_size[0] / 2))
        x_max = int(img_.shape[0] / 2 + int(im_size[0] / 2))
        y_min = int(img_.shape[1] / 2 - int(im_size[1] / 2))
        y_max = int(img_.shape[1] / 2 + int(im_size[1] / 2))

        new_origin = np.array([[y_min], [x_min]])

        img = img_[x_min:x_max, y_min:y_max, :]

        msk = msk_[x_min:x_max, y_min:y_max]

        bbox_infos_ = bbox_infos_ - new_origin
        bbox_infos = np.copy(bbox_infos_)

        if self.rotate_and_scale:
            img, msk, bbox_transformed = self._rotateAndScale(img, msk, bbox_infos_)
            bbox_infos = bbox_transformed
        # Random flip
        if self.random_flip:
            img, msk = self._random_flip(img, msk)

        # Adjust scale, possibly at random
        if self.random_scale is not None:
            target_size = self._random_target_size()
        else:
            target_size = self.shortest_size

        ret = self._prepare_frcnn_format(bbox_infos, im_size)
        (x1, y1, theta, x2, y2, cls) = ret
        if len(cls) == 0:
            print('NO valid boxes after augmentation, switch to gt values')
            ret = self._prepare_frcnn_format(bbox_infos_, im_size)
            img = img_[x_min:x_max, y_min:y_max, :]

            msk = msk_[x_min:x_max, y_min:y_max]

        bbox_infos = np.asarray(ret).T
        bbox_infos = bbox_infos.astype(np.float32)

        # Image transformations
        img = tfn.to_tensor(img)
        img = self._normalize_image(img)

        # Label transformations
        msk = np.stack([np.array(m, dtype=np.int32, copy=False) for m in msk], axis=0)

        # Convert labels to torch and extract bounding boxes
        msk = torch.from_numpy(msk.astype(np.long))

        bbx = torch.from_numpy(np.asarray(bbox_infos)).contiguous()
        if bbox_infos.shape[1] != 6:
            assert False

        return dict(img=img, msk=msk, bbx=bbx), im_size


class OCIDTestTransform:
    """Transformer function for OCID_grasp dataset, used at test time
    """

    def __init__(self,
                 shortest_size,
                 longest_max_size,
                 rgb_mean=None,
                 rgb_std=None):
        self.longest_max_size = longest_max_size
        self.shortest_size = shortest_size
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std

    def _adjusted_scale(self, in_width, in_height):
        min_size = min(in_width, in_height)
        scale = self.shortest_size / min_size
        return scale

    def _normalize_image(self, img):
        if self.rgb_mean is not None:
            img.sub_(img.new(self.rgb_mean).view(-1, 1, 1))
        if self.rgb_std is not None:
            img.div_(img.new(self.rgb_std).view(-1, 1, 1))
        return img

    def __call__(self, img):
        im_size = [self.shortest_size, self.longest_max_size]

        x_min = int(img.shape[0] / 2 - int(im_size[0] / 2))
        x_max = int(img.shape[0] / 2 + int(im_size[0] / 2))
        y_min = int(img.shape[1] / 2 - int(im_size[1] / 2))
        y_max = int(img.shape[1] / 2 + int(im_size[1] / 2))

        img = img[x_min:x_max, y_min:y_max, :]

        # Image transformations
        img = tfn.to_tensor(img)
        img = self._normalize_image(img)

        return img, im_size
