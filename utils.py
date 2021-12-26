#from PIL import Image
import os
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import random
from einops import rearrange
import imageio
from func_pfm import *


class TrainSetLoader(Dataset):
    def __init__(self, cfg):
        super(TrainSetLoader, self).__init__()
        files = os.listdir(cfg.trainset_dir)
        self.trainset_dir = cfg.trainset_dir
        self.files = files
        self.angRes = cfg.angRes
        self.patchsize = cfg.patchsize
        """ We use totally 16 LF images (0 to 15) for training. Since some images (4,6,15) have a reflection region,
                we decrease the occurrence frequency of them. """
        scene_idx = []
        for i in range(3):
            scene_idx = np.append(scene_idx, [0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14])
        for i in range(1):
            scene_idx = np.append(scene_idx, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

        self.scene_idx = scene_idx.astype('int')

        self.item_num = len(self.scene_idx)

    def __getitem__(self, index):
        scene_id = self.scene_idx[index]
        scene_name = self.files[scene_id]
        lf = np.zeros(shape=(9, 9, 512, 512, 3), dtype=int)

        """ Read inputs """
        for i in range(81):
            temp = imageio.imread(self.trainset_dir + scene_name + '/input_Cam0%.2d.png' % i)
            lf[i // 9, i - 9 * (i // 9), :, :, :] = temp

        dispGT = np.zeros(shape=(512, 512, 2), dtype=float)
        dispGT[:, :, 0] = np.float32(read_pfm(self.trainset_dir + scene_name + '/gt_disp_lowres.pfm'))
        mask_rgb = imageio.imread(self.trainset_dir + scene_name + '/valid_mask.png')
        dispGT[:, :, 1] = np.float32(mask_rgb[:, :, 1] > 0)
        dispGT = dispGT.astype('float32')

        """ Data Augmentation """
        lf = illuminance_augmentation((1/255) * lf.astype('float32'))
        #lf = viewpoint_augmentation(lf, self.angRes)
        lf, dispGT, scale = scale_augmentation(lf, dispGT, self.patchsize)
        if scale == 1:
            lf, dispGT, refocus_flag = refocus_augmentation(lf, dispGT)
        else:
            refocus_flag = 0

        sum_diff = 0
        glass_region = False
        while (sum_diff < 0.01 or glass_region == True):
            lf_crop, dispGT_crop = random_crop(lf, dispGT, self.patchsize, refocus_flag)
            if (scene_id == 4 or scene_id == 6 or scene_id == 15):
                glass_region = np.sum(dispGT_crop[:, :, 1]) < self.patchsize * self.patchsize
            if glass_region == False:
                sum_diff = np.sum(np.abs(lf_crop[self.angRes//2, self.angRes//2, :, :] -
                                         np.squeeze(lf_crop[self.angRes//2, self.angRes//2, self.patchsize//2, self.patchsize//2]))
                                  ) / (self.patchsize * self.patchsize)

        data = rearrange(lf_crop, 'a1 a2 h w -> (a1 h) (a2 w)', a1=self.angRes, a2=self.angRes)
        data, label = orientation_augmentation(data, dispGT_crop)
        data = data.astype('float32')
        label = label.astype('float32')
        data = ToTensor()(data.copy())
        label = ToTensor()(label.copy())

        return data, label

    def __len__(self):
        return self.item_num


def illuminance_augmentation(data):
    rand_3color = 0.05 + np.random.rand(3)
    rand_3color = rand_3color / np.sum(rand_3color)
    R = rand_3color[0]
    G = rand_3color[1]
    B = rand_3color[2]
    data_gray = np.squeeze(R * data[:, :, :, :, 0] + G * data[:, :, :, :, 1] + B * data[:, :, :, :, 2])
    gray_rand = 0.4 * np.random.rand() + 0.8
    data_gray = pow(data_gray, gray_rand)
    noise_rand = np.random.randint(0, 10)
    if noise_rand == 0:
        gauss = np.random.normal(0.0, np.random.uniform() * np.sqrt(0.2), data_gray.shape)
        data_gray = np.clip(data_gray + gauss, 0.0, 1.0)

    return data_gray


def refocus_augmentation(lf, dispGT):
    refocus_rand = np.random.randint(0, 5)
    refocus_flag = 0
    if refocus_rand == 0:
        refocus_flag = 1
        angRes, _, h, w = lf.shape
        center = (angRes - 1) // 2
        min_d = int(np.min(dispGT[:, :, 0]))
        max_d = int(np.max(dispGT[:, :, 0]))
        dispLen = 6 - (max_d - min_d)
        k = np.random.randint(dispLen + 1) - 3
        dd = k - min_d
        out_dispGT = np.zeros((h, w, 2), dtype=float)
        out_dispGT[:, :, 0] = dispGT[:, :, 0] + dd
        out_dispGT[:, :, 1] = dispGT[:, :, 1]
        out_lf = np.zeros((angRes, angRes, h, w), dtype=float)
        for u in range(angRes):
            for v in range(angRes):
                dh, dw = dd * (u - center), dd * (v - center)
                if (dh > 0) & (dw > 0):
                    out_lf[u, v, 0:-dh-1, 0:-dw-1] = lf[u, v, dh:-1, dw:-1]
                elif (dh > 0) & (dw == 0):
                    out_lf[u, v, 0:-dh-1, :] = lf[u, v, dh:-1, :]
                elif (dh > 0) & (dw < 0):
                    out_lf[u, v, 0:-dh-1, -dw:-1] = lf[u, v, dh:-1, 0:dw-1]
                elif (dh == 0) & (dw > 0):
                    out_lf[u, v, :, 0:-dw-1] = lf[u, v, :, dw:-1]
                elif (dh == 0) & (dw == 0):
                    out_lf[u, v, :, :] = lf[u, v, :, :]
                elif (dh == 0) & (dw < 0):
                    out_lf[u, v, :, -dw:-1] = lf[u, v, :, 0:dw-1]
                elif (dh < 0) & (dw > 0):
                    out_lf[u, v, -dh:-1, 0:-dw-1] = lf[u, v, 0:dh-1, dw:-1]
                elif (dh < 0) & (dw == 0):
                    out_lf[u, v, -dh:-1, :] = lf[u, v, 0:dh-1, :]
                elif (dh < 0) & (dw < 0):
                    out_lf[u, v, -dh:-1, -dw:-1] = lf[u, v, 0:dh-1, 0:dw-1]
                else:
                    pass
    else:
        out_lf, out_dispGT = lf, dispGT

    return out_lf, out_dispGT, refocus_flag


def scale_augmentation(lf, dispGT, patchsize):
    if patchsize > 48:
        kk = np.random.randint(14)
    else:
        kk = np.random.randint(17)
    if (kk < 8):
        scale = 1
    elif (kk < 14):
        scale = 2
    elif (kk < 17):
        scale = 3
    out_lf = lf[:, :, 0::scale, 0::scale]
    out_disp = dispGT[0::scale, 0::scale]
    out_disp[:, :, 0] = out_disp[:, :, 0] / scale

    return out_lf, out_disp, scale


def orientation_augmentation(data, dispGT):
    if random.random() < 0.5:  # flip along W-V direction
        data = data[:, ::-1]
        dispGT = dispGT[:, ::-1, :]
    if random.random() < 0.5:  # flip along H-U direction
        data = data[::-1, :]
        dispGT = dispGT[::-1, :, :]
    if random.random() < 0.5: # transpose between U-V and H-W
        data = data.transpose(1, 0)
        dispGT = dispGT.transpose(1, 0, 2)

    return data, dispGT


def viewpoint_augmentation(data_in, angRes):
    if (angRes == 3):
        #u, v = np.random.randint(0, 7), np.random.randint(0, 7)
        u, v = 3, 3
    if (angRes == 5):
        #u, v = np.random.randint(0, 5), np.random.randint(0, 5)
        u, v = 2, 2
    if (angRes == 7):
        #u, v = np.random.randint(0, 3), np.random.randint(0, 3)
        u, v = 1, 1
    if (angRes == 9):
        u, v = 0, 0
    data_out = data_in[u : u + angRes, v : v + angRes, :, :]

    return data_out


def random_crop(lf, dispGT, patchsize, refocus_flag):
    angRes, angRes, h, w = lf.shape
    if refocus_flag == 1:
        bdr = 16
    else:
        bdr = 0
    h_idx = np.random.randint(bdr, h - patchsize - bdr)
    w_idx = np.random.randint(bdr, w - patchsize - bdr)
    out_lf = lf[:, :, h_idx : h_idx + patchsize, w_idx : w_idx + patchsize]
    out_disp = dispGT[h_idx : h_idx + patchsize, w_idx : w_idx + patchsize, :]

    return out_lf, out_disp
