import torch
import time
import argparse
from utils import *
from model import Net


# Settings
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angRes", type=int, default=9, help="angular resolution")
    parser.add_argument('--model_name', type=str, default='DistgDisp')
    parser.add_argument('--testset_dir', type=str, default='./demo_input/')
    parser.add_argument('--crop', type=bool, default=True)
    parser.add_argument('--patchsize', type=int, default=128)
    parser.add_argument('--model_path', type=str, default='./log/DistgDisp.pth.tar')
    parser.add_argument('--save_path', type=str, default='./Results/')
    return parser.parse_args()

'''
Note: 1) We crop LFs into overlapping patches to save the CUDA memory during inference. 
      2) Since we have not optimize our cropping scheme, when cropping is performed, 
         the inference time will be longer than the one reported in our paper.
'''

def test(cfg):
    net = Net(cfg.angRes)
    net.to(cfg.device)
    model = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})
    net.load_state_dict(model['state_dict'])

    scene_list = os.listdir(cfg.testset_dir)

    angRes = cfg.angRes

    for scenes in scene_list:
        print('Working on scene: ' + scenes + '...')
        temp = imageio.imread(cfg.testset_dir + scenes + '/input_Cam000.png')
        lf = np.zeros(shape=(9, 9, temp.shape[0], temp.shape[1], 3), dtype=int)
        for i in range(81):
            temp = imageio.imread(cfg.testset_dir + scenes + '/input_Cam0%.2d.png' % i)
            lf[i // 9, i - 9 * (i // 9), :, :, :] = temp
        lf_gray = np.mean((1 / 255) * lf.astype('float32'), axis=-1, keepdims=False)
        angBegin = (9 - angRes) // 2
        lf_angCrop = lf_gray[angBegin:  angBegin + angRes, angBegin: angBegin + angRes, :, :]
        data = rearrange(lf_angCrop, 'u v h w -> (u h) (v w)')

        if cfg.crop == False:
            data = ToTensor()(data.copy())
            data = data.unsqueeze(0)
            with torch.no_grad():
                disp = net(data.to(cfg.device))
            disp = np.float32(disp[0,0,:,:].data.cpu())

        else:
            patchsize = cfg.patchsize
            stride = patchsize // 2
            subLFin = LFdivide(data, cfg.angRes, patchsize, stride)  # numU, numV, h*angRes, w*angRes
            numU, numV, H, W = subLFin.shape
            subLFout = np.zeros(shape=(numU, numV, patchsize, patchsize), dtype='float32')
            for u in range(numU):
                for v in range(numV):
                    sub_data = subLFin[u, v, :, :]
                    sub_data = ToTensor()(sub_data.copy())
                    with torch.no_grad():
                        sub_out = net(sub_data.unsqueeze(0).to(cfg.device))
                        subLFout[u, v, :, :] = sub_out.squeeze().cpu().numpy()
            bdr = (patchsize - stride) // 2
            disp = np.zeros(shape=(numU * stride, numV * stride), dtype='float32')

            for ku in range(numU):
                for kv in range(numV):
                    disp[ku * stride:(ku + 1) * stride, kv * stride:(kv + 1) * stride] = subLFout[ku, kv, bdr:-bdr, bdr:-bdr]

        print('Finished! \n')
        write_pfm(disp, cfg.save_path + '%s.pfm' % (scenes))

    return


def LFdivide(data, angRes, pz, stride):
    uh, vw = data.shape
    h0 = uh //angRes
    w0 = vw //angRes
    bdr = (pz - stride) // 2
    h = h0 + 2 * bdr
    w = w0 + 2 * bdr
    if (h - pz) % stride:
        numU = (h - pz)//stride + 2
    else:
        numU = (h - pz)//stride + 1
    if (w - pz) % stride:
        numV = (w - pz)//stride + 2
    else:
        numV = (w - pz)//stride + 1
    hE = stride * (numU-1) + pz
    wE = stride * (numV-1) + pz

    dataE = np.zeros(shape=(hE*angRes, wE*angRes), dtype='float32')
    for u in range(angRes):
        for v in range(angRes):
            Im = data[u*h0:(u+1)*h0, v*w0:(v+1)*w0]
            dataE[u*hE : u*hE+h, v*wE : v*wE+w] = ImageExtend(Im, bdr)
    subLF = np.zeros(shape=(numU, numV, pz*angRes, pz*angRes), dtype='float32')
    for kh in range(numU):
        for kw in range(numV):
            for u in range(angRes):
                for v in range(angRes):
                    uu = u*hE + kh*stride
                    vv = v*wE + kw*stride
                    subLF[kh, kw, u*pz:(u+1)*pz, v*pz:(v+1)*pz] = dataE[uu:uu+pz, vv:vv+pz]
    return subLF


def ImageExtend(Im, bdr):
    h, w = Im.shape
    Im_lr = Im[:, ::-1]
    Im_ud = Im[::-1, :]
    Im_diag = Im[::-1, ::-1]
    Im_up = np.concatenate((Im_diag, Im_ud, Im_diag), 1)
    Im_mid = np.concatenate((Im_lr, Im, Im_lr), 1)
    Im_down = np.concatenate((Im_diag, Im_ud, Im_diag), 1)
    Im_Ext = np.concatenate((Im_up, Im_mid, Im_down), 0)
    Im_out = Im_Ext[h-bdr : 2*h+bdr, w-bdr : 2*w+bdr]

    return Im_out


def LF2SAI(x):
    u, v, h, w = x.shape
    out = np.zeros((u * h, v * w), dtype=np.float32)
    for i in range(u):
        for j in range(v):
            out[i*h : (i+1)*h, j*w : (j+1)*w] = x[i, j, :, :]
    return out


if __name__ == '__main__':
    cfg = parse_args()
    test(cfg)
