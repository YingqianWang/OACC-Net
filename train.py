import time
import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import *
from tqdm import tqdm
from model import Net


# Settings
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--parallel', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument("--angRes", type=int, default=9, help="angular resolution")
    parser.add_argument('--model_name', type=str, default='OACC-Net')
    parser.add_argument('--trainset_dir', type=str, default='./dataset/training/')
    parser.add_argument('--validset_dir', type=str, default='./dataset/validation/')
    parser.add_argument('--patchsize', type=int, default=48)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--n_epochs', type=int, default=3500, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=3500, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decaying factor')
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='./log/OACC-Net.pth.tar')

    return parser.parse_args()

def train(cfg):
    if cfg.parallel:
        cfg.device = 'cuda:0'
    net = Net(cfg.angRes)
    net.to(cfg.device)
    cudnn.benchmark = True
    epoch_state = 0

    if cfg.load_pretrain:
        if os.path.isfile(cfg.model_path):
            model = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})
            net.load_state_dict(model['state_dict'], strict=False)
        else:
            print("=> no model found at '{}'".format(cfg.load_model))

    if cfg.parallel:
        net = torch.nn.DataParallel(net, device_ids=[0, 1])

    criterion_Loss = torch.nn.L1Loss().to(cfg.device)
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)
    scheduler._step_count = epoch_state

    loss_list = []

    for idx_epoch in range(epoch_state, cfg.n_epochs):
        train_set = TrainSetLoader(cfg)
        train_loader = DataLoader(dataset=train_set, num_workers=cfg.num_workers, batch_size=cfg.batch_size, shuffle=True)
        loss_epoch = []
        for idx_iter, (data, dispGT) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data, dispGT = data.to(cfg.device), dispGT.to(cfg.device)
            disp = net(data, dispGT)
            loss = criterion_Loss(disp, dispGT[:, 0, :, :].unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.data.cpu())

        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            print(time.ctime()[4:-5] + ' Epoch----%5d, loss---%f' % (idx_epoch + 1, float(np.array(loss_epoch).mean())))
            if cfg.parallel:
                save_ckpt({
                'epoch': idx_epoch + 1,
                'state_dict': net.module.state_dict(),
            }, save_path='./log/', filename=cfg.model_name + '.pth')
            else:
                save_ckpt({
                    'epoch': idx_epoch + 1,
                    'state_dict': net.state_dict(),
                }, save_path='./log/', filename=cfg.model_name + '.pth')
        if idx_epoch % 10 == 9:
            if cfg.parallel:
                save_ckpt({
                    'epoch': idx_epoch + 1,
                    'state_dict': net.module.state_dict(),
                }, save_path='./log/', filename=cfg.model_name + str(idx_epoch + 1) + '.pth')
            else:
                save_ckpt({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
            }, save_path='./log/', filename=cfg.model_name + str(idx_epoch + 1) + '.pth')

            valid(net, cfg, idx_epoch + 1)

        scheduler.step()


def valid(net, cfg, epoch):

    torch.no_grad()
    scene_list = ['boxes', 'cotton', 'dino', 'sideboard']
    angRes = cfg.angRes

    txtfile = open(cfg.model_name + '_MSE100.txt', 'a')
    txtfile.write('Epoch={}:\t'.format(epoch))
    txtfile.close()

    for scenes in scene_list:
        lf = np.zeros(shape=(9, 9, 512, 512, 3), dtype=int)
        for i in range(81):
            temp = imageio.imread(cfg.validset_dir + scenes + '/input_Cam0%.2d.png' % i)
            lf[i // 9, i - 9 * (i // 9), :, :, :] = temp
            del temp
        lf = np.mean((1 / 255) * lf.astype('float32'), axis=-1, keepdims=False)
        disp_gt = np.float32(
            read_pfm(cfg.validset_dir + scenes + '/gt_disp_lowres.pfm'))  # load groundtruth disparity map
        angBegin = (9 - angRes) // 2

        lf_angCrop = lf[angBegin:  angBegin + angRes, angBegin: angBegin + angRes, :, :]

        patchsize = 128
        stride = patchsize // 2
        mini_batch = 4

        data = torch.from_numpy(lf_angCrop)
        sub_lfs = LFdivide(data.unsqueeze(2), patchsize, stride)
        n1, n2, u, v, c, h, w = sub_lfs.shape
        sub_lfs = rearrange(sub_lfs, 'n1 n2 u v c h w -> (n1 n2) u v c h w')
        num_inference = (n1 * n2) // mini_batch
        with torch.no_grad():
            out_disp = []
            for idx_inference in range(num_inference):
                current_lfs = sub_lfs[idx_inference * mini_batch: (idx_inference + 1) * mini_batch, :, :, :, :, :]
                input_data = rearrange(current_lfs, 'b u v c h w -> b c (u h) (v w)')
                out_disp.append(net(input_data.to(cfg.device)))

            if (n1 * n2) % mini_batch:
                current_lfs = sub_lfs[(idx_inference + 1) * mini_batch:, :, :, :, :, :]
                input_data = rearrange(current_lfs, 'b u v c h w -> b c (u h) (v w)')
                out_disp.append(net(input_data.to(cfg.device)))

        out_disps = torch.cat(out_disp, dim=0)
        out_disps = rearrange(out_disps, '(n1 n2) c h w -> n1 n2 c h w', n1=n1, n2=n2)
        disp = LFintegrate(out_disps, patchsize, patchsize // 2)
        disp = disp[0: data.shape[2], 0: data.shape[3]]
        disp = np.float32(disp.data.cpu())

        mse100 = np.mean((disp[11:-11, 11:-11] - disp_gt[11:-11, 11:-11]) ** 2) * 100
        txtfile = open(cfg.model_name + '_MSE100.txt', 'a')
        txtfile.write('mse_{}={:3f}\t'.format(scenes, mse100))
        txtfile.close()

    txtfile = open(cfg.model_name + '_MSE100.txt', 'a')
    txtfile.write('\n')
    txtfile.close()

    return


def save_ckpt(state, save_path='./log', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))


if __name__ == '__main__':
    cfg = parse_args()
    train(cfg)
