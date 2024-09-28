#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: yuchun   time: 2020/7/10
from collections import namedtuple
import scipy.io as scio
import torch
from utils.affine import *
from com_psnr import quality
from net import *
from net.fcn import fcn
from net.losses import *
from net.noise import *
from utils.image_io import *
from models.skip3D import skip3D
from models.network_ffdnet import *
import os
from net.layers import MyRELU
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
Result = namedtuple("Result", ['recon', 'psnr', 'AB', 'c', 'F'])


class Dehaze(object):
    def __init__(self, image_original, meas, num_iter, R, mask, tau, lamda, beta):
        self.image_original = image_original  # 65536, 191
        self.measurement = meas  # 256 256 191
        self.num_iter = num_iter
        self.mask = mask
        self.image_net = None
        self.mask_net = None
        self.mse_loss = None
        self.learning_rate = 0.001
        self.reg_std = 0.001
        self.beta = beta
        self.parameters = None
        self.current_result = None
        self.input_depth = 1
        self.output_depth = 1
        self.exp_weight = 0.99
        self.blur_loss = None
        self.best_result = None
        self.best_result_av = None
        self.image_net_inputs = None
        self.mask_net_inputs = None
        self.image_out = None
        self.mask_out = None
        self.total_loss = None
        self.tau = tau
        self.rank = R
        self._init_all()
        self.out_avg = 0
        self.save_every = 1000
        self.previous = np.zeros(self.image_original.shape)
        self.max_psnr = 0
        self.lamda = lamda
        self.psnr_list = []
        self.psnr_av_list = []
        self.re_list = []
    def _init_images(self):


        self.measurement_torch = np_to_torch(self.measurement).type(torch.cuda.FloatTensor)
        # 65536 191
        self.measurement_torch = self.measurement_torch.squeeze(0)

        self.mask_torch = np_to_torch(self.mask).type(torch.cuda.FloatTensor)
        self.mask_torch = self.mask_torch.squeeze(0)
        self.rotate_theta = torch.zeros((32), device=0, requires_grad=True)
        self.Scale_factor = torch.ones((32), device=0, requires_grad=True)
        self.x = torch.zeros((32), device=0, requires_grad=True)
        self.y = torch.zeros((32), device=0, requires_grad=True)

    def _init_nets(self):
        pad = 'reflection'
        data_type = torch.cuda.FloatTensor

        # image_net =
        self.image_net = []
        self.parameters = []
        for i in range(self.rank):
            # net = skip(self.input_depth, self.output_depth, num_channels_down=[16, 32, 64, 128, 128, 128],
            #            num_channels_up=[16, 32, 64, 128, 128, 128],
            #            num_channels_skip=[0, 0, 4, 4, 4, 4],
            #            filter_size_down=[7, 7, 5, 5, 3, 3], filter_size_up=[7, 7, 5, 5, 3, 3],
            #            upsample_mode='bilinear', downsample_mode='avg',
            #            need_sigmoid=False, need_bias=True, pad=pad, act_fun='LeakyReLU').type(data_type)
            net = skip(self.input_depth, self.output_depth, num_channels_down=[16, 32, 64, 128, 128],
                       num_channels_up=[16, 32, 64, 128, 128],
                       num_channels_skip=[0, 0, 6, 6, 6],
                       filter_size_down=[9, 7, 5, 3, 1], filter_size_up=[9, 7, 5, 3, 1],
                       upsample_mode='bilinear', downsample_mode='avg',
                       need_sigmoid=False, pad=pad, act_fun='LeakyReLU').type(data_type)
            self.parameters = [p for p in net.parameters()] + self.parameters
            self.image_net.append(net)
        self.move_net = skip3D(1, 1,
                   num_channels_down=[16, 32, 64, 128, 128],
                   num_channels_up=[16, 32, 64, 128, 128],
                   num_channels_skip=[0, 0, 4, 4, 4],
                   filter_size_up=3, filter_size_down=3, filter_skip_size=1,
                   upsample_mode='nearest',  # downsample_mode='avg',
                   need1x1_up=False,
                   need_sigmoid=False, need_bias=True, pad=pad, act_fun='LeakyReLU').type(data_type)

        # self.move_net = skip3D(1, 1,
        #                        num_channels_down=[16, 32, 64, 128, 128],
        #                        num_channels_up=[16, 32, 64, 128, 128],
        #                        num_channels_skip=[0, 0, 4, 4, 4],
        #                        filter_size_up=3, filter_size_down=3, filter_skip_size=1,
        #                        upsample_mode='nearest',  # downsample_mode='avg',
        #                        need1x1_up=False,
        #                        need_sigmoid=False, need_bias=True, pad=pad, act_fun='LeakyReLU').type(data_type)

        # self.move_net = skip3D(1, 1,
        #                        num_channels_down=[16, 32, 64, 128, 128, 128],
        #                        num_channels_up=[16, 32, 64, 128, 128, 128],
        #                        num_channels_skip=[0, 0, 4, 4, 4, 4],
        #                        filter_size_up=3, filter_size_down=3, filter_skip_size=1,
        #                        upsample_mode='nearest',  # downsample_mode='avg',
        #                        need1x1_up=False,
        #                        need_sigmoid=False, need_bias=True, pad=pad, act_fun='LeakyReLU').type(data_type)


        self.parameters = [p for p in self.move_net.parameters()] + self.parameters

        self.mask_net = []
        for i in range(self.rank):
            net = fcn(32, 32, num_hidden=[32, 64, 128, 128, 64, 32]).type(data_type)
            self.parameters = self.parameters + [p for p in net.parameters()]
            self.mask_net.append(net)
        self.parameters_fangshe = []
        self.parameters_fangshe += [self.x]
        self.parameters_fangshe += [self.y]
        self.parameters_fangshe += [self.rotate_theta]
        self.parameters_fangshe += [self.Scale_factor]
        # self.corfficent_net = skip3D(1, 1,
        #                        num_channels_down=[16, 32, 64, 128, 128, 128],
        #                        num_channels_up=[16, 32, 64, 128, 128, 128],
        #                        num_channels_skip=[0, 0, 4, 4, 4, 4],
        #                        filter_size_up=3, filter_size_down=3, filter_skip_size=1,
        #                        upsample_mode='nearest',  # downsample_mode='avg',
        #                        need1x1_up=False,
        #                        need_sigmoid=False, need_bias=True, pad=pad, act_fun='LeakyReLU').type(data_type)
        # self.corfficent_net = skip3D(1, 1,
        #            num_channels_down=[16, 32, 64, 128, 128],
        #            num_channels_up=[16, 32, 64, 128, 128],
        #            num_channels_skip=[0, 0, 4, 4, 4],
        #            filter_size_up=3, filter_size_down=3, filter_skip_size=1,
        #            upsample_mode='nearest',  # downsample_mode='avg',
        #            need1x1_up=False,
        #            need_sigmoid=False, need_bias=True, pad=pad, act_fun='LeakyReLU').type(data_type)
        # self.parameters = [p for p in self.corfficent_net.parameters()] + self.parameters

        # self.device = 'cuda:{}'.format(0)
        # self.model = FFDNet(in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R').to(self.device)
        # self.model.load_state_dict(torch.load('pretrained_models/ffdnet_gray.pth'))
        # self.model.eval()

    def _init_loss(self):
        data_type = torch.cuda.FloatTensor
        self.mse_loss = torch.nn.MSELoss().type(data_type)
        self.SP_Loss = SPLoss().type(data_type)
        self.Nuclear_Loss = Nuclear_Loss().type(data_type)
        self.TVLoss = TVLoss().type(data_type)
        self.TV_b_Loss = TV_b_Loss().type(data_type)
        self.MyRELU = MyRELU().type(data_type)
        self.GrayLoss = GrayLoss().type(data_type)
        self.l1_loss = nn.L1Loss().type(data_type)

    def _init_inputs(self):
        # original_nois: 10*1*256*256
        original_noise = torch_to_np(
            get_noise1(1, 'noise', (1, 384, 384), var=1 / 10.).type(
                torch.cuda.FloatTensor).detach())
        # self.image_net_inputs: 10*1*191*191
        #scipy.io.savemat("input1.mat", {"original_noise": original_noise})
        self.image_net_inputs = np_to_torch(original_noise).type(torch.cuda.FloatTensor).detach()[0, :, :, :]
        # self.image_net_inputs = self.image_net_inputs  + (self.mask_net_inputs.clone().normal_() * 1/30)

        # original_noise: 10*1*256
        original_noise = torch_to_np(
            get_noise2(1, 'noise', self.image_original.shape[2], var=1 / 1.).type(torch.cuda.FloatTensor).detach())
        # self.mask_net_inputs: 10*1*256
        #scipy.io.savemat("input2.mat", {"original_noise": original_noise})
        self.mask_net_inputs = np_to_torch(original_noise).type(torch.cuda.FloatTensor).detach()[0, :, :, :]
        # self.mask_net_inputs:10*1*256
        self.mask_net_inputs = self.mask_net_inputs  # + (self.mask_net_inputs.clone().normal_() * 1/30)


        [a, b, c] = self.image_original.shape
        temp = np.random.rand(1, 1, 32, 256, 256)
        mask = temp < 1
        # original_noise = torch_to_np(
        #     get_noise3(1, 'noise', [self.image.shape[2], self.image.shape[0], self.image.shape[1]], var=1 / 10.).type(
        #         torch.cuda.FloatTensor).detach())
        # self.image_net_inputs: 10*1*191*191
        # scipy.io.savemat("input3.mat", {"original_noise": original_noise})
        self.move_net_inputs = np_to_torch(mask).type(torch.cuda.FloatTensor).detach()[0, :, :, :]

        self.move = torch.zeros(self.move_net_inputs.shape).type(torch.cuda.FloatTensor)
        # self.move = self.move.reshape((128, 128, 32))
        self.move_np = torch_to_np(self.move)

    def _init_all(self):
        self._init_images()
        self._init_nets()
        self._init_inputs()
        self._init_loss()


    def optimize(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        self.optimizer = torch.optim.Adam(self.parameters+self.parameters_fangshe, lr=0.001)
        self.optimizer_fangshe = torch.optim.Adam(self.parameters_fangshe, lr=0.0001)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[1000], gamma=0.1, last_epoch=-1)
        lr_scheduler_fangshe = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_fangshe, milestones=[3000], gamma=0.1, last_epoch=-1)
        for j in range(self.num_iter + 1):
            self.optimizer.zero_grad()
            self.optimizer_fangshe.zero_grad()
            self._optimization_closure(j)
            self._obtain_current_result(j)
            self._plot_closure(j)
            self.optimizer.step()
            self.optimizer_fangshe.step()
            lr_scheduler.step()
            lr_scheduler_fangshe.step()

    # def _optimization_closure_step1(self, step):
    #     self.corfficent_net_inputs = self.corfficent_net_inputs + (
    #                 self.corfficent_net_inputs.clone().normal_() * self.reg_std)
    #     out = self.corfficent_net(self.corfficent_net_inputs)
    #     self.total_loss = self.l1_loss(torch.ones_like(out) / 2, out)
    #     self.total_loss.backward()
    def _optimization_closure(self, step):

        ############image_out############################################################
        m = 0
        # self.image_net_inputs: 10*1*256*256   M: 10*1*256*256
        M = self.image_net_inputs + (self.image_net_inputs.clone().normal_() * self.reg_std)
        # out: 10*1*256*256
        out = self.image_net[0](M)
        for i in range(1, self.rank):
            out = torch.cat((out, self.image_net[i](M)), 0)
        # out: 10*1*256*256
        #out = out[:, :, :self.image.shape[0], :self.image.shape[1]]
        self.A = out[:, 0, :, :].permute(1, 2, 0)
        # self.image_out: 65536 10
        self.image_out = out[m, :, :, :].squeeze().reshape((-1, 1))
        for m in range(1, self.rank):
            self.image_out = torch.cat((self.image_out, out[m, :, :, :].squeeze().reshape((-1, 1))), 1)
        # self.image_out = self.MyRELU(self.image_out)
        # self.image_out:65536 10
        # self.image_out_np:65536 10
        self.image_out_np = torch_to_np(self.image_out)

        ######################mask_out################################################################################
        # M: 10*1*191
        M = self.mask_net_inputs + (self.mask_net_inputs.clone().normal_() * self.reg_std)
        # N: 10*1*191
        out = self.mask_net[0](M)
        for i in range(1, self.rank):
            out = torch.cat((out, self.mask_net[i](M)), 0)
        # self.mask_out: 10*191
        self.mask_out = out.squeeze(1)
        # self.mask_out = self.MyRELU(self.mask_out)
        self.mask_out_np = torch_to_np(self.mask_out)

        self.move_net_inputs = self.move_net_inputs + (self.move_net_inputs.clone().normal_() * self.reg_std)
        self.move = self.move_net(self.move_net_inputs).squeeze()
        frame_list = []
        for i in range(self.move.shape[0]):
            inp = torch.reshape(self.move[i, :, :], (256, 256))
            frame_list.append(inp)
        self.move = torch.stack(frame_list, dim=0)
        self.move_np = torch_to_np(self.move)
        ###################################################################################
        self.B = self.image_out.mm(self.mask_out)
        frame_list = []
        for i in range(self.B.shape[1]):
            inp = torch.reshape(self.B[:, i], (384, 384))
            frame_list.append(inp)
        self.B = torch.stack(frame_list, dim=0)
        self.B = affine_B(self.B.unsqueeze(0), self.x, self.y, self.rotate_theta, self.Scale_factor, 64).squeeze(0)
        self.image_com = self.B + self.move


        frame_list = []
        for i in range(self.image_com.shape[0]):
            frame_list.append(self.image_com[i, :, :])
        self.image_com = torch.stack(frame_list, dim=2)
        ####################################################################
        self.image_com_np = torch_to_np(self.image_com)
        # ######################Smoothing##########################################################################################
        self.out_avg = self.out_avg * self.exp_weight + self.image_com_np * (1 - self.exp_weight)
        self.res = np.sum(np.abs(self.out_avg - self.previous))/np.sum(self.previous)
        self.previous = self.out_avg
        a = self.image_com * self.mask_torch
        self.mea = torch.zeros((256, 256, 4)).type(torch.cuda.FloatTensor)
        for i in range(4):
            self.mea[:, :, i] = torch.sum(a[:, :, (i * 8):(i * 8 + 8)], 2)

        self.total_loss = self.mse_loss(self.mea, self.measurement_torch)
                           # + self.tau * (step // 100) * self.GrayLoss(self.M) + self.lamda * self.mse_loss(self.image_com, self.FFDNet_out) #+ self.beta * self.TVLoss(self.oup)
        self.total_loss.backward()

    def _obtain_current_result(self, step):
        self.psnr = quality(self.image_original, self.image_com_np.astype(np.float64))
        self.psnr_av = quality(self.image_original, self.out_avg.astype(np.float64))
        self.current_result = Result(recon=self.image_com_np, psnr=self.psnr, AB=self.image_out_np, c=self.mask_out_np, F=self.move_np)
        self.current_result_av = Result(recon=self.out_avg, psnr=self.psnr_av, AB=self.image_out_np, c=self.mask_out_np, F=self.move_np)
        if self.best_result is None or self.best_result.psnr < self.current_result.psnr:
            self.best_result = self.current_result
        if self.best_result_av is None or self.best_result_av.psnr < self.current_result_av.psnr:
            self.best_result_av = self.current_result_av

        self.max_psnr = self.best_result.psnr

        self.psnr_list.append(self.psnr)
        self.psnr_av_list.append(self.psnr_av)
        self.re_list.append(self.res)


    def _plot_closure_step1(self, step):
        print('Iteration %05d tol_loss %f' %(step, self.total_loss.item()))

    def _plot_closure(self, step):
        print('Iteration %05d  tol_loss %f    current_psnr: %f  max_psnr %f  current_psnr_av: %f max_psnr_av: %f  res: %f' % (
        step, self.total_loss.item(),
        self.current_result.psnr, self.best_result.psnr,
        self.current_result_av.psnr, self.best_result_av.psnr, self.res), '\r')

def dehaze(image_original, meas, num_iter, R, mask, tau, lamda, beta):
    for i in range(1):
        dh = Dehaze(image_original, meas, num_iter, R, mask, tau, lamda, beta)
        dh.optimize()
        psnr_list = np.array(dh.psnr_list)
        psnr_av_list = np.array(dh.psnr_av_list)
        res_list = np.array(dh.re_list)
        scio.savemat('./results/park_T.mat',
                     {'psnr_list': psnr_list, 'psnr_av_list': psnr_av_list, 'res_list': res_list, 'x': dh.best_result.recon, 'x_avg': dh.best_result_av.recon, 'psnr': dh.best_result.psnr,
                       'psnr_av': dh.best_result_av.psnr, 'F': dh.best_result.F, 'AB': dh.best_result.AB, 'c':dh.best_result.c})

if __name__ == "__main__":
    num_iter = 10000
    mat = scipy.io.loadmat("./data/park_3.mat")
    # 256 256 191
    image_original = mat["orig"]/1.0
    mask = mat["mask_full"]
    a = mask * image_original
    meas = np.zeros((256, 256, 4))
    for i in range(4):
        meas[:, :, i] = np.sum(a[:, :, (i * 8):(i * 8 + 8)], 2)
    # mask = np.reshape(mask, (mask.shape[0] * mask.shape[1], mask.shape[2]), order="F")
    # meas = np.reshape(meas, (meas.shape[0] * meas.shape[1], meas.shape[2]), order="F")
    for R in [4]:
        for lamda in [0]:
            for tau in [0]:
                for beta in [1]:
                    result = dehaze(image_original, meas, num_iter, R, mask, tau, lamda, beta)