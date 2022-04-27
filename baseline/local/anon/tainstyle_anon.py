import os
import argparse
import sys
from datetime import datetime, timedelta
import numpy as np
import torch
import torch.nn as nn
from weight_init import weight_init
from stye_anon_model import Discriminator, StyleEncoder, Generator
from utility import Normalizer, speakers
from preprocess import FRAMES, SAMPLE_RATE, FFTSIZE
import random
from pyworld import decode_spectral_envelope, synthesize
from torch.utils.data.dataloader import DataLoader, TensorDataset
import ast
import soundfile

def train_data_loader(speaker1_xvec, speaker2_xvec, gender=None, speaker_id=None, batch_size=8, shuffle=True, num_workers=10):
    tensor_dataset = TensorDataset(
        speaker1_xvec,
        speaker2_xvec
        )

    loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return loader

def test_data_loader(speaker1_xvec, gender=None, speaker_id=None, batch_size=8, shuffle=True, num_workers=10):
    tensor_dataset = TensorDataset(
        speaker1_xvec
        )

    loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return loader


class Solver(object):
    """docstring for Solver."""
    def __init__(self, train_data_loader, test_data_loader, config):

        self.config = config
        self.data_loader = data_loader

        # Model configurations.
        self.num_domain = 20 # number of style

        self.dim_latent = config.dim_latent
        self.dim_style = config.dim_style
        self.lambda_rec = config.lambda_rec
        self.lambda_style_rec = config.lambda_style_rec
        self.lambda_gp = config.lambda_gp
        self.mse_loss = nn.MSELoss()

        # Training configurations.
        self.data_dir = config.src_data
        self.test_dir = config.test_dir
        self.batch_size = 40
        self.epochs = 30

        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.s_lr = config.s_lr

        self.num_iters_decay = config.num_iters_decay
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.weight_init = config.weight_init

        # Test configurations.
        self.test_data_loader = test_data_loader

        # Miscellaneous.
        self.use_tensorboard = False
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = None
        self.sample_dir = 'Experiment/local/anon/sample_outputs/'
        self.model_save_dir = 'Experiment/local/anon/model_ckpt/'
        self.result_dir = config.xvec_out_dir

        # Step size.
        self.log_step = 300
        self.sample_step = 200
        self.model_save_step = 200
        self.lr_update_step = 200

    def build_optimizer(self):

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2], amsgrad=True)
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2], amsgrad=True)
        self.s_optimizer = torch.optim.Adam(self.S.parameters(), self.s_lr, [self.beta1, self.beta2], amsgrad=True)

    
    def build_model(self):
        self.G = Generator().to(self.device)
        self.D = Discriminator().to(self.device)
        self.S = StyleEncoder(num_domains=self.num_domain).to(self.device)

        if self.weight_init:
            self.G = weight_init(self.G)
            self.D = weight_init(self.D)
            self.S = weight_init(self.S)

        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
        self.print_network(self.S, 'S')

        # self.G = nn.DataParallel(self.G)
        # self.D = nn.DataParallel(self.D)
        # self.S = nn.DataParallel(self.S)

    def model_state(self, state):
        if state == 'train':
            self.G.train()
            self.D.train()
            self.S.train()

        elif state == "test":
            self.G.eval()
            self.D.eval()
            self.S.eval()

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator"""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def train(self):
        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        start_iters = 0
        # Build the model and tensorboard.
        self.build_model()

        # Build the optimizer.
        self.build_optimizer()

        norm = Normalizer()
        data_iter = iter(self.data_loader)

        print('Start training......')
        print("num_domain = {}".format(self.num_domain))
        start_time = datetime.now()

        for i in range(start_iters, self.epochs):
            self.model_state(state='train')
            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch labels.
            try:
                p1,  p2 = next(data_iter)

            except:
                data_iter = iter(self.data_loader)
                p1, p2 = next(data_iter)

            p1 = p1.to(self.device)
            p2 = p2.to(self.device)

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # use WGAN_GP critic

            # D adversarial loss
            out_src1 = self.D(p1)
            d_loss_real1 = -torch.mean(out_src1)

            style2 = self.S(p2)
            p1_2 = self.G(p1, style2)
            out_src2 = self.D(p1_2.detach())
            d_loss_fake2 = torch.mean(out_src2)

            out_src2 = self.D(p2)
            d_loss_real2 = -torch.mean(out_src2)

            style1 = self.S(p1)
            p2_1 = self.G(p2, style1)
            out_src1 = self.D(p2_1.detach())
            d_loss_fake1 = torch.mean(out_src1)

            d_loss_real = d_loss_real1 + d_loss_real2
            d_loss_fake = d_loss_fake1 + d_loss_fake2

            # Compute loss for gradient penalty.
            alpha = torch.rand(p1.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * p1.data + (1 - alpha) * p1_2.data).requires_grad_(True)
            out_src = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/d_loss'] = d_loss.item()


            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i + 1) % self.n_critic == 0:

                style1 = self.S(p1)
                style2 = self.S(p2)

                p1_2 = self.G(p1, style2)
                p2_1 = self.G(p2, style1)

                # G adversarial loss
                out_src2 = self.D(p1_2)
                g_loss_fake2 = -torch.mean(out_src2)

                out_src1 = self.D(p2_1)
                g_loss_fake1 = -torch.mean(out_src1)

                g_loss_fake = g_loss_fake1 + g_loss_fake2

                # Reconstruction(Cycle)
                p1_2_1 = self.G(p1_2, style1)
                g_loss_rec1 = self.mse_loss(p1_2_1, p1)

                p2_1_2 = self.G(p2_1, style2)
                g_loss_rec2 = self.mse_loss(p2_1_2, p2)

                g_loss_rec = g_loss_rec1 + g_loss_rec2

                # Style reconstruction
                style1_2 = self.S(p1_2)
                style_rec1 = torch.mean(torch.abs(style1_2 - style2))

                style2_1 = self.S(p2_1)
                style_rec2 = torch.mean(torch.abs(style2_1 - style1))

                style_rec = style_rec1 + style_rec2

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_style_rec * style_rec

                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()
                self.s_optimizer.step()

                # Logging.
                loss['G/g_loss'] = g_loss.item()
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/style_rec'] = style_rec.item()


            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #
            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = datetime.now() - start_time
                et = str(et)[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.epochs)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

            # Decay learning rates.
            if (i + 1) % self.lr_update_step == 0 and (i + 1) > (self.epochs - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}'.format(g_lr, d_lr))
        
                # Save model checkpoints.

        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i + 1))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i + 1))
        S_path = os.path.join(self.model_save_dir, '{}-S.ckpt'.format(i + 1))

        torch.save(self.G.state_dict(), G_path)
        torch.save(self.D.state_dict(), D_path)
        torch.save(self.S.state_dict(), S_path)

        print('Saved model checkpoints into {}...'.format(self.model_save_dir))

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.s_optimizer.zero_grad()

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        S_path = os.path.join(self.model_save_dir, '{}-S.ckpt'.format(resume_iters))

        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        self.S.load_state_dict(torch.load(S_path, map_location=lambda storage, loc: storage))


    @staticmethod
    def pad_coded_sp(coded_sp_norm):
        f_len = coded_sp_norm.shape[1]
        if  f_len >= FRAMES: 
            pad_length = FRAMES-(f_len - (f_len//FRAMES) * FRAMES)
        elif f_len < FRAMES:
            pad_length = FRAMES - f_len

        sp_norm_pad = np.hstack((coded_sp_norm, np.zeros((coded_sp_norm.shape[0], pad_length))))
        return sp_norm_pad 

    def test(self):
        """Translate speech using StarGAN ."""

        # Load the trained generator.
        self.build_model()
        self.restore_model(self.test_iters)
        self.model_state(state='test')

        # norm = Normalizer()

        # Set data loader.
        targets = self.test_data_loader

        for target in targets:
            d, trg_mel, src_speaker, trg_speaker = TestSet(self.test_dir).test_data(self.src_speaker, target)

            assert trg_speaker in speakers

            with torch.no_grad():

                trg_mel = torch.FloatTensor(trg_mel).to(self.device)
                trg_mel = trg_mel.view(1, 1, trg_mel.size(0), trg_mel.size(1))
                trg_mel = trg_mel.to(self.device)
                style = self.S(trg_mel)

                for filename, content in d.items():
                    f0 = content['f0']
                    ap = content['ap']
                    sp_norm_pad = self.pad_coded_sp(content['coded_sp_norm'])

                    convert_result = []
                    for start_idx in range(0, sp_norm_pad.shape[1] - FRAMES + 1, FRAMES):
                        one_seg = sp_norm_pad[:, start_idx : start_idx+FRAMES]
                        
                        one_seg = torch.FloatTensor(one_seg).to(self.device)
                        one_seg = one_seg.view(1,1,one_seg.size(0), one_seg.size(1))
                        one_seg = one_seg.to(self.device)
                        one_set_return = self.G(one_seg, style).data.cpu().numpy()# GPU->CPU Tensor->Numpy
                        one_set_return = np.squeeze(one_set_return)
                        one_set_return = norm.backward_process(one_set_return, trg_speaker)
                        convert_result.append(one_set_return)

                    convert_con = np.concatenate(convert_result, axis=1)
                    convert_con = convert_con[:, 0:content['coded_sp_norm'].shape[1]]
                    contigu = np.ascontiguousarray(convert_con.T, dtype=np.float64)   
                    decoded_sp = decode_spectral_envelope(contigu, SAMPLE_RATE, fft_size=FFTSIZE)
                    f0_converted = norm.pitch_conversion(f0, src_speaker, trg_speaker)
                    wav = synthesize(f0_converted, decoded_sp, ap, SAMPLE_RATE)

                    name = f'{src_speaker}-{trg_speaker}_iter{self.test_iters}_{filename}'
                    path = os.path.join(self.result_dir, name)
                    print(f'[save]:{path}')
                    # librosa.output.write_wav(path, wav, SAMPLE_RATE)
                    soundfile.write(path, wav, SAMPLE_RATE)

if __name__ == '__main__':
    print(sys.argv)
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('src_data')
    parser.add_argument('pool_data')
    parser.add_argument('xvec_out_dir')
    parser.add_argument('pseudo_xvecs_dir')
    parser.add_argument('src_xvec_dir')
    parser.add_argument('rand_level', choices=['utt', 'spk'])
    parser.add_argument('cross_gender', choices=['true', 'false'])
    parser.add_argument('random_seed', default=2020, type=int)
    parser.add_argument('--combine_genders', choices=['true', 'false'], default='false',
                        help='Flag to construct just one GMM for both genders')
    parser.add_argument('--pickle_file', help="Pickle file location to support reuse. Use 'None' to disable or leave blank")

    args = parser.parse_args()
    src_data = args.src_data
    pool_data = args.pool_data
    xvec_out_dir = args.xvec_out_dir
    pseudo_xvecs_dir = args.pseudo_xvecs_dir
    src_xvec_dir = args.src_xvec_dir
    rand_level = args.rand_level
    cross_gender = True if args.cross_gender == 'true' else False
    random_seed = args.random_seed
    pca_parameter = args.pca_size
    combine_genders = True if args.combine_genders == 'true' else False
