
# Copyright (C) 2020 <Henry Turner, Giulio Lovisotto, Ivan Martinovic>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#


import sys
from os.path import basename, join
import os
from datetime import datetime, timedelta
import operator
import argparse
# from Experiment.local.anon.tainstyle_anon import train_data_loader
import numpy as np
import random
from munch import Munch
from kaldiio import WriteHelper, ReadHelper
from sklearn import decomposition
from sklearn import metrics
from sklearn import mixture
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import BCEWithLogitsLoss
from stye_anon_model import Discriminator, StyleEncoder, Generator, MappingNetwork
from utility import Normalizer, speakers
import random

from core.solver import Solver

#############################################################################################################
class Model_arguments(object):
    def __init__(self, batch_size):
        self.batch_size=batch_size
        self.xvec_size=512
        self.num_domains=2
        self.latent_dim=16
        self.hidden_dim=512 #'Hidden dimension of mapping network
        self.style_dim=256 #'Style code dimension

        # weight for objective functions
        self.lambda_reg=1 #Weight for R1 regularization
        self.lambda_cyc=1 #Weight for cyclic consistency loss
        self.lambda_sty=1 #Weight for style reconstruction loss
        self.lambda_ds=1 #Weight for diversity sensitive loss
        self.ds_iter=10000 #Number of iterations to optimize diversity sensitive loss
        # training arguments
        self.total_iters=120000 #Number of total iterations

        # ******** set the following to zero if train from scratch
        self.resume_iter=120000 #Iterations to resume training/testing


        self.batch_size=50 #Batch size for training')
        self.lr=1e-4/4 #Learning rate for D, E and G')
        self.f_lr=1e-6/4 #Learning rate for F')
        self.beta1=0.0 #Decay rate for 1st moment of Adam
        self.beta2=0.99 #Decay rate for 2nd moment of Adam
        self.weight_decay=1e-4 #Weight decay for optimizer
        self.num_outs_per_domain=10 #Number of generated images per domain during sampling

        # misc
        self.mode='train' #choices=['train', 'sample'] 'This argument is used in solver
        self.num_workers=4 #'Number of workers used in DataLoader
        self.seed=777 #'Seed for random number generator

        # directory for training
        self.checkpoint_dir='local/anon/model_ckpt2/' #Directory for saving network checkpoints
        self.result_dir = 'local/anon/sample_outputs/'

        # step size
        self.print_every=50
        self.save_every=10000
        self.parser=10000
        self.w_hpf=0

#############################################################################################################
def load_src_spk_files(src_data):
    # assign new xvectors
    src_spk2gender_file = join(src_data, 'spk2gender')
    src_spk2utt_file = join(src_data, 'spk2utt')

    # Read source spk2gender and spk2utt
    src_spk2gender = {}
    src_spk2utt = {}
    print("Reading source spk2gender.")

    if not os.path.exists(src_spk2gender_file):
        raise ValueError("{} does not exist!".format(src_spk2gender_file))
    with open(src_spk2gender_file) as f:
        for line in f.read().splitlines():
            sp = line.split()
            src_spk2gender[sp[0]] = sp[1]
    print("Reading source spk2utt.")

    if not os.path.exists(src_spk2utt_file):
        raise ValueError("{} does not exist!".format(src_spk2utt_file))

    with open(src_spk2utt_file) as f:
        for line in f.read().splitlines():
            sp = line.split()
            src_spk2utt[sp[0]] = sp[1:]

    
    pd.DataFrame.from_dict(src_spk2gender, orient='index').to_csv("original_gender.csv")
    return src_spk2gender, src_spk2utt

def write_new_xvectors(pseudo_xvecs_dir, pseudo_xvec_map):
    # Write features as ark,scp
    print("Writing pseud-speaker xvectors to: "+pseudo_xvecs_dir)
    ark_scp_output = 'ark,scp:{}/{}.ark,{}/{}.scp'.format(
        pseudo_xvecs_dir, 'pseudo_xvector',
        pseudo_xvecs_dir, 'pseudo_xvector')

    with WriteHelper(ark_scp_output) as writer:
        for uttid, xvec in pseudo_xvec_map.items():
            # print("uttid", uttid)
            # print("xvec", xvec)
            writer(uttid, xvec)

def load_xvecs(xvec_file):
    original_xvecs = {}
    # Read source original xvectors.
    with ReadHelper('scp:' + xvec_file) as reader:
        for key, xvec in reader:
            # print key, mat.shape
            original_xvecs[key] = xvec
            # print(len(xvec))
    return original_xvecs

def write_new_spk2gender(pseudo_xvecs_dir, pseudo_gender_map):

    print("Writing pseudo-speaker spk2gender.")
    with open(join(pseudo_xvecs_dir, 'spk2gender'), 'w') as f:
        spk2gen_arr = [spk+' '+gender for spk,
                    gender in pseudo_gender_map.items()]
        sorted_spk2gen = sorted(spk2gen_arr)
        f.write('\n'.join(sorted_spk2gen) + '\n')

def prepare_data_loader(dataset, batch_size, domain, sample_size=35000, random=True):
    num_of_data = len(dataset)
    print("Number of data: ",len(dataset))
    
    xvector1_idx = np.repeat(range(num_of_data),num_of_data)
    xvector2_idx = np.tile(range(num_of_data),num_of_data)
    indexes = pd.DataFrame({'xvector1_idx':xvector1_idx, 'xvector2_idx':xvector2_idx})
    indexes = indexes[indexes['xvector1_idx']<indexes['xvector2_idx']]

    dataset = np.array(dataset)
    domain = np.array(domain)
    print('shape: ', np.shape)
    xvector1 = torch.tensor(dataset[indexes['xvector1_idx']],dtype=torch.float)
    xvector2 = torch.tensor(dataset[indexes['xvector2_idx']],dtype=torch.float)
    xvector3 = torch.tensor(dataset[np.random.choice(num_of_data, len(indexes['xvector2_idx']), replace=True)],dtype=torch.float)
    
    domain1 = torch.tensor(domain[indexes['xvector1_idx']],dtype=torch.long)
    domain2 = torch.tensor(domain[indexes['xvector2_idx']],dtype=torch.long)
    print("-----------------------------------------------")

    print(xvector1.shape)
    print(xvector2.shape)

    tensor_dataset = TensorDataset(
            xvector1,
            xvector2,
            domain1,
            domain2,
            xvector3
            )
    if random:
        sampler = torch.utils.data.RandomSampler(
            tensor_dataset, replacement=False, num_samples=sample_size)
    else:
        sampler = torch.utils.data.SequentialSampler(tensor_dataset)
    return DataLoader(tensor_dataset, batch_size=batch_size, sampler=sampler, drop_last=True)

def prepare_style_dataloader(dataset, batch_size, shuffle=False):    
    tensor_dataset = TensorDataset(
            torch.tensor(dataset,dtype=torch.float)
            )

    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

def generate_new_xvectors(transforms, original_xvecs,
                            src_spk2gender, src_spk2utt, 
                            cross_gender, threshold, 
                            rand_level='spk',
                            anon_pool=None):
    # store the new xvector, gender for the speaker
    pseudo_xvec_map = {}
    pseudo_gender_map = {}

    # transforms.model_state("test")
    # transforms.model_to_device()
    selected_transform = transforms
        
    for spk, gender in src_spk2gender.items():
        original_xvec = [np.array(original_xvecs[spk])]

        # If we are doing cross-gender VC, reverse the gender else gender remains same
        if cross_gender and np.random.random_sample()>0.5:
            gender_rev = {'m': 'f', 'f': 'm'}
            gender = gender_rev[gender]


        # the new gender of the speaker
        pseudo_gender_map[spk] = gender
        gender_mapping = {'m': 0, 'f': 1}
        if rand_level == 'spk':
            # For rand_level = spk, one xvector is assigned to all the utterances
            # of a speaker
            pseudo_xvec = generate_anonymized_xvectors(
                transforms=selected_transform, original_xvec=original_xvec,
                gender=gender_mapping[gender],distance_threshold=threshold,anon_pool=anon_pool)
            # Assign it to all utterances of the current speaker
            for uttid in src_spk2utt[spk]:
                pseudo_xvec_map[uttid] = pseudo_xvec

        elif rand_level == 'utt':
            # For rand_level = utt, random xvector is assigned to all the utterances
            # of a speaker
            for uttid in src_spk2utt[spk]:
                # Compute random vector for every utt
                pseudo_xvec = generate_anonymized_xvectors(
                    transforms=selected_transform, original_xvec=original_xvec,
                    gender=gender_mapping[gender],distance_threshold=threshold,anon_pool=anon_pool)
                # Assign it to all utterances of the current speaker
                pseudo_xvec_map[uttid] = pseudo_xvec
        else:
            print("rand_level not supported! Errors will happen!")

    return pseudo_xvec_map, pseudo_gender_map

def generate_anonymized_xvectors(transforms, original_xvec, gender, distance_threshold, anon_pool=None):
    similarity = np.inf
    count = 0
    while similarity > distance_threshold and count<10:
        new_xvect = transforms.sample_xvect(original_xvec, torch.tensor([gender],dtype=torch.long),anon_pool).squeeze().tolist()
        similarity = cosine_similarity(np.array(new_xvect).reshape(1, -1), 
                            original_xvec[0].reshape(1, -1)).mean()
        print("Xvect similarity:", similarity)
        count += 1
        # print("original_xvec", original_xvec)
        # print("new_xvect", new_xvect)

    return np.array(new_xvect)
    

def train_models(pool_data, xvec_out_dir, batch_size, resume_training=False, pickle_file=None):
     # Load and assemble all of the xvectors from the pool sources
    print("pool_data:", pool_data)
    pool_data_sources = os.listdir(pool_data)
    pool_data_sources = [x for x in pool_data_sources if os.path.isdir(
        join(pool_data, x)) and os.path.isfile(os.path.join(pool_data, x, 'wav.scp'))]
    print("pool_data_sources:", pool_data_sources)
    gender_pools = {'m': [], 'f': []}
    gender_mapping = {'m': 0, 'f': 1}
    xvector_pool = []
    xvector_gender = []

    for pool_source in pool_data_sources:
        print('Adding {} to the pool'.format(join(pool_data, pool_source)))
        pool_spk2gender_file = join(pool_data, pool_source, 'spk2gender')

        # Read pool spk2gender
        pool_spk2gender = {}
        with open(pool_spk2gender_file) as f:
            for line in f.read().splitlines():
                sp = line.split()
                pool_spk2gender[sp[0]] = sp[1]

        # Read pool xvectors
        pool_xvec_file = join(xvec_out_dir, 'xvectors_'+pool_source,
                              'spk_xvector.scp')
        if not os.path.exists(pool_xvec_file):
            raise ValueError(
                'Xvector file: {} does not exist'.format(pool_xvec_file))
    
        with ReadHelper('scp:'+pool_xvec_file) as reader:
            for key, xvec in reader:
                xvector_pool.append(xvec)
                gender = pool_spk2gender[key]
                xvector_gender.append(gender_mapping[gender])
                gender_pools[gender].append(xvec)

    print("Read ", len(gender_pools['m']), " male pool xvectors")
    print("Read ", len(gender_pools['f']), " female pool xvectors")

    # Fit and train
    train_data_loader = loaders = Munch(
        src=prepare_data_loader(xvector_pool, batch_size, xvector_gender))
    model_arg = Model_arguments(batch_size)
    if resume_training:
        # transforms = load_model(pickle_file)
        transforms = Solver(model_arg)
        # transforms.gnerate()
    else:
        transforms = Solver(model_arg)
    transforms.train(train_data_loader)
    
    return transforms, gender_pools

def load_model(pickle_file):
    print("Loading existing models:{}".format(pickle_file))
    return pickle.load(open(pickle_file, 'rb'))

if __name__ == "__main__":
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
    parser.add_argument('--threshold', default=0.85, type=float,
                        help='Threshold to repeat random x vector if voice too similar')
    parser.add_argument('--combine_genders', choices=['true', 'false'], default='true',
                        help='Flag to construct just one Generator for both genders (Better to select True)')
    parser.add_argument('--pickle_file', help="Pickle file location to support reuse. Use 'None' to disable or leave blank")

    args = parser.parse_args()
    batch_size=50

    src_data = args.src_data
    pool_data = args.pool_data
    xvec_out_dir = args.xvec_out_dir
    pseudo_xvecs_dir = args.pseudo_xvecs_dir
    src_xvec_dir = args.src_xvec_dir
    rand_level = args.rand_level
    cross_gender = True if args.cross_gender == 'true' else False
    random_seed = args.random_seed
    combine_genders = True if args.combine_genders == 'true' else False
    threshold = args.threshold

    pickle_file = 'data/mymodel/models3.pickle'
    if pickle_file == 'None':
        pickle_file = None

    if cross_gender:
        print("**Opposite gender speakers will be selected.**")
    else:
        print("**Same gender speakers will be selected.**")


    #data to be anonymised
    src_spk2gender, src_spk2utt = load_src_spk_files(src_data)
    original_xvecs = load_xvecs(xvec_file=src_xvec_dir + '/spk_xvector.scp')

    print("----------------------------------------------------")

    print('pickle_file: ',pickle_file)
    style_file = 'style_pool_f.csv'

    retrain_model = False
    use_anon_pool = False
    xvector_pool = None
    # cross_gender = True
    
    if not retrain_model and pickle_file is not None and os.path.exists(pickle_file) and os.path.exists(style_file):
        transforms = load_model(pickle_file)

        if use_anon_pool: 
            xvector_pool={}
            xvector_pool['f'] = pd.read_csv("xvec_pool_f.csv",index_col=0).to_numpy(dtype=np.float64)
            xvector_pool['m'] = pd.read_csv("xvec_pool_m.csv",index_col=0).to_numpy(dtype=np.float64)
    else:
        transforms, xvector_pool = train_models(
            pool_data, xvec_out_dir, batch_size, 
            pickle_file=pickle_file)

        pd.DataFrame(xvector_pool['f']).to_csv("xvec_pool_f.csv")
        pd.DataFrame(xvector_pool['m']).to_csv("xvec_pool_m.csv")

        if not use_anon_pool:
            xvector_pool=None

    pd.DataFrame(original_xvecs).to_csv("origin_xvec.csv")

    pseudo_xvec_map, pseudo_gender_map = generate_new_xvectors(
                            transforms, original_xvecs,
                            src_spk2gender, src_spk2utt, 
                            cross_gender, threshold, 
                            rand_level='spk', anon_pool=xvector_pool)
    
    # pd.DataFrame.from_dict(pseudo_xvec_map, orient='index').to_csv("new_xvec.csv")
    # pd.DataFrame.from_dict(pseudo_gender_map, orient='index').to_csv("new_gender.csv")

    write_new_xvectors(pseudo_xvecs_dir, pseudo_xvec_map)
    write_new_spk2gender(pseudo_xvecs_dir, pseudo_gender_map)