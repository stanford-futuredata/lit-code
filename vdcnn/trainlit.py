import yaml
import argparse
from tqdm import tqdm
import torch

import os
import torch.nn as nn
import torch.optim
import torch.utils.data

from collections import OrderedDict
from lit import LitTrainer, Section
from sentiment_datasets import *
import torchvision.transforms as transforms
import json
import vdcnn
from vdcnn import MODELS
from distillation_loss import AlphaDistillationLoss

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='LIT: Lightweight Iterative Training Module Using VDCNN')
parser.add_argument('-c', '--config', type=str, metavar='PATH', help='Path to config files for the experiment', action='append', required=True)

parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--start_segment', default=0, type=int, metavar='N', help='manual section number to start on: 0:LIT, 1:Outer layers Fine Tuning, 2:Full Model Fine Tuning')
parser.add_argument('--resume', action='store_true', help = "Flag wether you want to resume training or not")
args = parser.parse_args()
args.half = False

def main():
    if (args.start_epoch != 0) or (args.start_segment != 0):
        assert(len(args.config) == 1)
    for config_file in args.config:
        config = yaml.safe_load(open(config_file))
        trainer = LitTrainer(Builder(config))
        if not config['params']['evaluate']:
            trainer.train()
        t_loss, t_acc1 = trainer.validate_model()
        tqdm.write("\nSUMMARY")
        tqdm.write("#######################")
        tqdm.write("Learner Model Stats:")
        tqdm.write("\tLoss {i}".format(i=t_loss))
        tqdm.write("\tAccuracy {i}".format(i=t_acc1))
        tqdm.write("#######################")
        trainer.close_log_writer()


class Builder():
    def __init__(self, config):
        global args
        self.dataset = "amazon_review_full"
        self.lit_criterion = nn.MSELoss

        self.loss_scaling = 1
        self.resume = False
        self.half = False
        self.start_epoch = 0
        self.start_segment = 0

        model_config = config["models"]
        hyperparameter_config = config["hyperparameters"]
        param_config = config["params"]
        self.set_start_location(start_epoch=args.start_epoch, start_segment=args.start_segment, resume=args.resume)
        if args.half:
            self.use_half_precision()
        self.set_paths(save_dir=param_config["save_directory"], log_dir=param_config["logdir"], save_model_name=param_config["save_model_name"])

        self.set_data_loaders(batch_size=hyperparameter_config["batch_size"],
                                 workers= hyperparameter_config["data_loading_workers"],
                                 dataset = model_config["dataset"])

        self.make_trainable_model(arch=model_config["training_architecture"],
                                    training_checkpoint= model_config["training_checkpoint"])

        self.load_base_model(base_arch=model_config["base_architecture"],
                                    base_model_path=model_config["base_model_path"])

        self.set_hyperparameters(sequence= config["sequence"],
                                    momentum= hyperparameter_config["momentum"],
                                    weight_decay=hyperparameter_config["weight_decay"],
                                    save_every=param_config["save_every"],
                                    alpha=hyperparameter_config["alpha"],
                                    temperature=hyperparameter_config["temperature"],
                                    lit_beta=hyperparameter_config["beta"])
        self.make_lit_sections()
        self.copy_equivalent_layers()

    def use_half_precision(self):
        self.half = True

    def set_paths(self, save_dir, log_dir, save_model_name):
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.model_name = save_model_name

    def set_hyperparameters(self, sequence, momentum, weight_decay, save_every=1, alpha= 0.95, temperature=6.0, lit_beta=0.5):
        self.sequence = sequence
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.save_every = save_every
        self.fine_tuning_criterion = lambda : AlphaDistillationLoss(temperature=temperature, alpha = alpha, scale = True)
        self.beta = lit_beta

    def set_start_location(self, start_epoch, start_segment, resume = False):
        self.start_epoch = start_epoch
        self.start_segment = start_segment
        self.resume = resume

    def make_trainable_model(self, arch, training_checkpoint = None):
        assert self.val_data_loader
        print("Making Trainable VDCNN Model")
        num_classes = self.val_data_loader.dataset.classes
        trainable_model = MODELS[arch](num_classes = num_classes)
        if training_checkpoint and os.path.isfile(training_checkpoint):
            tqdm.write("=> loading trainable model '{}'".format(training_checkpoint))
            trainable_model.load_state_dict(torch.load(training_checkpoint))
        elif self.resume:
            path = os.path.join(self.save_dir, str("checkpoint_" + self.model_name))
            tqdm.write("=> loading trainable model '{}'".format(path))
            trainable_model.load_state_dict(torch.load(path))
        self.trainable_model = trainable_model

    def load_base_model(self, base_arch, base_model_path):
        assert self.val_data_loader
        print("Making Base VDCNN Model")
        num_classes = self.val_data_loader.dataset.classes
        base_model = MODELS[base_arch](num_classes = num_classes)
        tqdm.write("=> loading base model '{}'".format(base_model_path))
        try:
            base_model.load_state_dict(torch.load(base_model_path)["model"])
        except:
            base_model.load_state_dict(torch.load(base_model_path))
        self.base_model = base_model

    def set_data_loaders(self, batch_size, workers, root = "data", dataset = "amazon_review_full"):
        root = os.path.join(root, str(dataset + "_csv"))
        vocab = Vocab("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/|_#$%ˆ&*~`+=<>()[]{} ",  # noqa: E501
            offset=2, unknown=1)
        dset_transforms = transforms.Compose([
            transforms.Lambda(lambda doc: doc.lower()),
            vocab,
            PadOrTruncate(1014),
            transforms.Lambda(lambda doc: doc.astype(np.int64))
        ])
        train_dataset = DATASETS[dataset](root = root, train = True, transform = dset_transforms)
        test_dataset = DATASETS[dataset](root=root, train=False, transform=dset_transforms)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    shuffle=True,
                                                    batch_size=batch_size,
                                                    num_workers=workers,
                                                    pin_memory=True)

        val_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=workers,
                                                pin_memory=True)

        self.fine_tuning_data_loader = DataPrefetcher(train_loader)
        self.lit_training_data_loader = self.fine_tuning_data_loader
        self.val_data_loader = DataPrefetcher(val_loader)

    def make_lit_sections(self):
        assert (self.trainable_model and self.base_model)
        assert (isinstance(self.trainable_model, vdcnn.VDCNN))
        self.lit_sections = OrderedDict()
        #Set up equivalences from the feature map number to the actual feature map
        self.lit_sections[1] = Section(self.trainable_model.sections.section_0)
        self.lit_sections[2] = Section(self.trainable_model.sections.section_1)
        self.lit_sections[3] = Section(self.trainable_model.sections.section_2)
        self.lit_sections[4] = Section(self.trainable_model.sections.section_3)

    def copy_equivalent_layers(self):
        assert (self.trainable_model and self.base_model and self.lit_sections)
        self.trainable_model.character_embedding.load_state_dict(self.base_model.character_embedding.state_dict())
        self.trainable_model.conv1.load_state_dict(self.base_model.conv1.state_dict())
        self.trainable_model.kmax.load_state_dict(self.base_model.kmax.state_dict())
        self.trainable_model.fc1.load_state_dict(self.base_model.fc1.state_dict())
        self.trainable_model.fc2.load_state_dict(self.base_model.fc2.state_dict())
        self.trainable_model.fc3.load_state_dict(self.base_model.fc3.state_dict())

if __name__ == '__main__':
    main()
