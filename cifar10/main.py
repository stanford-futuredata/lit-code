import argparse
import numpy as np
import random
import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.cuda
import torch.optim
import torch.utils.data

import dataloaders
import resnet
from distillation_loss import AlphaDistillationLoss

#################
# Model Specs
teacher_arch = "resnet110"
teacher_model_checkpoint = "./models/resnet110.pth"
student_arch = "resnet20"
NB_SECTIONS = 3

# LIT Hyperparams
beta = 0.75
NOISE = False

# LIT Training
lit_lr = 0.1
lit_schedule = [100]
lit_epochs = 175

# Finetuning
finetune_starting_lr = 0.01
finetune_schedule = [55]
finetuning_epochs = 125

# KD Hyperparams
alpha = 0.95
temp = 6.0

# Rest of the hyperparams
momentum = 0.9
weight_decay = 0.0001
batch_size = 32
##################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=-1, type=int)
    args = parser.parse_args()

    # Determinism
    if args.seed >= 0:
        seed = args.seed
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    train_loader, val_loader = get_data_loaders()
    print("Loaded Data")

    teacher, student = setup_teacher_student(teacher_model_checkpoint)
    print("Loaded and Created Teacher + Student Models")

    ir_loss = torch.nn.MSELoss()
    kd_loss = AlphaDistillationLoss(temperature=temp, alpha=alpha)

    max_acc = 0.0

    # LIT Training
    optimizer = torch.optim.SGD(
            student.parameters(), lr=lit_lr,
            momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=lit_schedule)
    pbar = tqdm.trange(lit_epochs)
    for epoch in pbar:
        train_loss, train_acc1 = lit_epoch(
                teacher, student, train_loader, optimizer, scheduler, 
                ir_loss, kd_loss, train=True)
        val_loss, val_acc1 = lit_epoch(
                teacher, student, val_loader, optimizer, scheduler,
                ir_loss, kd_loss, train=False)
        if max_acc < val_acc1:
            torch.save(student.state_dict(), 'student.pth')
        max_acc = max(val_acc1, max_acc)
        pbar.set_description(
                'Train loss, acc: {:.2f}, {:.2f}, Val loss, acc: {:.2f}, {:.2f}, max_acc: {:.2f}'.format(
                        train_loss, train_acc1, val_loss, val_acc1, max_acc))
    print('Max acc so far: {:.2f}'.format(max_acc))

    # Fine Tuning
    optimizer = torch.optim.SGD(student.parameters(), lr=finetune_starting_lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=finetune_schedule)
    for epoch in range(finetuning_epochs):
        train_loss, train_acc1 = fine_tune_epoch(teacher, student, optimizer, scheduler, train_loader, kd_loss, train=True)
        val_loss, val_acc1 = fine_tune_epoch(teacher, student, optimizer, scheduler, val_loader, kd_loss, train=False)
        max_acc = max(val_acc1, max_acc)
        print("Fine Tuning Epoch [{}/{}]\t".format(epoch, finetuning_epochs) +
              "Train Loss {:.2f}\tTrain Acc1 {:.2f}\t".format(train_loss, train_acc1) +
              "Val Loss {:.2f}\tVal Acc1 {:.2f}".format(val_loss, val_acc1))
    print('Max acc so far: {:.2f}'.format(max_acc))


def fine_tune_epoch(teacher, student, optimizer, scheduler, loader, kd_loss, train=True):
    teacher.eval()
    if train:
        student.train()
        scheduler.step()
    else:
        student.eval()

    losses = AverageMeter()
    accuracies1 = AverageMeter()

    for i, (inp, target) in enumerate(loader):
        target = target.cuda(non_blocking=True)
        inp = inp.cuda().detach()

        with torch.no_grad():
            teacher_out = teacher(inp)

        with torch.set_grad_enabled(train):
            student_out = student(inp)
            loss = kd_loss(student_out, teacher_out, target)
            if train:
                loss.backward()
                optimizer.step()
                student.zero_grad()

        with torch.no_grad():
            prec1, prec5 = accuracy(student_out, target, topk=(1, 5))
            losses.update(loss.item(), inp.size(0))
            accuracies1.update(prec1[0], inp.size(0))

    return losses.avg, accuracies1.avg


def get_section(student, s_idx):
    return getattr(student, 'layer{}'.format(s_idx + 1))


def lit_epoch(teacher, student, loader,
              optimizer, scheduler, ir_loss, kd_loss, train=True):
    teacher.eval()
    if train:
        scheduler.step()
        student.eval()
        for p in student.parameters():
            p.requires_grad = False # set everything to false
        for s_idx in range(NB_SECTIONS):
            section = get_section(student, s_idx)
            section.train()
            for p in section.parameters():
                p.requires_grad = True
    else:
        student.eval()

    full_loss_log = AverageMeter()
    accuracies1 = AverageMeter()

    for i, (inp, target) in enumerate(loader):
        target = target.cuda(non_blocking=True)
        inp = inp.cuda(non_blocking=True).detach()

        with torch.no_grad():
            # Get the teacher intermediate reps
            features, soft_targets = teacher(inp, get_features=True)

        with torch.set_grad_enabled(train):
            # Do the full backward pass
            student_out = student(inp)
            full_loss = kd_loss(student_out, soft_targets, target) * beta

            if train:
                # full_loss.backward(retain_graph=True)
                # Now do section wise backwards
                for s_idx in range(NB_SECTIONS):
                    if NOISE:
                        noise = features[s_idx].data.new(features[s_idx].size()).normal_(0.0, 0.1)
                        sinp = features[s_idx] + noise
                    else:
                        sinp = features[s_idx]
                    section_out = get_section(student, s_idx)(sinp)
                    section_loss = ir_loss(section_out, features[s_idx + 1])
                    full_loss += section_loss * (1.0 - beta)
                full_loss /= 2
                full_loss.backward()
                optimizer.step()
                student.zero_grad()

        with torch.no_grad():
            prec1, prec5 = accuracy(student_out, target, topk=(1, 5))
            full_loss_log.update(full_loss.item(), inp.size(0))
            accuracies1.update(prec1[0], inp.size(0))

    return full_loss_log.avg, accuracies1.avg



def get_data_loaders():
    return dataloaders.CIFAR10DataLoaders.train_loader(batch_size=batch_size), dataloaders.CIFAR10DataLoaders.val_loader()


def setup_teacher_student(teacher_pth):
    teacher_model = resnet.resnet_models["cifar"][teacher_arch]()
    teacher_checkpoint = torch.load(teacher_pth)
    teacher_model.load_state_dict(teacher_checkpoint)

    student_model = resnet.resnet_models["cifar"][student_arch]()
    student_model.conv1.load_state_dict(teacher_model.conv1.state_dict())
    student_model.bn1.load_state_dict(teacher_model.bn1.state_dict())
    student_model.linear.load_state_dict(teacher_model.linear.state_dict())

    teacher_model = teacher_model.cuda()
    student_model = student_model.cuda()

    # freeze teacher model
    for p in teacher_model.parameters():
        p.requires_grad = False

    return teacher_model, student_model


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



if __name__ == '__main__':
    main()
