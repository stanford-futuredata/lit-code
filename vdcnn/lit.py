"""
Author: Animesh Koratana <koratana@stanford.edu>
LIT: Lightweight Iterative Trainer
"""
import os
import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.cuda
import torch.optim
import torch.utils.data
from tqdm import tqdm
from model_parallel import *

import threading
import fp16

from tensorboardX import SummaryWriter

beta = 0.5

class Section(object):
    def __init__(self, section):
        self.network = section
        self.optimizer = None
        self.lr_scheduler = None
        self.params = None
        self.init = False
        self.set_initial_lr = False
        self.device = torch.device("cpu")

        #Half vars
        self.half = None
        self.loss_scaling = 1

    def set_optimizer(self, optimizer_fn, train_params):
        self.optimizer = optimizer_fn
        self.params = train_params
        return self
    def set_lr_scheduler(self, scheduler):
        self.lr_scheduler = scheduler
        return self

    def build(self, half = False):
        assert self.network and self.optimizer and self.lr_scheduler and self.params
        self.half = half
        if self.half:
            # Cast network to half
            self.network = fp16.FP16(self.network)
            # Manage a fp32 version of the weights
            self.params = [param.clone().type(torch.cuda.FloatTensor).detach() for param in self.params]
            for p in self.params:
                p.requires_grad = True
        self.optimizer = self.optimizer(self.params)
        if self.set_initial_lr:
            for group in self.optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        self.lr_scheduler = self.lr_scheduler(self.optimizer)
        self.init = True
        return self

    def __call__(self, inp):
        return self.network(inp)

    def step(self, out, target, losses_log, criterion, lock):
        global beta
        assert(self.init)
        loss = criterion(out, target) * self.loss_scaling * (1 - beta)
        if self.half:
            loss.backward()
            fp16.set_grad(self.params, list(self.network.parameters()))
            if self.loss_scaling != 1:
                for param in self.params:
                    param.grad.data = param.grad.data/args.loss_scale
            self.optimizer.step()
            fp16.copy_in_params(self.network, self.params)
            self.network.zero_grad()
        else:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        with lock:
            losses_log.update(loss.item(), out.size(0))

    def eval(self, out, target, losses_log, criterion, lock):
        loss = criterion(out, target)
        with lock:
            losses_log.update(loss.item(), out.size(0))



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

def correct(outputs, targets, top=(1, )):
    with torch.no_grad():
        _, predictions = outputs.topk(max(top), dim=1, largest=True, sorted=True)
        targets = targets.view(-1, 1).expand_as(predictions)

        corrects = predictions.eq(targets).cpu().int().cumsum(1).sum(0)
        tops = list(map(lambda k: corrects.data[k - 1], top))
        return tops

class LitTrainer():
    def __init__(self, f):
        cudnn.benchmark = True
        global beta
        beta = f.beta

        self.distributed = torch.cuda.device_count() > 1

        # Check the save_dir exists or not
        if not os.path.exists(f.save_dir):
            os.makedirs(f.save_dir)

        if not os.path.exists(f.log_dir):
            os.makedirs(f.log_dir)
        self.writer = SummaryWriter(f.log_dir)
        self.save_dir = f.save_dir
        self.model_name = f.model_name
        self.save_every = f.save_every

        self.start_epoch = f.start_epoch
        self.start_segment = f.start_segment
        self.half = f.half
        self.lit_sections = f.lit_sections
        self.sequence = f.sequence
        self.momentum = f.momentum
        self.weight_decay = f.weight_decay
        self.loss_scaling = f.loss_scaling
        self._make_optimizers(self.lit_sections)
        self.trainable_model = LearnerModelParallel(f.trainable_model, self.lit_sections)
        for section in self.lit_sections.values():
            section.build(half = self.half)
        self.base_model = f.base_model
        if self.half:
            self.base_model = fp16.FP16(self.base_model)
        self.base_model = nn.DataParallel(self.base_model).cuda()
        for param in self.base_model.parameters(): param.requires_grad = False

        self.lit_train_loader = f.lit_training_data_loader
        self.fine_tuning_loader = f.fine_tuning_data_loader
        self.val_loader = f.val_data_loader

        self.lit_criterion = f.lit_criterion()
        self.fine_tuning_criterion = f.fine_tuning_criterion()

        self.best_accuracy1 = 0

    def _make_optimizers(self, lit_sections):
        lit_start_epoch = self.start_epoch if self.start_segment == 0 else 0
        for k, v in lit_sections.items():
            v.loss_scaling = self.loss_scaling
            v.set_optimizer(lambda params :torch.optim.SGD(params,
                                    lr = self.sequence["lit"]["lr"],
                                    momentum=self.momentum,
                                    weight_decay=self.weight_decay),
                                    train_params = v.network.parameters())
            if lit_start_epoch > 0:
                v.set_initial_lr = True
            v.set_lr_scheduler(lambda optimizer : torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                    milestones=self.sequence["lit"]["milestones"],
                                    last_epoch=lit_start_epoch-1, gamma=0.5))

    def train(self):
        self._train_lit()
        torch.cuda.empty_cache()
        self.trainable_model.cpu()
        self.trainable_model = self.trainable_model.module
        self._fine_tune()

    def _train_lit(self):
        if self.start_segment > 0: return
        # Freeze the whole model and then unfreeze only the sections
        for param in self.trainable_model.parameters():
            param.requires_grad = False
        self._unfreeze_training_model_sections(*list(self.lit_sections.keys()))
        for epoch in tqdm(range(self.start_epoch, self.sequence["lit"]["epochs"]), desc="LIT Training", dynamic_ncols=True):
            t_losses, t_data_time, t_batch_time = self._lit_train_one_epoch(epoch)
            v_losses, v_data_time, v_batch_time = self._lit_eval_one_epoch(epoch)
            for section in self.lit_sections.values(): section.lr_scheduler.step()
            for i, (t_loss, v_loss) in enumerate(zip(t_losses, v_losses)):
                # Add logging data
                self.writer.add_scalar('loss/section{t}/train'.format(t=i+1), t_loss, epoch)
                self.writer.add_scalar('loss/section{t}/validation'.format(t=i+1), v_loss, epoch)
            if epoch > 0 and epoch % self.save_every:
                torch.save(self.trainable_model.module.state_dict(),
                           os.path.join(self.save_dir, str("checkpoint_" + self.model_name)))

    def _fine_tune(self):
        torch.cuda.empty_cache()
        if self.start_segment<=2:
            # Unfreeze everything and then train everything together
            for param in self.trainable_model.parameters():
                param.requires_grad = True
            if self.half:
                self.trainable_model = nn.DataParallel(fp16.FP16(self.trainable_model)).cuda()
                param_copy = [param.clone().type(torch.cuda.FloatTensor).detach() for param in self.trainable_model.parameters()]
                for param in param_copy:
                    param.requires_grad = True
                optimizer = torch.optim.SGD(param_copy, self.sequence["full_model"]["lr"],
                                                                momentum=self.momentum,
                                                                weight_decay=self.weight_decay)
            else:
                param_copy = None
                self.trainable_model = nn.DataParallel(self.trainable_model).cuda()
                optimizer = torch.optim.SGD(self.trainable_model.parameters(), self.sequence["full_model"]["lr"],
                                                                momentum=self.momentum,
                                                                weight_decay=self.weight_decay)

            start_epoch = self.start_epoch if self.start_segment == 2 else 0
            if start_epoch > 0:
                for group in optimizer.param_groups:
                    group.setdefault('initial_lr', group['lr'])
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.sequence["full_model"]["milestones"], last_epoch=start_epoch-1)
            self._run_fine_tuning_epochs(optimizer, lr_scheduler, "full model", self.sequence["full_model"]["epochs"], start_epoch=start_epoch, params= param_copy)


    def _run_fine_tuning_epochs(self, optimizer, lr_scheduler, tag, num_epochs, start_epoch = 0, params = None):
        for epoch in tqdm(range(start_epoch, num_epochs), desc="Fine Tuning {}".format(tag), dynamic_ncols=True):
            t_loss, t_data_time, t_batch_time, t_accuracy1 = self._fine_tune_train_one_epoch(
                tag=tag, optimizer=optimizer, current_epoch=epoch)
            v_loss, v_data_time, v_batch_time, v_accuracy1 = self._fine_tune_evaluate_one_epoch(
                tag=tag, current_epoch=epoch)
            self.writer.add_scalar('loss/{t}/train'.format(t=tag), t_loss, epoch)
            self.writer.add_scalar('loss/{t}/validation'.format(t=tag), v_loss, epoch)
            self.writer.add_scalar('prec1/{t}/train'.format(t=tag), t_accuracy1, epoch)
            self.writer.add_scalar('prec1/{t}/validation'.format(t=tag), v_accuracy1, epoch)
            best = v_accuracy1 > self.best_accuracy1
            self.best_accuracy1 = max(v_accuracy1, self.best_accuracy1)
            if best:
                tqdm.write("Saving the best model with prec1@ {a}".format(a=v_accuracy1))
                torch.save(self.trainable_model.module.state_dict(), os.path.join(self.save_dir, self.model_name))
            if epoch > 0 and epoch % self.save_every:
                torch.save(self.trainable_model.module.state_dict(),
                           os.path.join(self.save_dir, str("checkpoint_" + self.model_name)))
            lr_scheduler.step()

    def _lit_train_one_epoch(self, current_epoch):
        global beta
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = [AverageMeter() for i in range(len(self.lit_sections.keys()))]

        # switch to appropriate mode
        self.trainable_model.train()
        self.base_model.eval()
        criterion = self.lit_criterion

        end = time.time()
        data_loader = self.lit_train_loader
        lock = threading.Lock()
        for i, (inp, target) in tqdm(enumerate(data_loader),
                                     desc='LIT Training: Epoch {epoch}'.format(epoch = current_epoch),
                                     dynamic_ncols=True, total= len(data_loader), leave=True):
            batch_size = target.size(0)
            assert batch_size < 2**32, 'Size is too large! correct will overflow'
            # measure data loading time
            data_time.update(time.time() - end)
            input_var = inp.cuda().detach()

            #Get teacher model's intermediate
            with torch.no_grad():
                teacher_features, soft_targets = self.base_model(input_var, get_features = True)
            learner_features = self.trainable_model(teacher_features)
            learner_out = self.trainable_model.module(input_var)
            full_loss = self.fine_tuning_criterion(learner_out, soft_targets, target) * beta
            full_loss.backward(retain_graph=True)

            jobs = []
            for id in self.lit_sections.keys():
                learner_output = learner_features[id]
                target_feature = teacher_features[id].cuda(self.lit_sections[id].device)
                losses_log = losses[id-1]
                p = threading.Thread(target=self.lit_sections[id].step, args=(learner_output, target_feature, losses_log, criterion, lock))
                jobs.append(p)
            for job in jobs:
                job.start()
            for job in jobs:
                job.join()
            batch_time.update(time.time() - end)
            end = time.time()
        # Clean up
        del inp, teacher_features, input_var, learner_output
        return [loss.avg for loss in losses], data_time.avg, batch_time.avg

    def _lit_eval_one_epoch(self, current_epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = [AverageMeter() for i in range(len(self.lit_sections.keys()))]

        # switch to appropriate mode
        self.trainable_model.eval()
        self.base_model.eval()
        criterion = self.lit_criterion

        end = time.time()

        lock = threading.Lock()
        with torch.no_grad():
            for i, (inp, target) in tqdm(enumerate(self.val_loader),
                                    desc='LIT Validation: Epoch {epoch}'.format(epoch=current_epoch),
                                    dynamic_ncols=True, total=len(self.val_loader), leave=True):
                batch_size = target.size(0)
                assert batch_size < 2**32, 'Size is too large! correct will overflow'
                # measure data loading time
                data_time.update(time.time() - end)
                input_var = inp.cuda().detach()

                # Get teacher model's intermediate
                teacher_features, _ = self.base_model(input_var, get_features=True)
                learner_features = self.trainable_model(teacher_features)

                jobs = []
                for id in self.lit_sections.keys():
                    learner_output = learner_features[id]
                    target_feature = teacher_features[id].cuda(self.lit_sections[id].device)
                    losses_log = losses[id-1]
                    p = threading.Thread(target=self.lit_sections[id].eval, args=(learner_output, target_feature, losses_log, criterion, lock))
                    jobs.append(p)
                for job in jobs:
                    job.start()
                for job in jobs:
                    job.join()

                batch_time.update(time.time() - end)
                end = time.time()

            # Clean up
            del inp, target, teacher_features, input_var, learner_output
        return [loss.avg for loss in losses], data_time.avg, batch_time.avg

    def _fine_tune_train_one_epoch(self, tag, optimizer, current_epoch, params = None):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()

        # switch to appropriate mode
        self.trainable_model.train()
        self.base_model.eval()

        criterion = self.fine_tuning_criterion
        end = time.time()
        data_loader = self.fine_tuning_loader

        for i, (inp, target) in tqdm(enumerate(data_loader),
                                     desc="Fine Tuning {tag}: Epoch {epoch}".format(tag = tag, epoch=current_epoch),
                                     dynamic_ncols=True, total=len(data_loader), leave=True):
            batch_size = target.size(0)
            assert batch_size < 2**32, 'Size is too large! correct will overflow'
            # measure data loading time
            data_time.update(time.time() - end)
            target_var = target.cuda(non_blocking = True)

            input_var = inp.cuda() if not self.distributed else inp

            # compute outputs and loss
            with torch.no_grad():
                teacher_outputs = self.base_model(input_var).detach().cuda(non_blocking = True)
            learner_output = self.trainable_model(input_var).cuda()
            loss = criterion(learner_output, teacher_outputs, target_var) * self.loss_scaling

            if self.half:
                self.trainable_model.zero_grad()
                loss.backward()
                fp16.set_grad(params, list(self.trainable_model.parameters()))
                if self.loss_scaling != 1:
                    for param in params:
                        param.grad.data = param.grad.data/self.loss_scaling
                optimizer.step()
                fp16.copy_in_params(self.trainable_model, params)
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Calculate vals for logging
            losses.update(loss.item(), batch_size)
            top_correct = correct(learner_output, target, top=(1, ))[0]
            accuracies.update(top_correct.item() * (100. / batch_size), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        del inp, target, loss, input_var, target_var, learner_output
        return losses.avg, data_time.avg, batch_time.avg, accuracies.avg

    def _fine_tune_evaluate_one_epoch(self, tag, current_epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()

        # switch to appropriate mode
        self.trainable_model.eval()
        self.base_model.eval()

        criterion = self.fine_tuning_criterion
        end = time.time()
        data_loader = self.val_loader

        with torch.no_grad():
            for i, (inp, target) in tqdm(enumerate(data_loader),
                                         desc="Evaluating {tag}: Epoch {epoch}".format(tag = tag, epoch=current_epoch),
                                         dynamic_ncols=True, total=len(data_loader), leave=True):
                batch_size = target.size(0)
                assert batch_size < 2**32, 'Size is too large! correct will overflow'
                # measure data loading time
                data_time.update(time.time() - end)
                target_var = target.cuda(non_blocking = True)
                input_var = inp.cuda() if not self.distributed else inp
                # compute outputs and loss
                teacher_outputs = self.base_model(input_var).detach().cuda(non_blocking = True)
                learner_output = self.trainable_model(input_var).cuda()
                loss = criterion(learner_output, teacher_outputs, target_var) * self.loss_scaling

                # Calculate vals for logging
                losses.update(loss.item(), batch_size)
                top_correct = correct(learner_output, target, top=(1, ))[0]
                accuracies.update(top_correct.item() * (100. / batch_size), batch_size)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
            del input_var, target_var, inp, target, learner_output
            return losses.avg, data_time.avg, batch_time.avg, accuracies.avg

    def close_log_writer(self):
        self.writer.close()
    def validate_model(self):
        # Load best trained model for validation
        best_model = torch.load(os.path.join(self.save_dir, self.model_name))
        try:
            self.trainable_model.module.load_state_dict(best_model)
        except:
            self.trainable_model.load_state_dict(best_model)
        self.trainable_model.cuda()

        if isinstance(self.trainable_model, LearnerModelParallel):
            self.trainable_model = self.trainable_model.module

        t_losses = AverageMeter()
        t_accuracy = AverageMeter()
        criterion = nn.CrossEntropyLoss()

        # switch to appropriate mode
        self.trainable_model.eval()
        # self.base_model.eval()
        with torch.no_grad():
            for i, (input, target) in tqdm(enumerate(self.val_loader), desc='Validating Model for Benchmark',
                                           dynamic_ncols=True, total= len(self.val_loader)):
                batch_size = target.size(0)
                assert batch_size < 2**32, 'Size is too large! correct will overflow'
                input = input.cuda()
                target_var = target.cuda(non_blocking=True)
                # compute outputs
                t_output = self.trainable_model(input)
                t_loss = criterion(t_output.cuda() + 1e-16, target_var)
                # Calculate vals for logging
                t_losses.update(t_loss.item(), input.size(0))
                top_correct = correct(t_output, target, top=(1, ))[0]
                t_accuracy.update(top_correct.item() * (100. / batch_size), batch_size)
        return t_losses.avg, t_accuracy.avg

    def _unfreeze_training_model_sections(self, *sections):
        for i in sections:
            for param in self.lit_sections[i].network.parameters():
                param.requires_grad = True
    def _freeze_training_model_sections(self, *sections):
        for i in sections:
            for param in self.lit_sections[i].network.parameters():
                param.requires_grad = False
