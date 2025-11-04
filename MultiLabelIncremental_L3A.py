import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.distributed as dist
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from src.helper_functions.helper_functions import mAP, add_weight_decay, reduce_tensor, \
    mean_average_precision, AverageMeter, normalize_sqrt
from src.models import my_create_model
from src.loss_functions.losses import AsymmetricLoss
from torch.cuda.amp import GradScaler, autocast
from src.helper_functions.IncrementalDataset import build_dataset, build_loader
from src.helper_functions.utils import build_logger, build_writer, print_to_excel, calculate_metrics, calculate_metrics_sk

import tqdm
import pandas as pd
import torch.nn.functional as F

class MultiLabelIncremental:
    def __init__(self, args):

        self.args = args
        # Distributed 
        self.world_size = args.world_size
        self.rank = args.local_rank

        # Output
        self.log_frequency = 100
        self.logger = build_logger(args.logger_dir, self.rank)
        self.logger.info('Running L3A')
        self.logger.info('Arguments:')
        for k, v in sorted(vars(args).items()):
            self.logger.info('{}={}'.format(k, v))

        self.save_model = args.save_model
        self.model_save_path = args.model_save_path
        if not os.path.exists(self.model_save_path) and self.rank == 0:
            os.makedirs(self.model_save_path)

        self.excel_path = args.excel_path

        # Train params
        self.epochs = args.epochs
        self.incr_lr = args.lr
        self.base_lr = args.base_lr
        self.weight_decay = args.weight_decay

        # Incremental params
        self.base_classes = args.base_classes
        self.task_size = args.task_size
        self.total_classes = args.total_classes

        self.num_classes = self.base_classes

        # model /load pretrained model info
        self.model_name = args.model_name
        self.pretrained_path = args.pretrained_path

        # resume /save old model and old dataset
        self.resume = args.resume
        self.resume_low_range = args.resume_low_range

        # datasets
        self.dataset_name = args.dataset_name
        self.root_dir = args.root_dir

        self.image_size = args.image_size
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.train_transforms = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            # normalize,
        ])
        self.val_transforms = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            # normalize, # no need, toTensor does normalization
        ])

        # model
        self.model = self.setup_model()
        self.model_without_ddp = self.model
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.rank],
                                                               broadcast_buffers=False,
                                                               find_unused_parameters=True)
        torch.backends.cudnn.benchmark = True

        # Analytic Learning params
        self.recurbase = args.recurbase
        self.hidden = args.Hidden
        self.repeat = args.repeat
        self.rg = args.rg
        self.rectifier_fac = args.rectifier_fac
        self.R = None
        self.A = None
        self.C = None
        self.current_cnt = None

        # Pseudo Label params
        self.pseudo_label = args.pseudo_label
        self.thre = args.thre
    
    def setup_model(self):
        """
        Create model
        Load Checkpoint from resume or pretrained weight
        """

        # Resume from checkpoint
        if self.resume:
            model = my_create_model(self.args, self.resume_low_range)
            model = model.cuda()
            
            self.lr = self.base_lr
            parameters = add_weight_decay(model, self.weight_decay)
            self.cls_criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
            self.optimizer = torch.optim.Adam(params=parameters, lr=self.lr, weight_decay=0)  # true wd, filter_bias_and_bn
    
            checkpoint = torch.load(self.resume, map_location='cpu')
            filtered_dict = {k: v for k, v in checkpoint['state_dict'].items() if (k in model.state_dict() and 'head.fc.weight' not in k and 'head.fc.bias' not in k)}
            model.load_state_dict(filtered_dict, strict=False)
            
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.logger.info('Create Model successfully, Loaded from resume path:{}, Loaded params:{}\n'
                             .format(self.resume, len(checkpoint['state_dict'])))

        # Load pretrained weights
        elif self.pretrained_path:  # make sure to load pretrained ImageNet model
            model = my_create_model(self.args, self.base_classes)
            model = model.cuda()
            state = torch.load(self.pretrained_path, map_location='cpu')

            # remove 'body' in params name
            if '21k' in self.pretrained_path:
                state = {(k if 'body.' not in k else k[5:]): v for k, v in state['model'].items()}
                filtered_dict = {k: v for k, v in state.items() if
                                (k in model.state_dict() and 'head.fc' not in k)}
            else:
                state = {(k if 'body.' not in k else k[5:]): v for k, v in state.items()}
                filtered_dict = {k: v for k, v in state.items() if
                                (k in model.state_dict() and 'head.fc' not in k)}
                
            model.load_state_dict(filtered_dict, strict=False)
            
            self.logger.info('Create Model successfully, Loaded from model_path:{}, Loaded params:{}\n'
                             .format(self.pretrained_path, len(filtered_dict)))

        return model

    def compute_loss(self, output, target):
        """
        Input: outputs of network, ground truth
        1. classification loss
        """
        loss = 0.
        logits = output['logits'].float()
        # Classification Loss
        loss = self.cls_criterion(logits, target)
        return loss

    # Weighted Analytic Classifier, Base Training
    def weight_cls_align(self, train_loader, wrapped_model):
        if hasattr(wrapped_model, 'module'):
            model = wrapped_model.module
        else:
            model = wrapped_model
        if hasattr(model, 'layer4'):
            pass
        else:
            model.layer4 = nn.Sequential()
        new_model = torch.nn.Sequential(model.space_to_depth, model.conv1, model.layer1, model.layer2,
                                    model.layer3, model.layer4, model.global_pool, model.head.fc[:2])
        model.eval()

        # calculate A and C matrices
        self.A = torch.zeros(model.head.fc[-1].weight.size(1), model.head.fc[-1].weight.size(1)).cuda()
        self.C = torch.zeros(model.head.fc[-1].weight.size(1), self.num_classes).cuda()
        
        self.current_cnt = [0] * self.num_classes
        for images, target in train_loader:
            target = target[:, :self.base_classes]
            label_multihot = target.float()
            self.current_cnt = [x + y for x, y in zip(self.current_cnt, label_multihot.sum(dim=0).tolist())]

        with torch.no_grad():
            for epoch in range(1):
                pbar = tqdm.tqdm(enumerate(train_loader), desc='Re-Alignment Base', total=len(train_loader), unit='batch')
                for i, (images, target) in pbar:
                    images = images.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)
                    target = target[:, :self.base_classes]

                    # feature extraction
                    new_activation = new_model(images)
                    new_activation = new_activation.double()
                    label_multihot = target.double()

                    Omega_list = []
                    for j in range(target.size(0)):
                        labels = label_multihot[j]
                        activation = new_activation[j]
                        label_sum = labels.sum().item()
                        ave_weight = 1.0 / label_sum

                        sample_labels = [k for k in range(self.num_classes) if labels[k]==1]
                        normalized_cnt = normalize_sqrt(self.current_cnt)
                        sample_normalized_cnt = [normalized_cnt[k] for k in sample_labels]

                        # sample specific weight
                        omega = torch.tensor(ave_weight * sum(sample_normalized_cnt)).double().cuda()
                        Omega_list.append(omega)
                    
                    Omega = torch.block_diag(*Omega_list)
                    self.A += torch.t(new_activation) @ Omega @ new_activation
                    self.C += torch.t(new_activation) @ Omega @ label_multihot

        A = self.A + self.rg * torch.eye(model.head.fc[-1].weight.size(1)).to(self.A)
        C = self.C

        R = torch.inverse(A)
        W = R @ C
        model.head.fc[-1].weight = torch.nn.parameter.Parameter(torch.t(W.float()))
        return R

    # Weighted Analytic Classifier, Class-Incremental Learning
    def weight_IL_align(self, train_loader, low_range, wrapped_model, R, repeat):
        if hasattr(wrapped_model, 'module'):
            model = wrapped_model.module
        else:
            model = wrapped_model
        if hasattr(model, 'layer4'):
            pass
        else:
            model.layer4 = nn.Sequential()
        new_model = torch.nn.Sequential(model.space_to_depth, model.conv1, model.layer1, model.layer2,
                                    model.layer3, model.layer4, model.global_pool, model.head.fc[:2])
        model.eval()
        
        W = (model.head.fc[-1].weight.t()).double()
        R = R.double()

        self.C = torch.cat((self.C, torch.zeros(model.head.fc[-1].weight.size(1), self.task_size).cuda()), dim=1)

        self.current_cnt = self.current_cnt + [0] * self.task_size

        for images, target in train_loader:
            target = target[:, :self.num_classes]
            label_multihot = target.float()
            self.current_cnt = [x + y for x, y in zip(self.current_cnt, label_multihot.sum(dim=0).tolist())]
        
        if self.pseudo_label:
            self.logger.info('Use default thre:{}'.format(self.thre))

        with torch.no_grad():
            for epoch in range(repeat):
                pbar = tqdm.tqdm(enumerate(train_loader), desc='Incremental Learning', total=len(train_loader), unit='batch')
                for i, (images, target) in pbar:
                    images = images.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)
                    
                    Omega_list = []
                    # Pseudo-Label Module
                    if self.pseudo_label:
                        old_output = model(images)
                        old_logits = old_output['logits']
                        old_logits = old_logits[:, :low_range].detach()
                        old_logits = torch.sigmoid(old_logits)
                        target[:, :low_range][old_logits > self.thre] = 1

                    target = target[:, :self.num_classes]  
                    new_activation = new_model(images)
                    new_activation = new_activation.double()
                    label_multihot = target.double()

                    for j in range(target.size(0)):
                        labels = label_multihot[j]
                        activation = new_activation[j]
                        label_sum = labels.sum().item()
                        ave_weight = 1.0 / label_sum

                        sample_labels = [k for k in range(self.num_classes) if labels[k]==1]
                        normalized_cnt = normalize_sqrt(self.current_cnt)
                        sample_normalized_cnt = [normalized_cnt[k] for k in sample_labels]
                        # sample specific weight
                        omega = torch.tensor(ave_weight * sum(sample_normalized_cnt)).double().cuda()
                        Omega_list.append(omega)

                    Omega = torch.block_diag(*Omega_list)
                    self.A += torch.t(new_activation) @ Omega @ new_activation
                    self.C += torch.t(new_activation) @ Omega @ label_multihot

        A = self.A + self.rg * torch.eye(model.head.fc[-1].weight.size(1)).to(self.A)
        C = self.C
        R = torch.inverse(A)
        W = R @ C
        model.head.fc[-1].weight = torch.nn.parameter.Parameter(torch.t(W.float()))
        return R
    
    # Base Training
    def cls_align(self, train_loader, wrapped_model):
        if hasattr(wrapped_model, 'module'):
            model = wrapped_model.module
        else:
            model = wrapped_model
        if hasattr(model, 'layer4'):
            pass
        else:
            model.layer4 = nn.Sequential()
        new_model = torch.nn.Sequential(model.space_to_depth, model.conv1, model.layer1, model.layer2,
                                    model.layer3, model.layer4, model.global_pool, model.head.fc[:2])
        model.eval()

        self.A = torch.zeros(model.head.fc[-1].weight.size(1), model.head.fc[-1].weight.size(1)).cuda()
        self.C = torch.zeros(model.head.fc[-1].weight.size(1), self.num_classes).cuda()

        with torch.no_grad():
            for epoch in range(1):
                pbar = tqdm.tqdm(enumerate(train_loader), desc='Re-Alignment Base', total=len(train_loader), unit='batch')
                for i, (images, target) in pbar:
                    images = images.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)
                    target = target[:, :self.base_classes]

                    new_activation = new_model(images)
                    new_activation = new_activation.float()

                    label_multihot = target.float()
                    
                    self.A += torch.t(new_activation) @ new_activation
                    self.C += torch.t(new_activation) @ (label_multihot)

        A = self.A + self.rg * torch.eye(model.head.fc[-1].weight.size(1)).to(self.A)
        C = self.C
        R = torch.inverse(A)
        W = R @ C
        model.head.fc[-1].weight = torch.nn.parameter.Parameter(torch.t(W.float()))
        return R


    # Class-Incremental Learning Agenda
    def IL_align(self, train_loader, low_range, wrapped_model, R, repeat):
        if hasattr(wrapped_model, 'module'):
            model = wrapped_model.module
        else:
            model = wrapped_model
        if hasattr(model, 'layer4'):
            pass
        else:
            model.layer4 = nn.Sequential()
        new_model = torch.nn.Sequential(model.space_to_depth, model.conv1, model.layer1, model.layer2,
                                    model.layer3, model.layer4, model.global_pool, model.head.fc[:2])  
        model.eval()
        
        W = (model.head.fc[-1].weight.t()).double()
        R = R.double()
        self.C = torch.cat((self.C, torch.zeros(model.head.fc[-1].weight.size(1), self.task_size).cuda()), dim=1)

        if self.pseudo_label:
            self.logger.info('Use default thre:{}'.format(self.thre))

        with torch.no_grad():
            for epoch in range(repeat):
                pbar = tqdm.tqdm(enumerate(train_loader), desc='Incremental Learning', total=len(train_loader), unit='batch')
                for i, (images, target) in pbar:
                    images = images.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)
                    
                    if self.pseudo_label:
                        old_output = model(images)
                        old_logits = old_output['logits']
                        old_logits = old_logits[:, :low_range].detach()
                        old_logits = torch.sigmoid(old_logits)
                        target[:, :low_range][old_logits > self.thre] = 1

                    target = target[:, :self.num_classes]
                    new_activation = new_model(images)
                    new_activation = new_activation.double()
                    label_multihot = target.double()

                    self.A += torch.t(new_activation) @ new_activation
                    self.C += torch.t(new_activation) @ (label_multihot)

        A = self.A + self.rg * torch.eye(model.head.fc[-1].weight.size(1)).to(self.A)
        C = self.C
        R = torch.inverse(A)
        W = R @ C
        model.head.fc[-1].weight = torch.nn.parameter.Parameter(torch.t(W.float()))
        return R

    def _before_task(self, low_range, high_range):
        self.model.eval()
        self.num_classes = high_range

    def _train_one_epoch(self, train_loader, scaler, low_range, high_range, epoch):
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        for i, (image, target) in enumerate(train_loader):
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)  # (batch,3,num_classes)
            target = target[:, :high_range]

            with autocast():
                output = self.model(image)  # sigmoid will be done in loss !
            loss = self.compute_loss(output, target)

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            self.scheduler.step()
            self.model.zero_grad(set_to_none=True)

            # reduce loss for distributed
            if self.world_size > 1:
                loss = reduce_tensor(loss.data, self.world_size)

            # Log trainning information
            if i % self.log_frequency == 0:
                self.logger.info(
                    'Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                    .format(epoch + 1, self.epochs, str(i).zfill(3), str(len(train_loader)).zfill(3),
                            self.scheduler.get_last_lr()[0], loss.item()))


    def _train_task(self, epochs, low_range, high_range, train_loader,
                    val_loader_base, val_loader_seen, val_loader_new):
        # Base training
        if low_range == 0:
            if not self.resume and epochs != 0:
                self.model.train()

                self.lr = self.base_lr
                parameters = add_weight_decay(self.model_without_ddp, self.weight_decay)
                self.cls_criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
                self.optimizer = torch.optim.Adam(params=parameters, lr=self.lr, weight_decay=0)  # true wd, filter_bias_and_bn
                self.scheduler = lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.lr, steps_per_epoch=len(train_loader),
                                                        epochs=self.epochs,
                                                        pct_start=0.2)
                scaler = GradScaler()
                for epoch in range(epochs):
                    # train_loader.sampler.set_epoch(epoch)
                    self._train_one_epoch(train_loader, scaler, low_range, high_range, epoch)
                    if self.save_model:
                        self._after_task(low_range, high_range, epoch)

            bias_fe = False
            if hasattr(self.model, 'module'):
                self.model.module.head.fc = nn.Sequential(nn.Linear(self.model.module.head.fc.weight.size(1), self.hidden, bias=bias_fe),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden, self.base_classes, bias=False)).cuda()
            else:
                self.model.head.fc = nn.Sequential(nn.Linear(self.model.head.fc.weight.size(1), self.hidden, bias=bias_fe),
                                        nn.ReLU(),
                                        nn.Linear(self.hidden, self.base_classes, bias=False)).cuda()

            if self.rectifier_fac:
                self.R = self.weight_cls_align(train_loader, self.model)
            else:
                self.R = self.cls_align(train_loader, self.model)

        else:
            if hasattr(self.model, 'module'):
                W = self.model.module.head.fc[-1].weight
                W = torch.cat([W, torch.zeros(self.task_size, self.hidden).cuda()], dim=0)
                self.model.module.head.fc[-1] = nn.Linear(self.hidden, self.num_classes, bias=False)
                self.model.module.head.fc[-1].weight = torch.nn.parameter.Parameter(W.float())
            else:
                W = self.model.head.fc[-1].weight
                W = torch.cat([W, torch.zeros(self.task_size, self.hidden).cuda()], dim=0)
                self.model.head.fc[-1] = nn.Linear(self.hidden, self.num_classes, bias=False)
                self.model.head.fc[-1].weight = torch.nn.parameter.Parameter(W.float())
            
            if self.rectifier_fac:
                self.R = self.weight_IL_align(train_loader, low_range, self.model, self.R, self.repeat)
            else:
                self.R = self.IL_align(train_loader, low_range, self.model, self.R, self.repeat)

        self.model.eval()
        val_result, val_result2 = self.validate(low_range, high_range, val_loader_base, val_loader_seen, val_loader_new,
                                    only_seen=(low_range == 0))

        # Base session
        if low_range == 0:
            val_result['base'] = val_result['seen']
            val_result['new'] = val_result['seen']
        self.logger.info('current_mAP_base = {:.2f}'.format(val_result['base'][0]))
        self.logger.info('current_mAP_seen = {:.2f}'.format(val_result['seen'][0]))
        self.logger.info('current_mAP_new = {:.2f}'.format(val_result['new'][0]))
        if self.rank == 0:
            self.logger.info('current other metrics:, mean_p_c:{}, mean_r_c:{}, mean_f_c:{}, precision_o:{}, recall_o:{}, f1_o:{}'
            .format(*[i for i in val_result2['seen']]))

        return val_result['seen'][0], val_result2['seen']


    # save model after each task
    def _after_task(self, low_range, high_range, epoch):
        save_data = {
                'state_dict': self.model_without_ddp.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
        torch.save(save_data, '{}/{}_{}_epoch{}_{}to{}.pth.tar'.format(
            self.model_save_path, self.dataset_name, self.model_name, epoch,
            low_range, high_range))
        

    def validate(self, low_range, high_range, val_loader_base, val_loader_seen, val_loader_new, only_seen=False):
        """
        Validate model on base / seen / new classes correspondingly
        """
        self.model.eval()
        self.logger.info("Starting validation")
        Sig = torch.nn.Sigmoid()

        val_result = {}
        val_result2 = {}
        val_result2['seen'] = [0]

        # validate on 3 datasets
        val_stage = ['base', 'seen', 'new']
        val_classes = [(0, self.base_classes), (0, high_range), (low_range, high_range)]
        for i, val_loader in enumerate([val_loader_base, val_loader_seen, val_loader_new]):
            if only_seen and val_stage[i] != 'seen':
                continue

            preds_regular = []
            targets = []

            for image, target in val_loader:
                image = image.cuda()
                target = target[:, val_classes[i][0]:val_classes[i][1]].cuda()

                with torch.no_grad():
                    with autocast():
                        output_regular = Sig(self.model(image)['logits'])
                        output_regular = output_regular[:, val_classes[i][0]:val_classes[i][1]].contiguous()

                if self.world_size > 1:  # This is for DDP
                    output_gather_list = [torch.zeros_like(output_regular) for _ in range(self.world_size)]
                    target_gather_list = [torch.zeros_like(target) for _ in range(self.world_size)]

                    dist.all_gather(output_gather_list, output_regular)
                    dist.all_gather(target_gather_list, target)

                    output_regular = torch.cat(output_gather_list, dim=0)
                    target = torch.cat(target_gather_list, dim=0)

                # for mAP calculation
                preds_regular.append(output_regular.detach())
                targets.append(target.detach())

            mAP_score_regular = 0
            score_regular = 0
            mean_p_c = 0
            mean_r_c = 0
            mean_f_c = 0
            precision_o = 0
            recall_o = 0
            f1_o = 0

            if self.rank == 0:
                mAP_score_regular, score_regular = mAP(torch.cat(targets).cpu().numpy(),
                                                       torch.cat(preds_regular).cpu().numpy())
                
                if val_stage[i] == 'seen':  # only calculate metrics for the seen classes
                    print('calculate metrics')
                    # mean_p_c, mean_r_c, mean_f_c, precision_o, recall_o, f1_o = calculate_metrics(
                    #     torch.cat(preds_regular).cpu(), torch.cat(targets).cpu(), thre = 0.525)
                    mean_p_c, mean_r_c, mean_f_c, precision_o, recall_o, f1_o = calculate_metrics_sk(
                        torch.cat(preds_regular).cpu(), torch.cat(targets).cpu(), thre = 0.525)
                    val_result2['seen'] = [mean_p_c, mean_r_c, mean_f_c, precision_o, recall_o, f1_o]

            
            val_result[val_stage[i]] = (mAP_score_regular, score_regular)

        return val_result, val_result2


    def train(self):
        mAP_meter = AverageMeter()
        mAP_list = np.zeros((self.total_classes-self.base_classes) // self.task_size + 1)
        
        if self.resume:
            base_stage = [(0, self.resume_low_range)]
            # Load the last training info
            incremental_stages = base_stage + [(low, low + self.task_size) for low in
                                  range(self.resume_low_range, self.total_classes, self.task_size)]
        else:
            base_stage = [(0, self.base_classes)]
            incremental_stages = base_stage + [
                (low, low + self.task_size) for low in range(self.base_classes, self.total_classes, self.task_size)]

        # Incremental learning stages
        for low_range, high_range in incremental_stages:
            train_dataset_without_old = build_dataset(self.dataset_name, self.root_dir, low_range, high_range,
                                          phase='train', transform=self.train_transforms)
            self.logger.info('Current incremental stage:({},{}), dataset length:{}'
                             .format(low_range, high_range, len(train_dataset_without_old)))
            train_dataset = train_dataset_without_old
            val_dataset_base = build_dataset(self.dataset_name, self.root_dir, 0, self.base_classes, phase='val',
                                             transform=self.val_transforms)
            val_dataset_seen = build_dataset(self.dataset_name, self.root_dir, 0, high_range, phase='val',
                                             transform=self.val_transforms)
            val_dataset_new = build_dataset(self.dataset_name, self.root_dir, low_range, high_range, phase='val',
                                            transform=self.val_transforms)

            # Build loaders
            train_loader = build_loader(train_dataset, self.batch_size, self.num_workers, phase='train')
            val_loader_base = build_loader(val_dataset_base, self.batch_size, self.num_workers, phase='val')
            val_loader_seen = build_loader(val_dataset_seen, self.batch_size, self.num_workers, phase='val')
            val_loader_new = build_loader(val_dataset_new, self.batch_size, self.num_workers, phase='val')

            # Training process
            self._before_task(low_range, high_range)
            mAP, metrics = self._train_task(self.epochs, low_range, high_range, train_loader, val_loader_base,
                                   val_loader_seen, val_loader_new)  # Calculate mAP for a phase
            mAP_meter.update(mAP)
            mAP_list[(high_range-self.base_classes)//self.task_size] = mAP
        
        # Print result to excel
        if self.rank == 0:
            if 'coco' in self.dataset_name:
                ds_name = 'COCO' 
            if 'voc' in self.dataset_name:
                ds_name = 'VOC'
            expe_name = self.args.output_name
            if self.resume:
                params = f"Model:{self.resume}, BS:{self.batch_size}, RG:{self.rg}, Hidden:{self.hidden}, Fac:{self.rectifier_fac}, Resume"
            else:
                params = f"LR:{self.lr}, epoch:{self.epochs}, BS:{self.batch_size}, RG:{self.rg}, Hidden:{self.hidden}, Fac:{self.rectifier_fac}, Pseudo:{self.pseudo_label}, Thre:{self.thre}"
            print_to_excel(self.excel_path, expe_name, ds_name, self.base_classes, 
                self.task_size, self.total_classes, params, mAP_list, metrics, git_hash=None)