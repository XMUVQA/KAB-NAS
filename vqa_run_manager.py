# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import time
import json
from datetime import timedelta
import numpy as np
import copy

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as Data
from openvqa.datasets.dataset_loader import EvalLoader
from utils import *
from models.normal_nets.proxyless_nets import ProxylessNASNets
from modules.mix_op import MixedEdge
from data_providers.vqa import VQADataProvider
#from data_providers.gqa import GQADataProvider

class VRunConfig:

    def __init__(self, __C,  lr_schedule_type, lr_schedule_param,
                 dataset='vqa', validation_frequency=1, print_frequency=20):
        self.n_epochs = __C.MAX_EPOCH
        self.init_lr = __C.LR_BASE
        self.lr_schedule_type = lr_schedule_type
        self.lr_schedule_param = lr_schedule_param
        self.__C = __C

        if dataset=='vqa':
            self.dataset = VQADataProvider(__C)
        #elif dataset=='gqa':
        #    self.dataset = GQADataProvider(__C)

        self.dataset_name=dataset

        self.train_loader = Data.DataLoader(
	        self.dataset.train_set,
	        batch_size=__C.BATCH_SIZE,
	        shuffle=True,
	        num_workers=__C.NUM_WORKERS,
	        pin_memory=__C.PIN_MEM,
	        drop_last=True
	    ) 
        self.val_loader = Data.DataLoader(
	        self.dataset.val_set,
	        batch_size=self.__C.EVAL_BATCH_SIZE,
	        shuffle=False,
	        num_workers=self.__C.NUM_WORKERS,
	        pin_memory=self.__C.PIN_MEM
	    ) 
        self.test_set = self.val_loader
        self.step_lr=__C.LR_DECAY_LIST
        self.lr_decay_rate=__C.LR_DECAY_R

       
        self.validation_frequency = validation_frequency
        self.print_frequency = print_frequency
        self._train_iter, self._valid_iter, self._test_iter = None, None, None

        

    @property
    def config(self):
        config = {}
        for key in self.__dict__:
            if not key.startswith('_'):
                config[key] = self.__dict__[key]
        return config

    def copy(self):
        return RunConfig(**self.config)

    """ learning rate """

    def _calc_learning_rate(self, epoch, batch=0, nBatch=None):
        #if self.lr_schedule_type == 'cosine':
        #    T_total = self.n_epochs * nBatch
        #    T_cur = epoch * nBatch + batch
        #    lr = 0.5 * self.init_lr * (1 + math.cos(math.pi * T_cur / T_total))
        #else:
        epoch+=1
        if epoch<self.step_lr[0]:
            lr = self.init_lr
        elif epoch>=self.step_lr[0] and epoch<self.step_lr[1]:
            lr = self.init_lr*self.lr_decay_rate
        else:
            lr = self.init_lr*self.lr_decay_rate*self.lr_decay_rate
           
        return lr

    def adjust_learning_rate(self, optimizer, epoch, batch=0, nBatch=None):
        """ adjust learning of a given optimizer and return the new learning rate """
        new_lr = self._calc_learning_rate(epoch, batch, nBatch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    """ data provider """

    @property
    def data_config(self):
        raise NotImplementedError

    """ optimizer """

    @property
    def train_next_batch(self):
        if self._train_iter is None:
            self._train_iter = iter(self.train_loader)
        try:
            data = next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_loader)
            data = next(self._train_iter)
        return data

    @property
    def valid_next_batch(self):
        if self._valid_iter is None:
            self._valid_iter = iter(self.val_loader)
        try:
            data = next(self._valid_iter)
        except StopIteration:
            self._valid_iter = iter(self.val_loader)
            data = next(self._valid_iter)
        return data

    @property
    def test_next_batch(self):
        if self._test_iter is None:
            self._test_iter = iter(self.test_loader)
        try:
            data = next(self._test_iter)
        except StopIteration:
            self._test_iter = iter(self.test_loader)
            data = next(self._test_iter)
        return data

    def build_optimizer(self, net_params):
        
        return torch.optim.Adam(net_params,self.init_lr)


class VQARunManager:

    def __init__(self,__C, path, net, run_config: VRunConfig, out_log=True):
        self.path = path
        self.net = net
        self.run_config = run_config
        self.out_log = out_log
        self.__C = __C

        self._logs_path, self._save_path = None, None
        self.best_acc = 0
        self.start_epoch = 0

        # initialize model (default)
        #self.net.init_model(run_config.model_init, run_config.init_div_groups)

        # a copy of net on cpu for latency estimation & mobile latency model
        #self.net_on_cpu_for_latency = copy.deepcopy(self.net).cpu()
        #self.latency_estimator = LatencyEstimator()

        # move network to GPU if available
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            self.net = torch.nn.DataParallel(self.net)
            self.net.to(self.device)
            cudnn.benchmark = True
        else:
            raise ValueError
            # self.device = torch.device('cpu')

        # net info
        #self.print_net_info(measure_latency)
        if self.__C.LOSS_FUNC=='bce':
            self.criterion = nn.BCEWithLogitsLoss(reduction=__C.LOSS_REDUCTION)
        else:
            self.criterion = nn.CrossEntropyLoss(reduction=__C.LOSS_REDUCTION)
        self.optimizer = self.run_config.build_optimizer(self.net.module.weight_parameters())
        

    """ save path and log path """

    @property
    def save_path(self):
        if self._save_path is None:
            save_path = os.path.join(self.path, 'checkpoint')
            os.makedirs(save_path, exist_ok=True)
            self._save_path = save_path
        return self._save_path

    @property
    def logs_path(self):
        if self._logs_path is None:
            logs_path = os.path.join(self.path, 'logs')
            os.makedirs(logs_path, exist_ok=True)
            self._logs_path = logs_path
        return self._logs_path

    """ net info """

   

   

    def print_net_info(self, measure_latency=None):
        # network architecture
        if self.out_log:
            print(self.net)

        # parameters
        if isinstance(self.net, nn.DataParallel):
            total_params = count_parameters(self.net.module)
        else:
            total_params = count_parameters(self.net)
        if self.out_log:
            print('Total training params: %.2fM' % (total_params / 1e6))
        net_info = {
            'param': '%.2fM' % (total_params / 1e6),
        }

        with open('%s/net_info.txt' % self.logs_path, 'w') as fout:
            fout.write(json.dumps(net_info, indent=4) + '\n')

    """ save and load models """

    def save_model(self, checkpoint=None, is_best=False, model_name=None):
        if checkpoint is None:
            checkpoint = {'state_dict': self.net.module.state_dict()}

        if model_name is None:
            model_name = 'checkpoint.pth.tar'

        checkpoint['dataset'] = self.run_config.dataset  # add `dataset` info to the checkpoint
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        model_path = os.path.join(self.save_path, model_name)
        with open(latest_fname, 'w') as fout:
            fout.write(model_path + '\n')
        torch.save(checkpoint, model_path)

        if is_best:
            best_path = os.path.join(self.save_path, 'model_best.pth.tar')
            torch.save({'state_dict': checkpoint['state_dict']}, best_path)

    def load_model(self, model_fname=None):
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        if model_fname is None and os.path.exists(latest_fname):
            with open(latest_fname, 'r') as fin:
                model_fname = fin.readline()
                if model_fname[-1] == '\n':
                    model_fname = model_fname[:-1]
        # noinspection PyBroadException
        try:
            if model_fname is None or not os.path.exists(model_fname):
                model_fname = '%s/checkpoint.pth.tar' % self.save_path
                with open(latest_fname, 'w') as fout:
                    fout.write(model_fname + '\n')
            if self.out_log:
                print("=> loading checkpoint '{}'".format(model_fname))

            if torch.cuda.is_available():
                checkpoint = torch.load(model_fname)
            else:
                checkpoint = torch.load(model_fname, map_location='cpu')

            self.net.module.load_state_dict(checkpoint['state_dict'])
            # set new manual seed
            new_manual_seed = int(time.time())
            torch.manual_seed(new_manual_seed)
            torch.cuda.manual_seed_all(new_manual_seed)
            np.random.seed(new_manual_seed)

            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch'] + 1
            if 'best_acc' in checkpoint:
                self.best_acc = checkpoint['best_acc']
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])

            if self.out_log:
                print("=> loaded checkpoint '{}'".format(model_fname))
        except Exception:
            if self.out_log:
                print('fail to load checkpoint from %s' % self.save_path)

    def save_config(self, print_info=True):
        """ dump run_config and net_config to the model_folder """
        os.makedirs(self.path, exist_ok=True)
        net_save_path = os.path.join(self.path, 'net.config')
        json.dump(self.net.module.config, open(net_save_path, 'w'), indent=4)
        if print_info:
            print('Network configs dump to %s' % net_save_path)

        run_save_path = os.path.join(self.path, 'run.config')
        json.dump(self.run_config.config, open(run_save_path, 'w'), indent=4)
        if print_info:
            print('Run configs dump to %s' % run_save_path)

    """ train and test """

    def write_log(self, log_str, prefix,version='default', should_print=True):
        """ prefix: valid, train, test """
        if prefix in ['valid', 'test']:
            with open(os.path.join(self.logs_path, '%s_valid_console.txt'%version), 'a') as fout:
                fout.write(log_str + '\n')
                fout.flush()
        if prefix in ['valid', 'test', 'train']:
            with open(os.path.join(self.logs_path, '%s_train_console.txt'%version), 'a') as fout:
                if prefix in ['valid', 'test']:
                    fout.write('=' * 10)
                fout.write(log_str + '\n')
                fout.flush()
        if should_print:
            print(log_str)

    def validate(self, is_test=False, net=None, use_train_mode=False):
        

        # Define the data_loader according to the dataset of run_config
        data_loader = self.run_config.val_loader
        __C = self.__C

	    # Get the VQA network 
        if net is None:
            net = self.net

        if use_train_mode:
            net.train()
        else:
            net.eval()

        batch_time = AverageMeter()
        losses = AverageMeter()
        #top1 = AverageMeter()
        #top5 = AverageMeter()

        end = time.time()
        # noinspection PyUnresolvedReferences
        ans_ix_list = []
        pred_list = []
        with torch.no_grad():
        	# get the batch of data 
            for i, (frcn_feat_iter,
                bbox_feat_iter,
                ques_ix_iter,
                ans_iter) in enumerate(data_loader):
            
                #print (frcn_feat_iter.size(),bbox_feat_iter.size(),ques_ix_iter.size(),ans_iter.size())
            	# load the data to GPU
                frcn_feat_iter,  bbox_feat_iter, ques_ix_iter, ans_iter = frcn_feat_iter.to(self.device),\
                 bbox_feat_iter.to(self.device), ques_ix_iter.to(self.device), ans_iter.to(self.device) 

                print("\rEvaluation: [step %4d/%4d]" % (
		            i,
		            int(self.run_config.dataset.val_data_size / self.__C.EVAL_BATCH_SIZE),
		        ), end='          ')

                # obtain the prediction, and get the argmax prediction
                pred = net(
				            frcn_feat_iter,
				            #grid_feat_iter,
				            bbox_feat_iter,
				            ques_ix_iter
				        )
                pred_np = pred.cpu().data.numpy()
                pred_argmax = np.argmax(pred_np, axis=1)
               
                if pred_argmax.shape[0] != __C.EVAL_BATCH_SIZE:
                    pred_argmax = np.pad(
                        pred_argmax,
                        (0, __C.EVAL_BATCH_SIZE - pred_argmax.shape[0]),
                        mode='constant',
                        constant_values=-1
                    )
                                
                if pred_np.shape[0] != __C.EVAL_BATCH_SIZE:
                    pred_np = np.pad(
                        pred_np,
                        ((0, __C.EVAL_BATCH_SIZE - pred_np.shape[0]), (0, 0)),
                        mode='constant',
                        constant_values=-1
                    )

                pred_list.append(pred_np)

                ans_ix_list.append(pred_argmax)
                #print (pred.size())
                #print (ans_iter.size())
                if self.__C.LOSS_FUNC=='ce':
                    ans_iter=ans_iter.view(-1)
                loss = self.criterion(pred,ans_iter)
                losses.update(loss, frcn_feat_iter.size(0))

                batch_time.update(time.time() - end)
                end = time.time()
        ans_ix_list = np.array(ans_ix_list).reshape(-1)
        print (ans_ix_list.shape)
        # get the eval_file path 
        if not is_test:
	        if __C.RUN_MODE not in ['train']:
	            result_eval_file = __C.CACHE_PATH + '/result_run_' + __C.CKPT_VERSION
	        else:
	            result_eval_file = __C.CACHE_PATH + '/result_run_' + __C.VERSION
        else:
            if __C.CKPT_PATH is not None:
                result_eval_file = __C.RESULT_PATH + '/result_run_' + __C.CKPT_VERSION
            else:
                result_eval_file = __C.RESULT_PATH + '/result_run_' + __C.CKPT_VERSION + '_epoch' + str(__C.CKPT_EPOCH)
        if __C.CKPT_PATH is not None:
            ensemble_file = __C.PRED_PATH + '/result_run_' + __C.CKPT_VERSION + '.pkl'
        else:
            ensemble_file = __C.PRED_PATH + '/result_run_' + __C.CKPT_VERSION + '_epoch' + str(__C.CKPT_EPOCH) + '.pkl'

        if __C.RUN_MODE not in ['train']:
            log_file = __C.LOG_PATH + '/log_run_' + __C.CKPT_VERSION + '.txt'
        else:
            log_file = __C.LOG_PATH + '/log_run_' + __C.VERSION + '.txt'

        accuracy = EvalLoader(__C).eval(self.run_config.dataset.val_set, ans_ix_list, result_eval_file, log_file, True)

        
        return losses.avg, accuracy 

    def train_one_epoch(self, train_loader, adjust_lr_func, train_log_func):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        #top1 = AverageMeter()
        #top5 = AverageMeter()

        # switch to train mode

        self.net.train()

        #dataset = self.run_config.train_set 
        #data_loader 
        end = time.time()

        __C = self.__C
        

        for step, (
                frcn_feat_iter,
                #grid_feat_iter,
                bbox_feat_iter,
                ques_ix_iter,
                ans_iter
        ) in enumerate(train_loader):

            data_time.update(time.time() - end)
            #new_lr = adjust_lr_func(i)
            frcn_feat_iter,  bbox_feat_iter, ques_ix_iter, ans_iter = frcn_feat_iter.to(self.device), \
                     bbox_feat_iter.to(self.device), ques_ix_iter.to(self.device), ans_iter.to(self.device) 
            new_lr = adjust_lr_func(step)
            pred = net(
				            frcn_feat_iter,
				            #grid_feat_iter,
				            bbox_feat_iter,
				            ques_ix_iter
				        )
            loss = self.criterion(loss,ans_iter)

            #

            # measure accuracy and record loss
            #acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            losses.update(loss, images.size(0))
            #top1.update(acc1[0], images.size(0))
            #top5.update(acc5[0], images.size(0))

            # compute gradient and do SGD step
            self.net.zero_grad()  # or self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.run_config.print_frequency == 0 or i + 1 == len(self.run_config.train_loader):
                batch_log = train_log_func(i, batch_time, data_time, losses, new_lr)
                self.write_log(batch_log, 'train')
        return losses.avg

    def train(self):
    	# initialize the train data_loader 
        #dataset = self.run_config.train_set 
        __C = self.__C
        train_loader = self.run_config.train_loader

        nBatch = len(train_loader)
        # log function 
        def train_log_func(epoch_, i, batch_time, data_time, losses, lr):
            batch_log = 'Train [{0}][{1}/{2}]\t' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                        'Loss {losses.val:.4f} ({losses.avg:.4f})'.format(epoch_ + 1, i, nBatch - 1,
                       batch_time=batch_time, data_time=data_time, losses=losses, top1=top1)
            
            batch_log += '\tlr {lr:.5f}'.format(lr=lr)
            return batch_log

        for epoch in range(self.start_epoch, self.run_config.n_epochs):
            print('\n', '-' * 30, 'Train epoch: %d' % (epoch + 1), '-' * 30, '\n')

            end = time.time()
            loss_avg = self.train_one_epoch( train_loader, 
                lambda i: self.run_config.adjust_learning_rate(self.optimizer, epoch, i, nBatch),
                lambda i, batch_time, data_time, losses, top1, top5, new_lr:
                train_log_func(epoch, i, batch_time, data_time, losses, top1, top5, new_lr),
            )
            time_per_epoch = time.time() - end
            seconds_left = int((self.run_config.n_epochs - epoch - 1) * time_per_epoch)
            print('Time per epoch: %s, Est. complete in: %s' % (
                str(timedelta(seconds=time_per_epoch)),
                str(timedelta(seconds=seconds_left))))

            if (epoch + 1) % self.run_config.validation_frequency == 0:
                val_loss, val_acc = self.validate(is_test=False, return_top5=True)
                is_best = val_acc > self.best_acc
                self.best_acc = max(self.best_acc, val_acc)
                val_log = 'Valid [{0}/{1}]\tloss {2:.3f}\ttop-1 acc {3:.3f} ({4:.3f})'.\
                    format(epoch + 1, self.run_config.n_epochs, val_loss, val_acc, self.best_acc)
                
                val_log += '\tTrain top-1 {top1.avg:.3f}'.format(top1=train_top1)
                self.write_log(val_log, 'valid')
            else:
                is_best = False

            self.save_model({
                'epoch': epoch,
                'best_acc': self.best_acc,
                'optimizer': self.optimizer.state_dict(),
                'state_dict': self.net.module.state_dict(),
            }, is_best=is_best)
