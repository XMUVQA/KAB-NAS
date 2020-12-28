# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.
import numpy as np
from vqa_run_manager import *
import random

def calculate_accuracy_score(predict, label):
    predict = predict.cpu().numpy()
    label = label.cpu().numpy()
    index_list = np.argmax(predict,axis=1).tolist()
    score=0
    for i in range (len(index_list)):
        score+= label[i][index_list[i]]
    score = score/len(index_list)
    return score





class ArchSearchConfig:

    def __init__(self, arch_init_type, arch_init_ratio, arch_opt_type, arch_lr,
                 arch_opt_param, arch_weight_decay):
        """ architecture parameters initialization & optimizer """
        self.arch_init_type = arch_init_type
        self.arch_init_ratio = arch_init_ratio

        self.opt_type = arch_opt_type
        self.lr = arch_lr
        self.opt_param = {} if arch_opt_param is None else arch_opt_param
        self.weight_decay = arch_weight_decay
        #self.target_hardware = target_hardware
        #self.ref_value = ref_value

    @property
    def config(self):
        config = {
            'type': type(self),
        }
        for key in self.__dict__:
            if not key.startswith('_'):
                config[key] = self.__dict__[key]
        return config

    def get_update_schedule(self, nBatch):
        raise NotImplementedError
    """
    def build_optimizer(self, params):
        

        :param params: architecture parameters
        :return: arch_optimizer
        
        if self.opt_type == 'adam':
            return torch.optim.Adam(
                params, self.lr, weight_decay=self.weight_decay, **self.opt_param
            )
        else:
            raise NotImplementedError
    """
class KabArchSearchConfig(ArchSearchConfig):

    def __init__(self, arch_init_type='normal', arch_init_ratio=1e-3, arch_opt_type='adam', arch_lr=1e-3,
                 arch_opt_param=None, arch_weight_decay=0, 
                 grad_update_arch_param_every=1, grad_update_steps=1, grad_binary_mode='full', 
                grad_reg_loss_params=None, **kwargs):
        super(KabArchSearchConfig, self).__init__(
            arch_init_type, arch_init_ratio, arch_opt_type, arch_lr, arch_opt_param, arch_weight_decay,
            
        )

        self.update_arch_param_every = grad_update_arch_param_every
        self.update_steps = grad_update_steps
        self.binary_mode = grad_binary_mode
        #self.data_batch = grad_data_batch

        #self.reg_loss_type = grad_reg_loss_type
        #self.reg_loss_params = {} if grad_reg_loss_params is None else grad_reg_loss_params

        print(kwargs.keys())

    def get_update_schedule(self, nBatch):
        schedule = {}
        for i in range(nBatch):
            if (i + 1) % self.update_arch_param_every == 0:
                schedule[i] = self.update_steps
        return schedule

    #def add_regularization_loss(self, ce_loss):
    #    return ce_loss
"""
class GradientArchSearchConfig(ArchSearchConfig):

    def __init__(self, arch_init_type='normal', arch_init_ratio=1e-3, arch_opt_type='adam', arch_lr=1e-3,
                 arch_opt_param=None, arch_weight_decay=0, 
                 grad_update_arch_param_every=1, grad_update_steps=1, grad_binary_mode='full', grad_data_batch=None,
                 grad_reg_loss_type=None, grad_reg_loss_params=None, **kwargs):
        super(GradientArchSearchConfig, self).__init__(
            arch_init_type, arch_init_ratio, arch_opt_type, arch_lr, arch_opt_param, arch_weight_decay,
            
        )

        self.update_arch_param_every = grad_update_arch_param_every
        self.update_steps = grad_update_steps
        self.binary_mode = grad_binary_mode
        self.data_batch = grad_data_batch

        self.reg_loss_type = grad_reg_loss_type
        self.reg_loss_params = {} if grad_reg_loss_params is None else grad_reg_loss_params

        print(kwargs.keys())

    def get_update_schedule(self, nBatch):
        schedule = {}
        for i in range(nBatch):
            if (i + 1) % self.update_arch_param_every == 0:
                schedule[i] = self.update_steps
        return schedule

    def add_regularization_loss(self, ce_loss):
        return ce_loss
"""




class ArchSearchRunManager:

    def __init__(self,__C, path, super_net, run_config: VRunConfig, arch_search_config: ArchSearchConfig):
        # init weight parameters & build weight_optimizer
        self.run_manager = VQARunManager(__C, path, super_net, run_config, True)

        self.arch_search_config = arch_search_config
        self.version = random.randint(0, 9999999)

        # init architecture parameters
        self.net.init_arch_params(
            self.arch_search_config.arch_init_type, self.arch_search_config.arch_init_ratio,
        )

        # build architecture optimizer
        #self.arch_optimizer = self.arch_search_config.build_optimizer(self.net.architecture_parameters())
        self.__C =__C
        self.warmup = True
        self.warmup_epoch = 0
        #self.with_enc = False

    @property
    def net(self):
        return self.run_manager.net.module

    def write_log(self, log_str, prefix, should_print=True, end='\n'):
        with open(os.path.join(self.run_manager.logs_path, '%s-%s.log' % (self.version,prefix)), 'a') as fout:
            fout.write(log_str + end)
            fout.flush()
        if should_print:
            print(log_str)

    def load_model(self, model_fname=None):
        latest_fname = os.path.join(self.run_manager.save_path, 'latest.txt')
        if model_fname is None and os.path.exists(latest_fname):
            with open(latest_fname, 'r') as fin:
                model_fname = fin.readline()
                if model_fname[-1] == '\n':
                    model_fname = model_fname[:-1]

        if model_fname is None or not os.path.exists(model_fname):
            model_fname = '%s/checkpoint.pth.tar' % self.run_manager.save_path
            with open(latest_fname, 'w') as fout:
                fout.write(model_fname + '\n')
        if self.run_manager.out_log:
            print("=> loading checkpoint '{}'".format(model_fname))

        if torch.cuda.is_available():
            checkpoint = torch.load(model_fname)
        else:
            checkpoint = torch.load(model_fname, map_location='cpu')

        model_dict = self.net.state_dict()
        model_dict.update(checkpoint['state_dict'])
        self.net.load_state_dict(model_dict)
        if self.run_manager.out_log:
            print("=> loaded checkpoint '{}'".format(model_fname))

        # set new manual seed
        new_manual_seed = int(time.time())
        torch.manual_seed(new_manual_seed)
        torch.cuda.manual_seed_all(new_manual_seed)
        np.random.seed(new_manual_seed)

        if 'epoch' in checkpoint:
            self.run_manager.start_epoch = checkpoint['epoch'] + 1
        if 'weight_optimizer' in checkpoint:
            self.run_manager.optimizer.load_state_dict(checkpoint['weight_optimizer'])
        if 'arch_optimizer' in checkpoint:
            self.arch_optimizer.load_state_dict(checkpoint['arch_optimizer'])
        if 'warmup' in checkpoint:
            self.warmup = checkpoint['warmup']
        if self.warmup and 'warmup_epoch' in checkpoint:
            self.warmup_epoch = checkpoint['warmup_epoch']

    """ training related methods """

    def validate(self):
        # get performances of current chosen network on validation set
        #self.run_manager.run_config.valid_loader.batch_sampler.batch_size = self.run_manager.run_config.test_batch_size
        #self.run_manager.run_config.valid_loader.batch_sampler.drop_last = False

        # set chosen op active
        self.net.set_chosen_op_active()
        # remove unused modules
        self.net.unused_modules_off()
        # test on validation set under train mode
        valid_res = self.run_manager.validate(is_test=False, use_train_mode=False)
        # unused modules back
        self.net.unused_modules_back()
        
        return valid_res

    def warm_up(self, warmup_epochs=10):
        lr_max = self.__C.WARMUP_LR 
        data_loader = self.run_manager.run_config.train_loader
        nBatch = len(data_loader)
        T_total = warmup_epochs * nBatch
        #val_loss, val_acc = self.validate()
        for epoch in range(self.warmup_epoch, warmup_epochs):
            print('\n', '-' * 30, 'Warmup epoch: %d' % (epoch + 1), '-' * 30, '\n')
            batch_time = AverageMeter()
            epoch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()

            # switch to train mode
            self.run_manager.net.train()

            end = time.time()
            epoch_end = time.time()
            for i, (frcn_feat_iter,
                bbox_feat_iter,
                ques_ix_iter,
                ans_iter) in enumerate(data_loader):

                data_time.update(time.time() - end)
                # lr
                #T_cur = epoch * nBatch + i
                # lr schedule schame 
                T_cur = epoch * nBatch + i
                if T_cur <= int(nBatch * (warmup_epochs + 1) * 0.25):
                    warmup_lr = lr_max* 0.25
                elif T_cur <= int(nBatch * (warmup_epochs + 1) * 0.5):
                    warmup_lr = lr_max * 0.5
                elif T_cur <= int(nBatch * (warmup_epochs + 1) * 0.75):
                    warmup_lr = lr_max * 0.75
                else:
                    warmup_lr=lr_max

                for param_group in self.run_manager.optimizer.param_groups:
                    param_group['lr'] = warmup_lr


                frcn_feat_iter,  bbox_feat_iter, ques_ix_iter, ans_iter = frcn_feat_iter.to(self.run_manager.device), \
                     bbox_feat_iter.to(self.run_manager.device), ques_ix_iter.to(self.run_manager.device), ans_iter.to(self.run_manager.device) 
                #print (ans_iter.size())
                print("\rEvaluation: [step %4d/%4d]" % (
                    i,
                    int(self.run_manager.run_config.dataset.data_size / self.__C.EVAL_BATCH_SIZE),
                ), end='          ')
                # compute output
                self.net.random_sampling()  # random sample binary gates
                self.net.unused_modules_off()  # remove unused module for speedup
                pred = self.run_manager.net(
                            frcn_feat_iter,
                            #grid_feat_iter,
                            bbox_feat_iter,
                            ques_ix_iter
                        )
                #print (pred.size(),ans_iter.size())
                #print (self.__C.LOSS_FUNC)
                if self.__C.LOSS_FUNC=='ce':
                    ans_iter=ans_iter.view(-1)
                loss = self.run_manager.criterion(pred,ans_iter)
                # loss
                
                # record loss
                losses.update(loss/frcn_feat_iter.size(0), frcn_feat_iter.size(0))
                # compute gradient and do SGD step
                self.run_manager.net.zero_grad()  # zero grads of weight_param, arch_param & binary_param
                loss.backward()
                self.run_manager.optimizer.step()  # update weight parameters
                # unused modules back
                self.net.unused_modules_back()
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.run_manager.run_config.print_frequency == 0 or i + 1 == nBatch:
                    batch_log = 'Warmup Train [{0}][{1}/{2}]\t' \
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                                'Loss {losses.val:.4f} ({losses.avg:.4f})\t lr {lr:.5f}'.format(epoch + 1, i, nBatch - 1, batch_time=batch_time, data_time=data_time,
                               losses=losses,  lr=warmup_lr)
                    self.run_manager.write_log(batch_log, 'train',self.version)

            # validation
            val_loss, val_acc = self.validate()
            val_log = 'Warmup Valid [{0}/{1}]\tloss {2:.3f}\tAcc {3:.3f}\t'.format(epoch + 1, warmup_epochs, val_loss, val_acc)
            
            self.run_manager.write_log(val_log, 'valid',self.version)
            self.warmup = epoch + 1 < warmup_epochs

            epoch_time.update(time.time()-epoch_end)
            epoch_log = 'Epoch Time {epoch_time.val:.3f} ({epoch_time.avg: .3f})\t'.format(epoch_time=epoch_time)
            self.run_manager.write_log(epoch_log,'train',self.version)

            state_dict = self.net.state_dict()
            # rm architecture parameters & binary gates
            for key in list(state_dict.keys()):
                if 'AP_path_alpha' in key or 'AP_path_wb' in key:
                    state_dict.pop(key)
            checkpoint = {
                'state_dict': state_dict,
                'warmup': self.warmup,
            }
            if self.warmup:
                checkpoint['warmup_epoch'] = epoch,
            self.run_manager.save_model(checkpoint, model_name='warmup.pth.tar')

    def train(self, fix_net_weights=False):
        data_loader = self.run_manager.run_config.train_loader
        nBatch = len(data_loader)
        if fix_net_weights:
            data_loader = [(0, 0)] * nBatch

        arch_param_num = len(list(self.net.architecture_parameters()))
        weight_param_num = len(list(self.net.weight_parameters()))
        print(
            '#arch_params: %d\t#weight_params: %d' %
            (arch_param_num,  weight_param_num)
        )

        update_schedule = self.arch_search_config.get_update_schedule(nBatch)

        for epoch in range(self.run_manager.start_epoch, self.run_manager.run_config.n_epochs):
            print('\n', '-' * 30, 'Train epoch: %d' % (epoch + 1), '-' * 30, '\n')
            
            batch_time = AverageMeter()
            epoch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            entropy = AverageMeter()

            # switch to train mode
            self.run_manager.net.train()

            end = time.time()
            epoch_end= time.time()
            for i, (
                    frcn_feat_iter,
                    #grid_feat_iter,
                    bbox_feat_iter,
                    ques_ix_iter,
                    ans_iter
            ) in enumerate(data_loader):

                data_time.update(time.time() - end)
                # lr
                lr = self.run_manager.run_config.adjust_learning_rate(
                    self.run_manager.optimizer, epoch, batch=i, nBatch=nBatch
                )
                # network entropy
                net_entropy = self.net.entropy()
                entropy.update(net_entropy.data.item() / arch_param_num, 1)
                # train weight parameters if not fix_net_weights
                if not fix_net_weights:
                    frcn_feat_iter,  bbox_feat_iter, ques_ix_iter, ans_iter = frcn_feat_iter.to(self.run_manager.device), \
                     bbox_feat_iter.to(self.run_manager.device), ques_ix_iter.to(self.run_manager.device), ans_iter.to(self.run_manager.device) 
                    # compute output
                    self.net.random_sampling()  # random sample binary gates
                    self.net.unused_modules_off()  # remove unused module for speedup
                    pred = self.run_manager.net(
                            frcn_feat_iter,
                            #grid_feat_iter,
                            bbox_feat_iter,
                            ques_ix_iter
                        )
                    # loss
                    if self.__C.LOSS_FUNC=='ce':
                        ans_iter=ans_iter.view(-1)
                    loss = self.run_manager.criterion(pred,ans_iter)
                    
                    # measure accuracy and record loss
                    
                    losses.update(loss/frcn_feat_iter.size(0), frcn_feat_iter.size(0))
                    
                    # compute gradient and do SGD step
                    self.run_manager.net.zero_grad()  # zero grads of weight_param, arch_param & binary_param
                    loss.backward()
                    self.run_manager.optimizer.step()  # update weight parameters
                    # unused modules back
                    self.net.unused_modules_back()
                # skip architecture parameter updates in the first epoch
                if epoch >= 0:
                    # update architecture parameters according to update_schedule
                    for j in range(update_schedule.get(i, 0)):
                        start_time = time.time()
                        ave_ce_loss,ave_score,ave_arch_loss, max_reward = self.kab_update_step()
                        used_time = time.time() - start_time
                        log_str = 'Architecture [%d-%d]\t Time %.4f\t Arch Loss %.4f\t CE Loss %.4f\t Ave Score %.4f\t Max Reward %.4f\t' % \
                                  (epoch + 1, i, used_time, ave_arch_loss, ave_ce_loss,ave_score,max_reward)
                        self.write_log(log_str, prefix='gradient', should_print=False)
                        
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                # training log
                if i % self.run_manager.run_config.print_frequency == 0 or i + 1 == nBatch:
                    batch_log = 'Train [{0}][{1}/{2}]\t' \
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                                'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                                'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                                'Entropy {entropy.val:.5f} ({entropy.avg:.5f})\t lr {lr:.5f}'.format(
                                    epoch + 1, i, nBatch - 1, batch_time=batch_time, data_time=data_time,
                               losses=losses, entropy=entropy,lr=lr)
                    self.run_manager.write_log(batch_log, 'train',self.version)

                #epoch_end=time.time()

                #print ('%d-epcoh, the complete time is %.4f'%(epoch,(epoch_end- epoch_start)))

            # print current network architecture
            self.write_log('-' * 30 + 'Current Architecture [%d]' % (epoch + 1) + '-' * 30, prefix='arch')
            """
            if self.with_enc:
                for idx, block in enumerate(self.net.lang_layers):
                    self.write_log('%d. %s' % (idx, block.module_full_str), prefix='arch')
            """
            for idx, block in enumerate(self.net.graph_layers):
                self.write_log('%d. %s' % (idx, block.module_full_str), prefix='arch')
            self.write_log('-' * 60, prefix='arch')

            # validate
            if (epoch + 1) % self.run_manager.run_config.validation_frequency == 0:
                val_loss, val_top1 = self.validate()
                self.run_manager.best_acc = max(self.run_manager.best_acc, val_top1)
                val_log = 'Valid [{0}/{1}]\tloss {2:.3f}\ttop-1 acc {3:.3f} ({4:.3f})'.format(epoch + 1, self.run_manager.run_config.n_epochs, val_loss, val_top1,
                           self.run_manager.best_acc)
                self.run_manager.write_log(val_log, 'valid',self.version)
            # save model
            self.run_manager.save_model({
                'warmup': False,
                'epoch': epoch,
                'weight_optimizer': self.run_manager.optimizer.state_dict(),
                #'arch_optimizer': self.arch_optimizer.state_dict(),
                'state_dict': self.net.state_dict()
            })
            epoch_time.update(time.time()-epoch_end)
            epoch_end = time.time()
            epoch_log = 'Epoch Time {epoch_time.val:.3f} ({epoch_time.avg:.3f})\t '.format(epoch_time=epoch_time)
            self.run_manager.write_log(epoch_log,'train',self.version)
        # convert to normal network according to architecture parameters
        normal_net = self.net.cpu().convert_to_normal_net()
        print('Total training params: %.2fM' % (count_parameters(normal_net) / 1e6))
        os.makedirs(os.path.join(self.run_manager.path, 'learned_net'), exist_ok=True)
        json.dump(normal_net.config, open(os.path.join(self.run_manager.path, 'learned_net/net.config'), 'w'), indent=4)
        json.dump(
            self.run_manager.run_config.config,
            open(os.path.join(self.run_manager.path, 'learned_net/run.config'), 'w'), indent=4,
        )
        torch.save(
            {'state_dict': normal_net.state_dict(), 'dataset': self.run_manager.run_config.dataset_name},
            os.path.join(self.run_manager.path, 'learned_net/init')
        )


    


    def kab_update_step(self, fast=True):
        #assert isinstance(self.arch_search_config, RLArchSearchConfig)
        # prepare data
        self.run_manager.net.train()
        # Mix edge mode
        MixedEdge.MODE = self.arch_search_config.binary_mode
        time1 = time.time()  # time
        # sample a batch of data from validation set
        frcn_feat_iter,bbox_feat_iter,ques_ix_iter, ans_iter = self.run_manager.run_config.valid_next_batch
        frcn_feat_iter,  bbox_feat_iter, ques_ix_iter, ans_iter = frcn_feat_iter.to(self.run_manager.device), \
                     bbox_feat_iter.to(self.run_manager.device), ques_ix_iter.to(self.run_manager.device), ans_iter.to(self.run_manager.device) 
        time2 = time.time()  # time
        if self.__C.LOSS_FUNC=='ce':
            ans_iter=ans_iter.view(-1)
        grad_buffer = []
        reward_buffer = []
        net_info_buffer = []
        ce_loss_buffer = []
        score_buffer = []
        arch_loss_buffer = []
        batch_size = 5
        active_index_buffer = []
        up_weight_buffer = []


        for i in range(batch_size):
            self.net.reset_binary_gates()  # random sample binary gates
            self.net.unused_modules_off()  # remove unused module for speedup
            # validate the sampled network
            with torch.no_grad():
                pred = self.run_manager.net(
                            frcn_feat_iter,
                            #grid_feat_iter,
                            bbox_feat_iter,
                            ques_ix_iter
                        )
            # loss
                ce_loss = self.run_manager.criterion(pred, ans_iter)
                if self.__C.LOSS_FUNC=='ce':
                    score= accuracy(pred, ans_iter)[0] 
                    score = score.cpu().numpy()[0] 
                else:
                    score = calculate_accuracy_score(pred,ans_iter)*100.0
                #print (score) 

            ce_loss_buffer.append(ce_loss)
            score_buffer.append(score)
            arch_loss = 0

            # calcualte update term 
            # pi(active_index)(1-pi(active_index))
            active_index_list=[]
            up_weight_list = []
            for m in self.net.redundant_modules:
                sample = m.active_index[0]
                active_index_list.append(sample)
                prob= m.probs_over_ops[sample]
                up_weight= prob*(1-prob)
                up_weight_list.append(up_weight)
            active_index_buffer.append(active_index_list)
            up_weight_buffer.append(up_weight_list)

            for m in self.net.redundant_modules:
                if m.AP_path_alpha.grad is not None:
                    m.AP_path_alpha.grad.data.zero_()
                arch_loss = arch_loss + m.log_prob
            #arch_loss = self.run_manager.net.calculate_log_entropies()
            arch_loss = -arch_loss
            arch_loss_buffer.append(arch_loss)
            

            self.net.unused_modules_back()


            #net_info = {'acc': acc1[0].item()}
            # get additional net info for calculating the reward
        ave_ce_loss = sum(ce_loss_buffer)/batch_size
        ave_arch_loss = sum(arch_loss_buffer)/batch_size
        ave_score = sum(score_buffer)/batch_size

        for i in range(batch_size):

            #reward = (ave_ce_loss- ce_loss_buffer[i])/ (ave_ce_loss+ce_loss_buffer[i])
            reward = score_buffer[i]-ave_score 
            reward_buffer.append(reward)

        for idx, m in enumerate(self.net.redundant_modules):
            #m.AP_path_alpha.grad.data.zero_()
            for j in range(batch_size):
                sample = active_index_buffer[j][idx]
                m.AP_path_alpha[sample] += self.arch_search_config.lr*reward_buffer[j]*up_weight_buffer[j][idx]
            #print (m.AP_path_alpha)
                #m.AP_path_alpha.grad.data += reward_buffer[j] * grad_buffer[j][idx]
            #m.AP_path_alpha.grad.data /= batch_size
        #self.arch_optimizer.step()
        #ave_reward = sum(reward_buffer)/len(reward_buffer)
        max_reward = max(reward_buffer)

        return ave_ce_loss,ave_score,ave_arch_loss, max_reward



        

    
    