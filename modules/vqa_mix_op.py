# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import numpy as np

from torch.nn.parameter import Parameter
import torch.nn.functional as F

from modules.layers import *
from modules.vqa_layers import *
import random

def build_enc_candidate_ops(candidate_ops, in_channels,mid_channels, out_channels):
    if candidate_ops is None:
        raise ValueError('please specify a candidate set')

    name2ops = {
        
        'Zero': lambda in_C, out_C, S: ZeroLayer(stride=S),
    }
    # add MBConv layers
    name2ops = {
        'ZeroLayer': lambda in_C, mid_C, out_C: ZeroLayer(),
         #######################################################################################
        'FullGraphLayer_2': lambda in_C, mid_C, out_C: FullGraphLayer(in_C,mid_C, out_C, 2, 0.1),
        'FullGraphLayer_4': lambda in_C, mid_C, out_C: FullGraphLayer(in_C,mid_C, out_C, 4, 0.1),
        'FullGraphLayer_8': lambda in_C, mid_C, out_C: FullGraphLayer(in_C,mid_C, out_C, 8, 0.1),
        #######################################################################################
        
    }
    

    return [
        name2ops[name](in_channels, mid_channels,out_channels) for name in candidate_ops
    ]
def build_candidate_ops(candidate_ops, in_channels,mid_channels, out_channels):
    if candidate_ops is None:
        raise ValueError('please specify a candidate set')

    name2ops = {
        
        'Zero': lambda in_C, out_C, S: ZeroLayer(stride=S),
    }
    # add MBConv layers
    name2ops = {
        'ZeroLayer': lambda in_C, mid_C, out_C: ZeroLayer(),
         #######################################################################################
        'FullGraphLayer_2': lambda in_C, mid_C, out_C: FullGraphLayer(in_C,mid_C, out_C, 2, 0.1),
        'FullGraphLayer_4': lambda in_C, mid_C, out_C: FullGraphLayer(in_C,mid_C, out_C, 4, 0.1),
        'FullGraphLayer_8': lambda in_C, mid_C, out_C: FullGraphLayer(in_C,mid_C, out_C, 8, 0.1),
        #######################################################################################
        'SEGraphLayer_2': lambda in_C, mid_C, out_C: SEGraphLayer(in_C,mid_C, out_C, 2, 0.1),
       	'SEGraphLayer_4': lambda in_C, mid_C, out_C: SEGraphLayer(in_C,mid_C, out_C, 4, 0.1),
       	'SEGraphLayer_8': lambda in_C, mid_C, out_C: SEGraphLayer(in_C,mid_C, out_C, 8, 0.1),
        #######################################################################################
        'CoGraphLayer_2': lambda in_C, mid_C, out_C: CoGraphLayer(in_C,mid_C, out_C, 2, 0.1),
       	'CoGraphLayer_4': lambda in_C, mid_C, out_C: CoGraphLayer(in_C,mid_C, out_C, 4, 0.1),
       	'CoGraphLayer_8': lambda in_C, mid_C, out_C: CoGraphLayer(in_C,mid_C, out_C, 8, 0.1),
        
    }
    

    return [
        name2ops[name](in_channels, mid_channels,out_channels) for name in candidate_ops
    ]


class MixedEdge(MyModule):
    MODE = None  # full, two, None, full_v2

    def __init__(self, candidate_ops):
        super(MixedEdge, self).__init__()

        self.candidate_ops = nn.ModuleList(candidate_ops)
        self.AP_path_alpha = Parameter(torch.Tensor(self.n_choices))  # architecture parameters
        #self.AP_path_wb = Parameter(torch.Tensor(self.n_choices))  # binary gates

        self.active_index = [0]
        self.inactive_index = None

        self.log_prob = None
        self.current_prob_over_ops = None

    @property
    def n_choices(self):
        return len(self.candidate_ops)

    @property
    def probs_over_ops(self):
        probs = F.softmax(self.AP_path_alpha, dim=0)  # softmax to probability
        return probs

    @property
    def chosen_index(self):
        probs = self.probs_over_ops.data.cpu().numpy()
        index = int(np.argmax(probs))
        return index, probs[index]

    @property
    def chosen_op(self):
        index, _ = self.chosen_index
        return self.candidate_ops[index]

    @property
    def random_op(self):
        index = np.random.choice([_i for _i in range(self.n_choices)], 1)[0]
        return self.candidate_ops[index]

    def entropy(self, eps=1e-8):
        probs = self.probs_over_ops
        log_probs = torch.log(probs + eps)
        entropy = - torch.sum(torch.mul(probs, log_probs))
        return entropy

    def is_zero_layer(self):
        return self.active_op.is_zero_layer()

    @property
    def active_op(self):
        """ assume only one path is active """
        return self.candidate_ops[self.active_index[0]]

    def set_chosen_op_active(self):
        chosen_idx, _ = self.chosen_index
        self.active_index = [chosen_idx]
        self.inactive_index = [_i for _i in range(0, chosen_idx)] + \
                              [_i for _i in range(chosen_idx + 1, self.n_choices)]

    """ """

    def forward(self, x,x_mask):
        
        output = self.active_op(x,x_mask)
        return output

    @property
    def module_str(self):
        chosen_index, probs = self.chosen_index
        return 'Mix(%s, %.3f)' % (self.candidate_ops[chosen_index].module_str, probs)

    @property
    def module_full_str(self):
        full_str=''
        for i in range(self.n_choices):
            full_str +='Mix(%s,%.3f) '%(self.candidate_ops[i].module_str,self.probs_over_ops[i])
        return full_str

    @property
    def config(self):
        raise ValueError('not needed')

    @staticmethod
    def build_from_config(config):
        raise ValueError('not needed')

    """
    def get_flops(self, x):
        # Only active paths taken into consideration when calculating FLOPs 
        flops = 0
        for i in self.active_index:
            delta_flop, _ = self.candidate_ops[i].get_flops(x)
            flops += delta_flop
        return flops, self.forward(x)


    """ 
    """
    """ 

    def randomize(self):
        self.log_prob = None
        # reset binary gates
        #self.AP_path_wb.data.zero_()
        # binarize according to probs
        probs = self.probs_over_ops
        
        sample = random.randint(0,self.n_choices-1)
        self.active_index = [sample]
        self.inactive_index = [_i for _i in range(0, sample)] + \
                              [_i for _i in range(sample + 1, self.n_choices)]
        self.log_prob = torch.log(probs[sample])
        self.current_prob_over_ops = probs
        # set binary gate
        # self.AP_path_wb.data[sample] = 1.0
        # avoid over-regularization

        for _i in range(self.n_choices):
            for name, param in self.candidate_ops[_i].named_parameters():
                param.grad = None

    def binarize(self):
        """ prepare: active_index, inactive_index, AP_path_wb, log_prob (optional), current_prob_over_ops (optional) """
        self.log_prob = None
        # reset binary gates
        #self.AP_path_wb.data.zero_()
        # binarize according to probs
        probs = self.probs_over_ops
        if random.randint(0,5)==1:
            sample = random.randint(0,self.n_choices-1)
        sample = torch.multinomial(probs.data, 1)[0].item()
        self.active_index = [sample]
        self.inactive_index = [_i for _i in range(0, sample)] + \
                              [_i for _i in range(sample + 1, self.n_choices)]
        self.log_prob = torch.log(probs[sample])
        self.current_prob_over_ops = probs
        # set binary gate
        # self.AP_path_wb.data[sample] = 1.0
        # avoid over-regularization

        for _i in range(self.n_choices):
            for name, param in self.candidate_ops[_i].named_parameters():
                param.grad = None

    def set_arch_param_grad(self):
        #binary_grads = self.AP_path_wb.grad.data
        
        if self.AP_path_alpha.grad is None:
            self.AP_path_alpha.grad = torch.zeros_like(self.AP_path_alpha.data)
        
        probs = self.probs_over_ops.data
        self.AP_path_alpha.grad.data[self.active_index[0]]+= (1.0-probs[self.active_index[0]])
        """
        for i in range(self.n_choices):
            for j in range(self.n_choices):
                self.AP_path_alpha.grad.data[i] += binary_grads[j] * probs[j] * (delta_ij(i, j) - probs[i])
        """
        return

    def rescale_updated_arch_param(self):
        if not isinstance(self.active_index[0], tuple):
            assert self.active_op.is_zero_layer()
            return
        involved_idx = [idx for idx, _ in (self.active_index + self.inactive_index)]
        old_alphas = [alpha for _, alpha in (self.active_index + self.inactive_index)]
        new_alphas = [self.AP_path_alpha.data[idx] for idx in involved_idx]

        offset = math.log(
            sum([math.exp(alpha) for alpha in new_alphas]) / sum([math.exp(alpha) for alpha in old_alphas])
        )

        for idx in involved_idx:
            self.AP_path_alpha.data[idx] -= offset


class ArchGradientFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, x_mask, binary_gates, run_func, backward_func):
        ctx.run_func = run_func
        ctx.backward_func = backward_func

        detached_x = detach_variable(x)

        with torch.enable_grad():
            output = run_func(detached_x,x_mask)
        ctx.save_for_backward(detached_x, output)
        return output.data

    @staticmethod
    def backward(ctx, grad_output):
        detached_x, output = ctx.saved_tensors

        grad_x = torch.autograd.grad(output, detached_x, grad_output, only_inputs=True)
        # compute gradients w.r.t. binary_gates
        binary_grads = ctx.backward_func(detached_x.data, output.data, grad_output.data)

        return grad_x[0], binary_grads, None, None

