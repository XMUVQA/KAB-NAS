# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

from queue import Queue
import copy

#from modules.mix_op import *
from modules.vqa_mix_op import *
from models.normal_nets.proxyless_nets import *
from models.normal_nets.vqa_proxyless_nets import *
from utils import LatencyEstimator


class SuperVQAProxylessNASNets(VQAProxylessNASNets):

    def __init__(self, __C, pretrained_emb, token_size, answer_size,graph_candidates):
        self._redundant_modules = None
        self._unused_modules = None

        language_encoder = LanguageEncoder(__C.WORD_EMBED_SIZE,__C.HIDDEN_SIZE) 
        lang_adapter = AdapterLayer(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        img_adapter = ImgAdapterLayer(__C.FEAT_SIZE['vqa']['FRCN_FEAT_SIZE'], __C.HIDDEN_SIZE)

        graph_layers = []
        for i in range (__C.LAYER):
            graph_layer = MixedEdge(candidate_ops = build_candidate_ops(graph_candidates,__C.HIDDEN_SIZE,__C.FF_SIZE,__C.HIDDEN_SIZE)) 
            graph_layers.append(graph_layer)

        att_flat_1 = AttFlatLayer(__C.HIDDEN_SIZE,__C.FLAT_MLP_SIZE,__C.FLAT_OUT_SIZE)
        att_flat_2 = AttFlatLayer(__C.HIDDEN_SIZE,__C.FLAT_MLP_SIZE,__C.FLAT_OUT_SIZE)

        classifier = Classifier(__C.FLAT_OUT_SIZE,answer_size)


        
        super(SuperVQAProxylessNASNets, self).__init__(pretrained_emb, token_size, __C.WORD_EMBED_SIZE, language_encoder, 
            lang_adapter, img_adapter, graph_layers, att_flat_1, att_flat_2, classifier)

        # set bn param
        #self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

    @property
    def config(self):
        raise ValueError('not needed')

    @staticmethod
    def build_from_config(config):
        raise ValueError('not needed')

    """ weight parameters, arch_parameters & binary gates """

    def architecture_parameters(self):
        for name, param in self.named_parameters():
            if 'AP_path_alpha' in name:
                yield param

    def binary_gates(self):
        for name, param in self.named_parameters():
            if 'AP_path_wb' in name:
                yield param

    def weight_parameters(self):
        for name, param in self.named_parameters():
            if 'AP_path_alpha' not in name and 'AP_path_wb' not in name:
                yield param

    """ architecture parameters related methods """

    @property
    def redundant_modules(self):
        if self._redundant_modules is None:
            module_list = []
            for m in self.modules():
                if m.__str__().startswith('MixedEdge'):
                    module_list.append(m)
            self._redundant_modules = module_list
        return self._redundant_modules

    def entropy(self, eps=1e-8):
        entropy = 0
        for m in self.redundant_modules:
            module_entropy = m.entropy(eps=eps)
            entropy = module_entropy + entropy
        return entropy

    def init_arch_params(self, init_type='normal', init_ratio=1e-3):
        for param in self.architecture_parameters():
            if init_type == 'normal':
                param.data.normal_(0, init_ratio)
            elif init_type == 'uniform':
                param.data.uniform_(-init_ratio, init_ratio)
            else:
                raise NotImplementedError

    def reset_binary_gates(self):
        for m in self.redundant_modules:
            try:
                m.binarize()
            except AttributeError:
                print(type(m), ' do not support binarize')
                
    def random_sampling(self):
        for m in self.redundant_modules:
            try:
                m.randomize()
            except AttributeError:
                print(type(m), ' do not support randomize')

    def set_arch_param_grad(self):
        for m in self.redundant_modules:
            try:
                m.set_arch_param_grad()
            except AttributeError:
                print(type(m), ' do not support `set_arch_param_grad()`')

    def rescale_updated_arch_param(self):
        for m in self.redundant_modules:
            try:
                m.rescale_updated_arch_param()
            except AttributeError:
                print(type(m), ' do not support `rescale_updated_arch_param()`')

    """ training related methods """

    def unused_modules_off(self):
        self._unused_modules = []
        for m in self.redundant_modules:
            unused = {}
            if MixedEdge.MODE in ['full', 'two', 'full_v2']:
                involved_index = m.active_index + m.inactive_index
            else:
                involved_index = m.active_index
            for i in range(m.n_choices):
                if i not in involved_index:
                    unused[i] = m.candidate_ops[i]
                    m.candidate_ops[i] = None
            self._unused_modules.append(unused)

    def calculate_log_entropies(self):
        arch_loss = 0
        for m in self.redundant_modules: 
            arch_loss -= m.log_prob
        #arch_loss /=len(self.redundant_modules)
        return arch_loss


    def unused_modules_back(self):
        if self._unused_modules is None:
            return
        for m, unused in zip(self.redundant_modules, self._unused_modules):
            for i in unused:
                m.candidate_ops[i] = unused[i]
        self._unused_modules = None

    def set_chosen_op_active(self):
        for m in self.redundant_modules:
            try:
                m.set_chosen_op_active()
            except AttributeError:
                print(type(m), ' do not support `set_chosen_op_active()`')

    def set_active_via_net(self, net):
        assert isinstance(net, VQAProxylessNASNets)
        for self_m, net_m in zip(self.redundant_modules, net.redundant_modules):
            self_m.active_index = copy.deepcopy(net_m.active_index)
            self_m.inactive_index = copy.deepcopy(net_m.inactive_index)

   
    def convert_to_normal_net(self):
        queue = Queue()
        queue.put(self)
        while not queue.empty():
            module = queue.get()
            for m in module._modules:
                child = module._modules[m]
                if child is None:
                    continue
                if child.__str__().startswith('MixedEdge'):
                    module._modules[m] = child.chosen_op
                else:
                    queue.put(child)
        return VQAProxylessNASNets(self.pretrained_emb, self.token_size, self.WORD_EMBED_SIZE, self.language_encoder, 
            self.lang_adapter, self.img_adapter, list(self.graph_layers), self.att_flat_1, self.att_flat_2, self.classifier)
    
