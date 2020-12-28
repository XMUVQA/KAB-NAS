# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

from utils import *
from collections import OrderedDict
from modules.vqa_modules import MHAtt,FFN,MHSEAtt
from openvqa.ops.fc import FC, MLP
from openvqa.ops.layer_norm import LayerNorm
from openvqa.utils.make_mask import make_mask
import torch.nn as nn
import torch.nn.functional as F
import torch

def set_layer_from_config(layer_config):
    if layer_config is None:
        return None

    name2layer = {
        FullGraphLayer.__name__: FullGraphLayer,
        SEGraphLayer.__name__: SEGraphLayer,
        CoGraphLayer.__name__: CoGraphLayer,
        ZeroLayer.__name__: ZeroLayer,
        AdapterLayer.__name__: AdapterLayer,
        LanguageEncoder.__name__: LanguageEncoder,
        AttFlatLayer.__name__: AttFlatLayer,
        Classifier.__name__: Classifier
    }

    layer_name = layer_config.pop('name')
    layer = name2layer[layer_name]
    return layer.build_from_config(layer_config)



def make_self_mask(batch,y_num,x_num):
    mask = torch.ones(batch,y_num+x_num,y_num+x_num) 
    #x_mask = torch.ones(batch,y_num+x_num,y_num+x_num) 

    mask[:,:y_num,:y_num]=0
    mask[:,y_num:,y_num:]=0
    mask=mask.unsqueeze(1).cuda()
    mask = mask.byte()
    return mask 

def make_co_mask(batch,y_num,x_num):
    mask = torch.ones(batch,y_num+x_num,y_num+x_num) 

    mask[:,:y_num,y_num:] = 0
    mask[:,y_num:,:y_num] = 0
    mask=mask.unsqueeze(1).cuda()
    mask = mask.byte()
    return mask 


class FullGraphLayer(MyModule):

    def __init__(self, in_channels, mid_channels,out_channels, 
        split_num,dropout_rate=0,ops_order='full'):
        super(FullGraphLayer, self).__init__()
        self.in_channels = in_channels # 
        self.out_channels = out_channels
        self.split_num = split_num

        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        self.add_module( 'mhatt', MHAtt(in_channels, out_channels, split_num,dropout_rate))
        self.add_module( 'ffn', FFN(in_channels,mid_channels,out_channels,dropout_rate))
        self.add_module ( 'dropout1', nn.Dropout(dropout_rate))
        self.add_module ('norm1', LayerNorm(out_channels))
        self.add_module ( 'dropout2', nn.Dropout(dropout_rate))
        self.add_module ('norm2', LayerNorm(out_channels))

        """ modules """
       
    @property
    def ops_list(self):
        return self.ops_order.split('_')

    

    def weight_op(self):
        raise NotImplementedError

    """ Methods defined in MyModule """

    def forward(self, y,y_mask):
        y = self._modules['norm1'](y + self._modules['dropout1']( self._modules['mhatt'](y,y,y,y_mask)))

        y = self._modules['norm2'] (y + self._modules['dropout2'] ( self._modules['ffn'](y)))

        return y

    @property
    def module_str(self):
        return 'FullGraphLayer_%dsplits'% self.split_num

    @property
    def config(self):
        return {
            'in_channels': self.in_channels,
            'mid_channels' : self.mid_channels,
            'out_channels': self.out_channels,
            'split_num': self.split_num,

            #'use_bn': self.use_bn,
            #'act_func': self.act_func,
            'dropout_rate': self.dropout_rate,
            'ops_order': self.ops_order,
            **super(FullGraphLayer, self).config
        }

    @staticmethod
    def build_from_config(config):
        raise FullGraphLayer(**conifg)

    def get_flops(self, x):
        raise NotImplementedError

    @staticmethod
    def is_zero_layer():
        return False


class SEGraphLayer(MyModule):

    def __init__(self, in_channels, mid_channels,out_channels, 
        split_num,dropout_rate=0,ops_order='full'):
        super(SEGraphLayer, self).__init__()
        self.in_channels = in_channels # 
        self.out_channels = out_channels
        self.split_num = split_num

        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        self.add_module( 'mhseatt', MHSEAtt(in_channels, out_channels, split_num,dropout_rate))
        self.add_module( 'ffn', FFN(in_channels,mid_channels,out_channels,dropout_rate))
        self.add_module ( 'dropout1', nn.Dropout(dropout_rate))
        self.add_module ('norm1', LayerNorm(out_channels))
        self.add_module ( 'dropout2', nn.Dropout(dropout_rate))
        self.add_module ('norm2', LayerNorm(out_channels))

        """ modules """
       
    @property
    def ops_list(self):
        return self.ops_order.split('_')

    

    def weight_op(self):
        raise NotImplementedError

    """ Methods defined in MyModule """

    def forward(self, y,y_mask):
        batch_num = y.size(0)
        mask_num = y.size(1)
        s_mask = make_self_mask(batch_num,mask_num-100,100)
        y = self._modules['norm1'](y + self._modules['dropout1']( self._modules['mhseatt'](y,y,y,y_mask,s_mask)))

        y = self._modules['norm2'] (y + self._modules['dropout2'] ( self._modules['ffn'](y)))

        return y

    @property
    def module_str(self):
        return 'SEGraphLayer_%dsplits'% self.split_num

    @property
    def config(self):
        return {
            'in_channels': self.in_channels,
            'mid_channels' : self.mid_channels,
            'out_channels': self.out_channels,
            'split_num': self.split_num,

            #'use_bn': self.use_bn,
            #'act_func': self.act_func,
            'dropout_rate': self.dropout_rate,
            'ops_order': self.ops_order,
            **super(SEGraphLayer, self).config
        }

    @staticmethod
    def build_from_config(config):
        raise SEGraphLayer(**conifg)

    def get_flops(self, x):
        raise NotImplementedError

    @staticmethod
    def is_zero_layer():
        return False


class CoGraphLayer(MyModule):

    def __init__(self, in_channels, mid_channels,out_channels, 
        split_num,dropout_rate=0,ops_order='full'):
        super(CoGraphLayer, self).__init__()
        self.in_channels = in_channels # 
        self.out_channels = out_channels
        self.split_num = split_num

        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        self.add_module( 'mhseatt', MHSEAtt(in_channels, out_channels, split_num,dropout_rate))
        self.add_module( 'ffn', FFN(in_channels,mid_channels,out_channels,dropout_rate))
        self.add_module ( 'dropout1', nn.Dropout(dropout_rate))
        self.add_module ('norm1', LayerNorm(out_channels))
        self.add_module ( 'dropout2', nn.Dropout(dropout_rate))
        self.add_module ('norm2', LayerNorm(out_channels))

        """ modules """
       
    @property
    def ops_list(self):
        return self.ops_order.split('_')

    

    def weight_op(self):
        raise NotImplementedError

    """ Methods defined in MyModule """

    def forward(self, y,y_mask):
        batch_num = y.size(0)
        mask_num = y.size(1)
        co_mask = make_co_mask(batch_num,mask_num-100,100)
        y = self._modules['norm1'](y + self._modules['dropout1']( self._modules['mhseatt'](y,y,y,y_mask,co_mask)))

        y = self._modules['norm2'] (y + self._modules['dropout2'] ( self._modules['ffn'](y)))

        return y

    @property
    def module_str(self):
        return 'CoGraphLayer_%dsplits'% self.split_num

    @property
    def config(self):
        return {
            'in_channels': self.in_channels,
            'mid_channels' : self.mid_channels,
            'out_channels': self.out_channels,
            'split_num': self.split_num,

            #'use_bn': self.use_bn,
            #'act_func': self.act_func,
            'dropout_rate': self.dropout_rate,
            'ops_order': self.ops_order,
            **super(CoGraphLayer, self).config
        }

    @staticmethod
    def build_from_config(config):
        raise CoGraphLayer(**conifg)

    def get_flops(self, x):
        raise NotImplementedError

    @staticmethod
    def is_zero_layer():
        return False







class ZeroLayer(MyModule):

    def __init__(self):
        super(ZeroLayer, self).__init__()
        #self.stride = stride

    def forward(self, y,y_mask):
       return y 
    @property
    def module_str(self):
        return 'Zero'

    @property
    def config(self):
        return {
            'name': ZeroLayer.__name__,
            #'stride': self.stride,
        }

    @staticmethod
    def build_from_config(config):
        return ZeroLayer(**config)

    def get_flops(self, y,y_mask):
        return 0, self.forward(y)

    @staticmethod
    def is_zero_layer():
        return True


class AdapterLayer(MyModule):

    def __init__(self, in_size, out_size):
        super(AdapterLayer, self).__init__()
        self.frcn_linear = nn.Linear(in_size, out_size)

    def forward(self, feat):
        mask = make_mask(feat)
        feat = self.frcn_linear(feat)
        return feat,mask 

    @property
    def module_str(self):
        return 'Adapter'

    @property
    def config(self):
        return {
            'name': AdapterLayer.__name__,
            'in_size': self.in_size,
            'out_size': self.out_size,
        }

    @staticmethod
    def build_from_config(config):
        return AdapterLayer(**config)

    #def get_flops(self, x):
    #    return 0, self.forward(x)

    @staticmethod
    def is_zero_layer():
        return False

class ImgAdapterLayer(MyModule):

    def __init__(self, in_size, out_size):
        super(ImgAdapterLayer, self).__init__()
        self.frcn_linear = nn.Linear(in_size+out_size, out_size)
        self.bbox_linear = nn.Linear(5,out_size)

    def forward(self, feat,bbox):
        mask = make_mask(feat)
        b_feat = self.bbox_linear(bbox)
        feat = torch.cat((feat, b_feat), dim=-1)
        feat = self.frcn_linear(feat)
        return feat,mask 

    @property
    def module_str(self):
        return 'Adapter'

    @property
    def config(self):
        return {
            'name': AdapterLayer.__name__,
            'in_size': self.in_size,
            'out_size': self.out_size,
        }

    @staticmethod
    def build_from_config(config):
        return AdapterLayer(**config)

    #def get_flops(self, x):
    #    return 0, self.forward(x)

    @staticmethod
    def is_zero_layer():
        return False

class LanguageEncoder(MyModule):

    def __init__(self, embed_size, hidden_size, layers=1):
        super(LanguageEncoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.layers = layers 
        self.encoder = self.lstm = nn.LSTM(
            input_size=self.embed_size,
            hidden_size=self.hidden_size,
            num_layers=self.layers,
            batch_first=True
        )

    def forward(self, embeddings):
        return self.encoder(embeddings) 
        

    @property
    def module_str(self):
        return 'LanguageEncoder'

    @property
    def config(self):
        return {
            'name': LanguageEncoder.__name__,
            'embed_size': self.embed_size,
            'hidden_size': self.hidden_size,
        }

    @staticmethod
    def build_from_config(config):
        return LanguageEncoder(**config)

    #def get_flops(self, x):
    #    return 0, self.forward(x)

    @staticmethod
    def is_zero_layer():
        return False

class AttFlatLayer(MyModule):

    def __init__(self, in_size, mid_size, out_size, glimpses=1,dropout=0.2):
        super(AttFlatLayer, self).__init__()
        self.in_size = in_size
        self.mid_size = mid_size
        self.out_size = out_size
        self.glimpses = glimpses
        self.dropout =dropout 

        self.mlp = MLP(
            in_size=in_size,
            mid_size=mid_size,
            out_size=glimpses,
            dropout_r=dropout,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            in_size * glimpses,
            out_size
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted
        

    @property
    def module_str(self):
        return 'LanguageEncoder'

    @property
    def config(self):
        """
        self.in_size = in_size
        self.mid_size = mid_size
        self.out_size = out_size
        self.glimpses = glimpses
        self.dropout =dropout 
        """
        return {
            'name': AttFlatLayer.__name__,
            'in_size': self.in_size,
            'mid_size': self.mid_size,
            'out_size': self.out_size,
            'glimpses': self.glimpses,
            'dropout': self.dropout,
        }

    @staticmethod
    def build_from_config(config):
        return AttFlatLayer(**config)

    #def get_flops(self, x):
    #    return 0, self.forward(x)

    @staticmethod
    def is_zero_layer():
        return False

        
class Classifier(MyModule):

    def __init__(self, size,ans_size):
        super(Classifier, self).__init__()
        self.proj_norm = LayerNorm(size)
        self.proj = nn.Linear(size, ans_size)

    def forward(self, x):
        x = self.proj_norm(x)
        pred = self.proj(x)
        return pred

    @property
    def module_str(self):
        return 'Classifier'

    @property
    def config(self):
        return {
            'name': Classifier.__name__,
            'size': self.size,
            'ans_size': self.ans_size,
        }

    @staticmethod
    def build_from_config(config):
        return Classifier(**config)

    #def get_flops(self, x):
    #    return 0, self.forward(x)

    @staticmethod
    def is_zero_layer():
        return False
