# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

#from models.normal_nets.proxyless_nets import ProxylessNASNets
#from run_manager import RunConfig
from vqa_run_manager import VRunConfig

class VQARunConfig(VRunConfig):

    def __init__(self, __C, lr_schedule_type='cosine', lr_schedule_param=None,
                 dataset='vqa',validation_frequency=1, print_frequency=10,n_worker=32):
        super(VQARunConfig, self).__init__(
            __C,lr_schedule_type,lr_schedule_param,dataset,validation_frequency,print_frequency
        )

        self.n_worker = n_worker
        self.__C = __C
        #self.resize_scale = resize_scale
        #self.distort_color = distort_color

        #print(kwargs.keys())

    @property
    def data_config(self):
        return {
            
            'n_worker': self.n_worker,
            
        }
