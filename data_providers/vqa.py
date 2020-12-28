# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import torch.utils.data
from openvqa.datasets.dataset_loader import DatasetLoader
import torch.utils.data as Data

from data_providers.base_provider import *
import copy


class VQADataProvider(DataProvider):

    def __init__(self, __C):

        #self._save_path = save_path
        #train_transforms = self.build_train_transform(distort_color, resize_scale)
        #train_dataset = datasets.ImageFolder(self.train_path, train_transforms)

        print('Loading dataset........')

        self.train_set = DatasetLoader(__C).DataSet()
        __C_eval = copy.deepcopy(__C)
        setattr(__C_eval, 'RUN_MODE', 'val')
        self.val_set = DatasetLoader(__C_eval).DataSet()
        self.test_set =self.val_set
        """
        self.train = Data.DataLoader(
                train_set,
                batch_size=__C.EVAL_BATCH_SIZE,
                shuffle=False,
                num_workers=__C.NUM_WORKERS,
                pin_memory=__C.PIN_MEM
            )
        self.valid = Data.DataLoader(
                vali_set,
                batch_size=__C.EVAL_BATCH_SIZE,
                shuffle=True,
                num_workers=__C.NUM_WORKERS,
                pin_memory=__C.PIN_MEM
            )
        """

        #self.test = self.valid 

        self.ans_size = self.train_set.ans_size
        self.data_size = self.train_set.data_size
        self.val_data_size = self.val_set.data_size
        self.token_size = self.train_set.token_size
        self.pretrained_emb = self.train_set.pretrained_emb 



    @staticmethod
    def name():
        return 'vqa'
    """
    @property
    def data_shape(self):
        return 3, self.image_size, self.image_size  # C, H, W
    """

   
