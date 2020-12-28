# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

from modules.layers import *
from modules.vqa_layers import *
from openvqa.utils.make_mask import make_mask
import torch.nn as nn
import torch.nn.functional as F
import torch
import json


class VQAProxylessNASNets(MyNetwork):

    def __init__(self, pretrained_emb, token_size, embed_size, language_encoder, lang_adapter, img_adapter, graph_layers, att_flat_1, att_flat_2, classifier):
        super(VQAProxylessNASNets, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=embed_size
        )
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
        self.language_encoder = language_encoder
        self.lang_adapter = lang_adapter
        self.img_adapter = img_adapter
        self.graph_layers = nn.ModuleList(graph_layers)
        self.num_layer = len(graph_layers)
        self.att_flat_1 = att_flat_1
        self.att_flat_2 = att_flat_2
        self.classifier = classifier 


    def forward(self, frcn_feat, bbox_feat, ques_ix):

        #lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        lang_feat = self.embedding(ques_ix) 
        lang_feat, _ = self.language_encoder(lang_feat)
        lang_feat_p, lang_feat_mask = self.lang_adapter(lang_feat)

        img_feat, img_feat_mask = self.img_adapter(frcn_feat,bbox_feat)

        joint_mask = torch.cat([lang_feat_mask,img_feat_mask],-1) 
        joint_feat = torch.cat([lang_feat,img_feat],1)

        for i in range(self.num_layer):
            joint_feat = self.graph_layers[i] (joint_feat,joint_mask)

        graph_feat_1 = self.att_flat_1(joint_feat,joint_mask)
        graph_feat_2 = self.att_flat_2(joint_feat,joint_mask)

        graph_feat = graph_feat_1 + graph_feat_2

        proj_feat = self.classifier(graph_feat)

        return proj_feat

    @property
    def module_str(self):
        _str = ''
        for graph_layer in self.graph_layers:
            _str += graph_layer.unit_str + '\n'
        return _str

    @property
    def config(self):
        return {
            'name': ProxylessNASNets.__name__,
            #'bn': self.get_bn_param(),
            #'embedding': self.embedding.config,
            'language_encoder': self.language_encoder.config,
            'lang_adapter': self.lang_adapter.config,
            'img_adapter': self.img_adapter.config,
            'graph_layers':[
                graph_layer.config for graph_layer in self.graph_layers
            ],
            'att_flat_1': self.att_flat_1.config,
            'att_flat_2': self.att_flat_2.config,
            'classifier': self.classifier.config,
        }

    @staticmethod
    def build_from_config(config,pretrained_emb,token_size,embed_size):
        #embedding = set_layer_from_config(config['embedding'])
        language_encoder = set_layer_from_config(config['language_encoder'])
        lang_adapter = set_layer_from_config(config['lang_adapter'])
        img_adapter = set_layer_from_config(config['img_adapter'])
        att_flat_1 = set_layer_from_config(config['att_flat_1'])
        att_flat_2 = set_layer_from_config(config['att_flat_2'])
        classifier = set_layer_from_config(config['classifier'])
        graph_layers = []
        for graph_config in config['graph_layers']:
            graph_layers.append(set_layer_from_config(graph_config))

        net = VQAProxylessNASNets(pretrained_emb,token_size,embed_size, language_encoder, lang_adapter, img_adapter, graph_layers, att_flat_1, att_flat_2, classifier)

        return net


class VQAENCProxylessNASNets(MyNetwork):

    def __init__(self, pretrained_emb, token_size, embed_size, language_encoder, lang_adapter, img_adapter, lang_layers,graph_layers, att_flat_1, att_flat_2, classifier):
        super(VQAENCProxylessNASNets, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=embed_size
        )
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
        self.language_encoder = language_encoder
        self.lang_adapter = lang_adapter
        self.img_adapter = img_adapter
        self.graph_layers = nn.ModuleList(graph_layers)
        self.lang_layers = nn.ModuleList(lang_layers)
        self.num_layer = len(graph_layers)
        self.att_flat_1 = att_flat_1
        self.att_flat_2 = att_flat_2
        self.classifier = classifier 
        self.lang_num_layer = len(lang_layers)


    def forward(self, frcn_feat, bbox_feat, ques_ix):

        #lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        lang_feat = self.embedding(ques_ix) 
        lang_feat, _ = self.language_encoder(lang_feat)
        _, lang_feat_mask = self.lang_adapter(lang_feat)

        img_feat, img_feat_mask = self.img_adapter(frcn_feat,bbox_feat)

        for i in range(self.lang_num_layer):
            lang_feat = self.lang_layers[i](lang_feat,lang_feat_mask)

        joint_mask = torch.cat([lang_feat_mask,img_feat_mask],-1) 
        joint_feat = torch.cat([lang_feat,img_feat],1)

        for i in range(self.num_layer):
            joint_feat = self.graph_layers[i] (joint_feat,joint_mask)

        graph_feat_1 = self.att_flat_1(lang_feat,lang_feat_mask)
        graph_feat_2 = self.att_flat_2(joint_feat,joint_mask)

        graph_feat = graph_feat_1 + graph_feat_2

        proj_feat = self.classifier(graph_feat)

        return proj_feat

    @property
    def module_str(self):
        _str = ''
        for graph_layer in self.graph_layers:
            _str += graph_layer.unit_str + '\n'
        return _str

    @property
    def config(self):
        return {
            'name': ProxylessNASNets.__name__,
            #'bn': self.get_bn_param(),
            #'embedding': self.embedding.config,
            'language_encoder': self.language_encoder.config,
            'lang_adapter': self.lang_adapter.config,
            'img_adapter': self.img_adapter.config,
            'graph_layers':[
                graph_layer.config for graph_layer in self.graph_layers
            ],
            'att_flat_1': self.att_flat_1.config,
            'att_flat_2': self.att_flat_2.config,
            'classifier': self.classifier.config,
        }

    @staticmethod
    def build_from_config(config,pretrained_emb,token_size,embed_size):
        #embedding = set_layer_from_config(config['embedding'])
        language_encoder = set_layer_from_config(config['language_encoder'])
        lang_adapter = set_layer_from_config(config['lang_adapter'])
        img_adapter = set_layer_from_config(config['img_adapter'])
        att_flat_1 = set_layer_from_config(config['att_flat_1'])
        att_flat_2 = set_layer_from_config(config['att_flat_2'])
        classifier = set_layer_from_config(config['classifier'])
        graph_layers = []
        for graph_config in config['graph_layers']:
            graph_layers.append(set_layer_from_config(graph_config))

        net = VQAENCProxylessNASNets(pretrained_emb,token_size,embed_size, language_encoder, lang_adapter, img_adapter, graph_layers, att_flat_1, att_flat_2, classifier)

        return net