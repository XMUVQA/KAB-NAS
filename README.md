# KAB-NAS
This is the source code for "K-armed Bandit based Multi-modal Network Architecture Search for Visual Question Answering" built based on two high-quality projects, i.e., [ProxylessNAS](https://github.com/mit-han-lab/proxylessnas) and [OpenVQA](https://github.com/MILVLG/openvqa). 

## Quick Start

1 Get Data
The features we use for VQA and GQA are detectron_fix_100 from [Pythia](https://github.com/facebookresearch/mmf) and  the offical features from [GQA](https://cs.stanford.edu/people/dorarad/gqa/about.html). 
The annotation files are from [VQA2.0](https://visualqa.org/) and [GQA](https://cs.stanford.edu/people/dorarad/gqa/about.html).
Put these features and files to /openvqa/data, and modify the path config in /openvqa/core/base_cfgs.py. 

2.Network Search 
```bash
python vqa_arch_search.py --RUN='train' --DATASET='vqa' --SPLIT='train' 
```


