# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import argparse,yaml

from models import VQARunConfig
from openvqa.models.model_loader import CfgLoader

from vqa_nas_manager import *
from models.super_nets.vqa_super_proxyless import *




parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='VQA_NAS/')
parser.add_argument('--gpu', help='gpu available', default='2')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--debug', help='freeze the weight parameters', action='store_true')
parser.add_argument('--manual_seed', default=0, type=int)

parser.add_argument('--structure', type=str, default='agan',choices=['agan'])
""" run config """

parser.add_argument('--dataset', type=str, default='vqa', choices=['gqa','vqa'])
parser.add_argument('--MODEL', type=str, default='agan', choices=['agan'])



#parser.add_argument('--opt_type', type=str, default='sgd', choices=['sgd'])
parser.add_argument('--momentum', type=float, default=0.9)  # opt_param
parser.add_argument('--no_nesterov', action='store_true')  # opt_param
parser.add_argument('--weight_decay', type=float, default=4e-5)
parser.add_argument('--label_smoothing', type=float, default=0.1)

parser.add_argument('--validation_frequency', type=int, default=1)
parser.add_argument('--print_frequency', type=int, default=10)

parser.add_argument('--n_worker', type=int, default=32)

parser.add_argument('--dropout', type=float, default=0.1)

# architecture search config
""" arch search algo and warmup """
#parser.add_argument('--arch_algo', type=str, default='grad', choices=['grad', 'rl'])
parser.add_argument('--warmup_epochs', type=int, default=10)
""" shared hyper-parameters """
parser.add_argument('--arch_init_type', type=str, default='normal', choices=['normal', 'uniform'])
parser.add_argument('--arch_init_ratio', type=float, default=1e-3)
parser.add_argument('--arch_opt_type', type=str, default='adam', choices=['adam'])
parser.add_argument('--arch_lr', type=float, default=1e-2)
parser.add_argument('--arch_adam_beta1', type=float, default=0)  # arch_opt_param
parser.add_argument('--arch_adam_beta2', type=float, default=0.999)  # arch_opt_param
parser.add_argument('--arch_adam_eps', type=float, default=1e-8)  # arch_opt_param
parser.add_argument('--arch_weight_decay', type=float, default=0)
#parser.add_argument('--target_hardware', type=str, default=None, choices=['mobile', 'cpu', 'gpu8', 'flops', None])
""" Grad hyper-parameters """
parser.add_argument('--grad_update_arch_param_every', type=int, default=5)
parser.add_argument('--grad_update_steps', type=int, default=1)

""" RL hyper-parameters """
#parser.add_argument('--rl_update_per_epoch', action='store_true')
#parser.add_argument('--rl_update_steps_per_epoch', type=int, default=300)
#parser.add_argument('--rl_baseline_decay_weight', type=float, default=0.99)
#

# VQA args
parser.add_argument('--RUN', dest='RUN_MODE',
                      choices=['train', 'val', 'test'],
                      help='{train, val, test}',
                      type=str, required=True)



parser.add_argument('--DATASET', dest='DATASET',
                  choices=['vqa', 'gqa', 'clevr'],
                  help='{'
                       'vqa,'
                       'gqa,'
                       'clevr,'
                       '}'
                    ,
                  type=str, required=True)


parser.add_argument('--SPLIT', dest='TRAIN_SPLIT',
                  choices=['train', 'train+val', 'train+val+vg'],
                  help="set training split, "
                       "vqa: {'train', 'train+val', 'train+val+vg'}"
                       "gqa: {'train', 'train+val'}"
                       "clevr: {'train', 'train+val'}"
                    ,
                  type=str)

parser.add_argument('--EVAL_EE', dest='EVAL_EVERY_EPOCH',
                  choices=['True', 'False'],
                  help='True: evaluate the val split when an epoch finished,'
                       'False: do not evaluate on local',
                  type=str)

parser.add_argument('--SAVE_PRED', dest='TEST_SAVE_PRED',
                  choices=['True', 'False'],
                  help='True: save the prediction vectors,'
                       'False: do not save the prediction vectors',
                  type=str)

parser.add_argument('--BS', dest='BATCH_SIZE',
                  help='batch size in training',
                  type=int)

parser.add_argument('--GPU', dest='GPU',
                  help="gpu choose, eg.'0, 1, 2, ...'",
                  type=str)



parser.add_argument('--CKPT_V', dest='CKPT_VERSION',
                  help='checkpoint version',
                  type=str)

parser.add_argument('--CKPT_E', dest='CKPT_EPOCH',
                  help='checkpoint epoch',
                  type=int)

parser.add_argument('--CKPT_PATH', dest='CKPT_PATH',
                  help='load checkpoint path, we '
                       'recommend that you use '
                       'CKPT_VERSION and CKPT_EPOCH '
                       'instead, it will override'
                       'CKPT_VERSION and CKPT_EPOCH',
                  type=str)

parser.add_argument('--ACCU', dest='GRAD_ACCU_STEPS',
                  help='split batch to reduce gpu memory usage',
                  type=int)

parser.add_argument('--NW', dest='NUM_WORKERS',
                  help='multithreaded loading to accelerate IO',
                  type=int)

parser.add_argument('--PINM', dest='PIN_MEM',
                  choices=['True', 'False'],
                  help='True: use pin memory, False: not use pin memory',
                  type=str)

parser.add_argument('--VERB', dest='VERBOSE',
                  choices=['True', 'False'],
                  help='True: verbose print, False: simple print',
                  type=str)


if __name__ == '__main__':

    args = parser.parse_args()
    cfg_file = "configs/{}/{}.yml".format(args.DATASET, args.MODEL)
    with open(cfg_file, 'r') as f:
        yaml_dict = yaml.load(f)

    __C = CfgLoader(yaml_dict['MODEL_USE']).load()
    args = __C.str_to_bool(args)
    args_dict = __C.parse_to_dict(args)

    args_dict = {**yaml_dict, **args_dict}
    __C.add_args(args_dict)
    __C.proc()

    #args.manual_seed = random.randint(0, 5000)

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    os.makedirs(args.path, exist_ok=True)

    # build run config from args
    args.lr_schedule_param = None
    args.opt_param = {
        'momentum': args.momentum,
        'nesterov': not args.no_nesterov,
    }
    run_config = VQARunConfig(
        __C 
    )

    # debug, adjust run_config
    if args.debug:
        run_config.n_worker = 0

    """
    enc_candidates=[

    'FullGraphLayer_2','FullGraphLayer_4',
        'FullGraphLayer_8'
    ]
	"""

    graph_candidates=[
        
        'FullGraphLayer_2','FullGraphLayer_4',
        'FullGraphLayer_8',
        'CoGraphLayer_2','CoGraphLayer_4',
        'CoGraphLayer_8',
        'SEGraphLayer_2',
        'SEGraphLayer_4','SEGraphLayer_8'
        #'ZeroLayer'
    ]
    """
    graph_candidates=[
        
        'FullGraphLayer_8','SEGraphLayer_8',
        
        'CoGraphLayer_8'
    ]"""
    """
    if args.structure == 'agan+enc':
        super_net = SuperVQAENCProxylessNASNets(
        __C, run_config.dataset.pretrained_emb,run_config.dataset.token_size,run_config.dataset.ans_size,enc_candidates,graph_candidates
        )
    else:
    """
    super_net = SuperVQAProxylessNASNets(
    __C, run_config.dataset.pretrained_emb,run_config.dataset.token_size,run_config.dataset.ans_size,graph_candidates
	)

    # build arch search config from args
    if args.arch_opt_type == 'adam':
        args.arch_opt_param = {
            'betas': (args.arch_adam_beta1, args.arch_adam_beta2),
            'eps': args.arch_adam_eps,
        }
    else:
        args.arch_opt_param = None
    
    
    arch_search_config = KabArchSearchConfig(args.arch_init_type, args.arch_init_ratio,
        args.arch_opt_type,args.arch_lr,args.arch_opt_param,args.arch_weight_decay)
    

    print('Run config:')
    for k, v in run_config.config.items():
        print('\t%s: %s' % (k, v))
    print('Architecture Search config:')
    for k, v in arch_search_config.config.items():
        print('\t%s: %s' % (k, v))

    # arch search run manager
    arch_search_run_manager = ArchSearchRunManager(__C,args.path, super_net, run_config, arch_search_config)
    #if args.structure == 'agan+enc':
    #	arch_search_run_manager.with_enc = True
    # resume
    if args.resume:
        try:
            arch_search_run_manager.load_model()
        except Exception:
            from pathlib import Path
            home = str(Path.home())
            warmup_path = os.path.join(
                home, 'VQA_NAS/warmup.pth.tar' 
            )
            if os.path.exists(warmup_path):
                print('load warmup weights')
                arch_search_run_manager.load_model(model_fname=warmup_path)
            else:
                print('fail to load models')

    # warmup
    if arch_search_run_manager.warmup:
        arch_search_run_manager.warm_up(warmup_epochs=args.warmup_epochs)

    # joint training
    arch_search_run_manager.train(fix_net_weights=args.debug)
