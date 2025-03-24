from __future__ import print_function
import argparse
import os
import random
import sys
import torch
from timeit import default_timer as timer
import numpy as np
import json
import pandas as pd
import csv
# Internal Imports
from dataset.path_omic_dataset import classification_dataset
from utils.file_utils import save_pkl
from utils.core_utils_train import train, test
from utils.utils_train import get_custom_exp_code


# export CUDA_VISIBLE_DEVICES=1
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# PyTorch Imports
import torch

### Training settings
parser = argparse.ArgumentParser(
    description='Configurations for classification Analysis on TCGA-BRCA Data.')

### Checkpoint + Misc. Pathing Parameters
parser.add_argument('--data_root_dir',   type=str, default=r'./pt_files',
                    help='Data directory to WSI features (extracted via CLAM')
parser.add_argument('--seed', 			 type=int, default=1,
                    help='Random seed for reproducible experiment (default: 1)') 
parser.add_argument('--n_classes', 			 type=int, default=2,
                    help='classes (default: 2)')
parser.add_argument('--k', 			     type=int, default=5,
                    help='Number of folds (default: 5)')
parser.add_argument('--results_dir',     type=str, default=r'five_fold/5foldcv',
                    help='Results directory (Default: ./results)')
parser.add_argument('--which_splits',    type=str, default=r'5foldcv',
                    help='Which splits folder to use in ./splits/ (Default: ./splits/5foldcv')
parser.add_argument('--split_dir',       type=str, default=r'ER',
                    help='Which cancer type within ./splits/<which_splits> to use for training. Used synonymously for "task" (Default: tcga_blca)')
parser.add_argument('--overwrite',     	 action='store_true', default=False,
                    help='Whether or not to overwrite experiments (if already ran)')
parser.add_argument('--path_load_model', type=str, default=None, help= 'path of ckpt for loading')
parser.add_argument('--train_mode', default=r'auc', type=str, choices=['loss', 'auc'], help='Loss or AUC supervised training')
parser.add_argument('--stage', default=r'test', type=str, choices=['train', 'test'], help='train or test')
parser.add_argument('--topk', default = 8192, type=int, help='topk omic features to use') # 17920 8192 4096 2048 1024 512
parser.add_argument('--pvalues_path_mrna', default=r'cox_mrna.csv', type=str, help='pvalues_path')
parser.add_argument('--path_mrna', default=r'data_mrna.csv', type=str, help='mrna_path')
parser.add_argument('--pvalues_path_cnv', default=r'cox_cnv.csv', type=str, help='pvalues_path')
parser.add_argument('--path_cnv', default=r'data_cnv.csv', type=str, help='cnv_path')
parser.add_argument('--Filter', default=r'top', type=str, choices=['top', 'bottom', 'random'], help='param_code')
parser.add_argument('--alpha', default=1/6, type=float, help='alpha')

### Model Parameters.
parser.add_argument('--model_type', type=str, default=r'online_kd', help='Type of model (Default: online_kd)')
parser.add_argument('--mode', type=str, choices=['mrna', 'cnv', 'path', 'multi'],
                    default=r'multi', help='Specifies which modalities to use / collate function in dataloader.')
parser.add_argument('--fusion', type=str, choices=['concat', 'bilinear'], default=r'bilinear', help='Type of fusion. (Default: concat).')
parser.add_argument('--drop_out', type=float, default=0.25, help='Enable dropout (p=0.25)')
parser.add_argument('--temp', type=int, default=4, help='Temperature coefficient (t=8)')
parser.add_argument('--intermediate_loss_fn', default='SP',type=str,choices=['SP', 'NOT'], help='Intermediate layer knowledge distillation function')
parser.add_argument('--logits_loss_fn', type=str, default='KL', choices=['KL', 'NOT'], help='logits layer knowledge distillation function')
parser.add_argument('--OR', type=str, default='OR', choices=['OR', 'NOR'], help='Use Orthogonal decomposition')
parser.add_argument('--DA', type=str, default='DA', choices=['DA', 'NDA'], help='Use Domain Alignment')


### Optimizer Parameters + classification Loss Function
parser.add_argument('--opt',             type=str,
                    choices=['adam', 'sgd', 'adamw'], default='adamw')
parser.add_argument('--batch_size',      type=int, default=1,
                    help='Batch Size (Default: 1, due to varying bag sizes)')
parser.add_argument('--gc',              type=int,
                    default=32, help='Gradient Accumulation Step.')
parser.add_argument('--max_epochs',      type=int, default=100,
                    help='Maximum number of epochs to train (default: 100)')
parser.add_argument('--lr',				 type=float, default=2e-4,
                    help='Learning rate (default: 0.0002)')
parser.add_argument('--bag_loss',        type=str, choices=['L1Loss', 'CrossEntropyLoss', 'MSELoss']
                    , default=r'CrossEntropyLoss', help='slide-level classification loss function (default: nll_surv)')
parser.add_argument('--reg', 			 type=float, default=1e-5,
                    help='L2-regularization weight decay (default: 1e-5)')
parser.add_argument('--reg_type',        type=str, choices=['None', 'omic', 'pathomic'],
                    default='None', help='Which network submodules to apply L1-Regularization (default: None)')
parser.add_argument('--lambda_reg',      type=float, default=1e-4,
                    help='L1-Regularization Strength (Default 1e-4)')
parser.add_argument('--weighted_sample', action='store_true',
                    default=True, help='Enable weighted sampling')
parser.add_argument('--early_stopping',  action='store_true',
                    default=True, help='Enable early stopping')
parser.add_argument('--start', type=int, default=0, help='start fold')

args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = get_custom_exp_code(args)
args.task = '_'.join(args.split_dir.split('_')[:2]) + '_Classification'

### Sets Seed for reproducible experiments.
def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)


print('\nLoad Dataset')

# Creates results_dir Directory.
if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir, exist_ok=True)

exp_code = str(args.exp_code) + '_s{}'.format(args.seed)

print("===="*30)
print("Experiment Name:", exp_code)
print("===="*30)

args.results_dir = os.path.join(args.results_dir, args.which_splits, args.param_code, exp_code)
if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir)
print("logs saved at ", args.results_dir)

if ('result_val.pkl' in os.listdir(args.results_dir)) and (not args.overwrite) and (args.stage == 'train'):
    print("Exp Code <%s> already exists! Exiting script." % args.exp_code)
    sys.exit()
if ('test.json' in os.listdir(args.results_dir)) and (not args.overwrite) and (args.stage == 'test'):
    print("Exp Code <%s> already exists! Exiting script." % args.exp_code)
    sys.exit()

args.split_dir = os.path.join('./splits', args.which_splits, args.split_dir)
print("split_dir", args.split_dir)
assert os.path.isdir(args.split_dir)

# save configue
args_dict = vars(args)
with open(os.path.join(args.results_dir, 'configue.json'), 'w') as f:
    json.dump(args_dict, f, indent=4)
print("################# Settings ###################")
for key, val in args_dict.items():
    print("{}:  {}".format(key, val))

def main(args):
    # Create Results Directory
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)


    folds = args.k
    start = args.start
    
    # Start 5-Fold CV Evaluation.
    val_summary_all_folds = {}
    test_list = []
    val_list = []
    for i in range(start, folds):
        start_t = timer()
        seed_torch(args.seed)
        args.results_pt_path = os.path.join(
            args.results_dir, 's_{}.pt'.format(i))
        if args.stage == 'train':
            # Gets the Train + Val Dataset Loader.
            if os.path.exists(args.results_pt_path):
                print("Skipping Split %d" % i)
                continue
            train_dataset = classification_dataset(wsi_dir=args.data_root_dir, 
                                        topk = args.topk, 
                                        pvalues_path_cnv= args.pvalues_path_cnv, 
                                        pvalues_path_mrna= args.pvalues_path_mrna,
                                        cnv_path = args.path_cnv,
                                        mrna_path= args.path_mrna,
                                        Filter = args.Filter,
                                        state ='train',
                                        data_splits_csv = os.path.join(args.split_dir, 'splits_{}.csv'.format(i)),
                                        transform=None,
                                        mode=args.mode,
                                        num_classes=args.n_classes)
            val_dataset = classification_dataset(wsi_dir=args.data_root_dir, 
                                        topk = args.topk, 
                                        pvalues_path_cnv= args.pvalues_path_cnv,
                                        pvalues_path_mrna= args.pvalues_path_mrna,
                                        cnv_path = args.path_cnv,
                                        mrna_path = args.path_mrna,
                                        Filter=args.Filter,
                                        state ='val',
                                        data_splits_csv = os.path.join(args.split_dir, 'splits_{}.csv'.format(i)),
                                        transform=None,
                                        mode=args.mode,
                                        num_classes=args.n_classes)
            args.results_pkl_path = os.path.join(args.results_dir, 'result_val.pkl')
            print('training: {}, validation: {},'.format(
                len(train_dataset), len(val_dataset)))
            datasets = (train_dataset, val_dataset)

            if 'multi' in args.mode or 'cnv' in args.mode or 'mrna' in args.mode:
                cnv_sizes = train_dataset.cnv_sizes
                mrna_sizes = train_dataset.mrna_sizes

            if os.path.isfile(args.results_pt_path):
                print("Skipping Split %d" % i)
                if args.train_mode == 'loss':
                    val_summary_results, val_print_results = test(datasets, i, "s_{}.pt".format(i), args)
                elif args.train_mode == 'auc':
                    val_summary_results, val_print_results = test(datasets, i, "s_{}.pt".format(i), args)
                val_list.append(val_summary_results)
                val_summary_all_folds[i] = val_print_results
                continue

        elif args.stage == 'test':
            test_dataset = classification_dataset(wsi_dir=args.data_root_dir, 
                                        topk = args.topk, 
                                        pvalues_path_cnv= args.pvalues_path_cnv,
                                        pvalues_path_mrna= args.pvalues_path_mrna,
                                        cnv_path = args.path_cnv,
                                        mrna_path = args.path_mrna,
                                        state ='test',
                                        Filter=args.Filter,
                                        data_splits_csv = os.path.join(args.split_dir, 'splits_{}.csv'.format(i)),
                                        transform=None,
                                        mode=args.mode,
                                        num_classes=args.n_classes)
            args.results_pkl_path = os.path.join(args.results_dir, 'result_test.pkl')
            print('testing: {},'.format(len(test_dataset)))
            datasets = test_dataset
            if 'multi' in args.mode or 'cnv' in args.mode or 'mrna' in args.mode:
                cnv_sizes = test_dataset.cnv_sizes
                mrna_sizes = test_dataset.mrna_sizes
        else:
            raise NotImplementedError 


        ### Specify the input dimension size if using genomic features.
        if 'multi' in args.mode or 'cnv' in args.mode or 'mrna' in args.mode:
            args.cnv_sizes = cnv_sizes
            args.mrna_sizes = mrna_sizes
            print('mrna Dimensions', args.mrna_sizes)
            print('cnv Dimensions', args.cnv_sizes)
        else:
            args.omic_input_dim = 0
        
        # Run Train-Val on classification Task.
        if args.stage == 'train':
            val_summary_results, val_print_results = train(datasets, i, args)
            val_list.append(val_summary_results)
            val_summary_all_folds[i] = val_print_results
        elif args.stage == 'test':
            all_outputs = test(datasets, i, "s_{}.pt".format(i), args)
            test_list.append(all_outputs)
        else:
            raise NotImplementedError 


    #train
    if args.stage == 'train':
        end_t = timer()
        print('Fold %d Time: %f seconds' % (i, end_t - start_t))
        print('=============================== summary ===============================')

        # val
        result_auc = []
        result_ap = []
        for i, k in enumerate(val_summary_all_folds):
            auc = val_summary_all_folds[k]['result_auc']
            ap = val_summary_all_folds[k]['result_ap']
            print("Fold {}, auc: {:.4f}, ap: {:.4f}".format(k, auc, ap))
            result_auc.append(auc)
            result_ap.append(ap)
        result_auc = np.array(result_auc)
        result_ap = np.array(result_ap)

        print("test Avg AUC of {} folds: {:.4f}, stdp: {:.4f}, stds: {:.4f}".format(
            len(val_summary_all_folds), result_auc.mean(), result_auc.std(), result_auc.std(ddof=1)))
        print('test AVG AP of {} folds: {:.4f}'.format(
            len(val_summary_all_folds), result_ap.mean(), result_ap.std(), result_ap.std(ddof=1)))
    
        df = pd.DataFrame(val_list)
        mean_values = df.mean()
        df.loc['mean'] = mean_values
        df.to_csv(os.path.join(args.results_dir, 'val.csv'), index=False)
        save_pkl(args.results_pkl_path, val_list)
    # test
    elif args.stage == 'test':
        end_t = timer()
        print('Fold %d Time: %f seconds' % (i, end_t - start_t))
        print('=============================== summary ===============================')

        metrics_mean_std = {
            "Path Metrics": {},
            "Omic Metrics": {},
            "Mul Metrics": {}
        }

        # Calculate the mean and variance of the results
        for metric_type in metrics_mean_std.keys():
            all_keys = set()
            for d in test_list:
                all_keys.update(d[metric_type].keys())
            
            for key in all_keys:
                values = [d[metric_type][key] for d in test_list if key in d[metric_type]]
                metrics_mean_std[metric_type][key] = {
                    "mean": np.mean(values),
                    "std": np.std(values)
                }
        print(json.dumps(metrics_mean_std, indent=4))
        with open(os.path.join(args.results_dir, 'test.json'), 'w') as f:
            json.dump(metrics_mean_std, f, indent=4)
                
        metrics_names = ['Path Metrics', 'Omic Metrics', 'Mul Metrics']
        for metric_name in metrics_names:
            filename = f'output_{metric_name.replace(" ", "_").lower()}.csv'
            all_keys = set()
            for d in test_list:
                if metric_name in d and isinstance(d[metric_name], dict):
                    all_keys.update(d[metric_name].keys())
            with open(os.path.join(args.results_dir, filename), 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, fieldnames=all_keys)
                dict_writer.writeheader()
                for d in test_list:
                    if metric_name in d and isinstance(d[metric_name], dict):
                        dict_writer.writerow(d[metric_name])
    else:
        raise NotImplementedError 


if __name__ == "__main__":
    start = timer()
    results = main(args)
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))