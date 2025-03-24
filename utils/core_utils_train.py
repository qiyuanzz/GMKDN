from argparse import Namespace
import os
import torchmetrics
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from utils.utils_train import get_optim, get_split_loader, get_split_loader_and_index
import torch.nn as nn
from utils.loss import DistillKL, Similarity, CRDLoss

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False, metric='auc'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.auc_max = np.Inf
        self.metric = metric

    def __call__(self, epoch, auc, model, ckpt_name = 'checkpoint.pt'):

        if self.metric == 'auc':
            score = auc

            if epoch < self.warmup:
                pass
            elif self.best_score is None:
                self.best_score = score
                self.save_checkpoint(auc, model, ckpt_name)
            elif score <= self.best_score:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience and epoch > self.stop_epoch:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(auc, model, ckpt_name)
                self.counter = 0
        elif self.metric == 'loss':
            score = -auc

            if epoch < self.warmup:
                pass
            elif self.best_score is None:
                self.best_score = score
                self.save_checkpoint(auc, model, ckpt_name)
            elif score <= self.best_score:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience and epoch > self.stop_epoch:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(auc, model, ckpt_name)
                self.counter = 0

    def save_checkpoint(self, auc, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            if self.metric == 'auc':
                print(f'auc increased ({self.auc_max:.6f} --> {auc:.6f}).  Saving model ...')
            elif self.metric == 'loss':
                print(f'val_loss decreased ({self.auc_max:.6f} --> {auc:.6f}).  Saving model ...')

        torch.save(model.state_dict(), ckpt_name)
        self.auc_max = auc



def test(datasets: tuple, cur: int, model_weights, args: Namespace):
    """   
        train for a single fold
    """
    # print('\nTraining Fold {}!'.format(cur))
    args.writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(args.writer_dir):
        os.mkdir(args.writer_dir)



    test_split = datasets

    if args.bag_loss == 'CrossEntropyLoss':
        loss_fn = nn.CrossEntropyLoss()
    elif args.bag_loss == 'L1Loss':
        loss_fn = nn.L1Loss()
    elif args.bag_loss == 'MSELoss':
        loss_fn = nn.MSELoss()
    else:
        raise NotImplementedError

    

    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, 'topk': args.topk}
    args.fusion = None if args.fusion == 'None' else args.fusion
    if args.model_type == 'online_kd':
        from models.online_kd import  ABMIL_SNN, ABMIL, omic_transformer
        model_dict = {'n_classes': args.n_classes , 'topk' : args.topk, 'dropout': args.drop_out}
        tea_model_abmil_snn = ABMIL_SNN(**model_dict)
        tea_model_snn = omic_transformer(**model_dict)
        model_dict = {'n_classes': args.n_classes , 'dropout': args.drop_out}
        model = ABMIL(**model_dict)
        
    else:
        raise NotImplementedError
        
    if not args.path_load_model:
        model.load_state_dict(torch.load(os.path.join(args.results_dir , model_weights)))
        tea_model_abmil_snn.load_state_dict(torch.load(os.path.join(args.results_dir , 'tea_mulmodel', model_weights)))
        tea_model_snn.load_state_dict(torch.load(os.path.join(args.results_dir , 'tea_unimodel', model_weights)))
    




    test_loader = get_split_loader(test_split,  testing = False, mode=args.mode, batch_size=args.batch_size)
    test_AUROC = torchmetrics.AUROC(task='binary', num_classes=args.n_classes, average = 'macro')
    metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(task='binary', num_classes = args.n_classes, average = 'micro'),
                                                torchmetrics.Precision(task='binary', num_classes = args.n_classes),
                                                torchmetrics.Recall(task='binary', num_classes = args.n_classes,
                                                                average = 'macro'),
                                                torchmetrics.Specificity(task='binary', average = 'macro',
                                                                    num_classes = args.n_classes),
                                                torchmetrics.F1Score(task='binary', average = 'macro',
                                                                    num_classes = args.n_classes),
                                                torchmetrics.CohenKappa(task='binary', num_classes = args.n_classes)])
    test_AP = torchmetrics.AveragePrecision(task='binary', num_classes=args.n_classes, average = 'macro')

    omic_AUROC = torchmetrics.AUROC(task='binary', num_classes=args.n_classes, average = 'macro')
    omic_metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(task='binary', num_classes = args.n_classes, average = 'micro'),
                                                torchmetrics.Precision(task='binary', num_classes = args.n_classes),
                                                torchmetrics.Recall(task='binary', num_classes = args.n_classes,
                                                                average = 'macro'),
                                                torchmetrics.Specificity(task='binary', average = 'macro',
                                                                    num_classes = args.n_classes),
                                                torchmetrics.F1Score(task='binary', average = 'macro',
                                                                    num_classes = args.n_classes),
                                                torchmetrics.CohenKappa(task='binary', num_classes = args.n_classes)])
    omic_AP = torchmetrics.AveragePrecision(task='binary', num_classes=args.n_classes, average = 'macro')

    mul_AUROC = torchmetrics.AUROC(task='binary', num_classes=args.n_classes, average = 'macro')
    mul_metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(task='binary', num_classes = args.n_classes, average = 'micro'),
                                                torchmetrics.Precision(task='binary', num_classes = args.n_classes),
                                                torchmetrics.Recall(task='binary', num_classes = args.n_classes,
                                                                average = 'macro'),
                                                torchmetrics.Specificity(task='binary', average = 'macro',
                                                                    num_classes = args.n_classes),
                                                torchmetrics.F1Score(task='binary', average = 'macro',
                                                                    num_classes = args.n_classes),
                                                torchmetrics.CohenKappa(task='binary', num_classes = args.n_classes)])
    mul_AP = torchmetrics.AveragePrecision(task='binary', num_classes=args.n_classes, average = 'macro')
    all_auc = [test_AUROC, omic_AUROC, mul_AUROC]
    all_metrics = [metrics, omic_metrics, mul_metrics]
    all_AP = [test_AP, omic_AP, mul_AP]

    if args.mode == 'multi':
        if args.model_type == 'online_kd':
            from trainer.online_kd_trainer import test_classification_coattn
            test_loss, all_outputs = test_classification_coattn(model, tea_model_snn, tea_model_abmil_snn, test_loader, all_auc, all_metrics, all_AP, loss_fn, args)
        else:
            raise NotImplementedError
        
    else:
        raise NotImplementedError

    return all_outputs




def train(datasets: tuple, cur: int, args: Namespace):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    args.writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(args.writer_dir):
        os.mkdir(args.writer_dir)

    print('\nInit train/val splits...', end=' ')
    train_split, val_split = datasets
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print('\nInit loss function...', end=' ')

    if args.bag_loss == 'CrossEntropyLoss':
        loss_fn = nn.CrossEntropyLoss()
    elif args.bag_loss == 'L1Loss':
        loss_fn = nn.L1Loss()
    elif args.bag_loss == 'MSELoss':
        loss_fn = nn.MSELoss()
    else:
        raise NotImplementedError

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = False, 
        weighted = args.weighted_sample, mode=args.mode, batch_size=args.batch_size)
    val_loader = get_split_loader(val_split,  testing = False, mode=args.mode, batch_size=args.batch_size)
    print('Done!')

    print('Done!')
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    args.fusion = None if args.fusion == 'None' else args.fusion
    module_list = nn.ModuleList([])
    if args.model_type == 'online_kd':
        from models.online_kd import  ABMIL_SNN, ABMIL, omic_transformer

        model_dict = {'n_classes': args.n_classes , 'topk' : args.topk, 'dropout': args.drop_out}
        tea_model_abmil_snn = ABMIL_SNN(**model_dict)
        tea_model_snn = omic_transformer(**model_dict)

        model_dict = {'n_classes': args.n_classes , 'dropout': args.drop_out}
        model = ABMIL(**model_dict)


        
        module_list.append(tea_model_abmil_snn)
        module_list.append(tea_model_snn)
        if args.intermediate_loss_fn == 'SP':
            inter_fn = Similarity()
        else:
            inter_fn = False

        
        if args.logits_loss_fn == 'KL':
            logtis_fn = DistillKL(T=args.temp)
        else:
            logtis_fn = False
    else:
        raise NotImplementedError
    
    
    print('\nInit optimizer ...', end=' ')
    module_list.append(model)
    optimizer = get_optim(module_list, args)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        if args.train_mode == 'auc':
            early_stopping = EarlyStopping(warmup=0, patience=10, stop_epoch=5, verbose = True, metric='auc')
        elif args.train_mode == 'loss':
            early_stopping = EarlyStopping(warmup=0, patience=10, stop_epoch=5, verbose = True, metric='loss')
        else:
            raise NotImplementedError
    else:
        early_stopping = None
    
    max_auc = 0.
    best_val_dict = {}
    AUROC = torchmetrics.AUROC(task='binary', num_classes=args.n_classes, average = 'macro')
    val_AUROC = torchmetrics.AUROC(task='binary', num_classes=args.n_classes, average = 'macro')
    metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(task='binary', num_classes = args.n_classes, average = 'micro'),
                                                torchmetrics.Precision(task='binary', num_classes = args.n_classes),
                                                torchmetrics.Recall(task='binary', num_classes = args.n_classes,
                                                                average = 'macro'),
                                                torchmetrics.Specificity(task='binary', average = 'macro',
                                                                    num_classes = args.n_classes),
                                                torchmetrics.F1Score(task='binary', average = 'macro',
                                                                    num_classes = args.n_classes),
                                                torchmetrics.CohenKappa(task='binary', num_classes = args.n_classes)])
    train_AP = torchmetrics.AveragePrecision(task='binary', num_classes=args.n_classes, average = 'macro')
    val_AP = torchmetrics.AveragePrecision(task='binary', num_classes=args.n_classes, average = 'macro')
    val_metrics = metrics.clone()
    print("running with {} {}".format(args.model_type, args.mode))

    for epoch in range(args.max_epochs):
        if args.mode == 'multi':
            if args.model_type == 'online_kd':
                from trainer.online_kd_trainer import train_loop_classification_coattn, validate_classification_coattn
                train_loop_classification_coattn(epoch, model, tea_model_abmil_snn, tea_model_snn, train_loader, optimizer, scheduler,
                                                inter_fn, logtis_fn, AUROC, train_AP, metrics, loss_fn, args)
                val_loss, val_auc, val_ap, all_val_metrics, tea_mulmodel, tea_unimodel, val_stop  = validate_classification_coattn(cur, epoch, model, tea_model_abmil_snn, tea_model_snn, val_loader,
                                                                                            val_AUROC, val_AP, val_metrics, early_stopping, loss_fn, args)
                os.makedirs(os.path.join(args.results_dir, 'tea_mulmodel'), exist_ok=True)
                torch.save(tea_mulmodel.state_dict(), os.path.join(args.results_dir, 'tea_mulmodel', 's_{}.pt'.format(cur)))
                os.makedirs(os.path.join(args.results_dir, 'tea_unimodel'), exist_ok=True)
                torch.save(tea_unimodel.state_dict(), os.path.join(args.results_dir, 'tea_unimodel', 's_{}.pt'.format(cur)))
                if val_stop:
                    break
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError


        if val_auc > max_auc:
            max_auc = val_auc
            best_val_dict = {
                            'val_loss':val_loss, 
                            'val_auc':val_auc.item(), 
                            'val_ap':val_ap.item(),
                            'val_acc': all_val_metrics['BinaryAccuracy'].item(), 
                            'BinaryPrecision':all_val_metrics['BinaryPrecision'].item(),
                            'BinaryRecall':all_val_metrics['BinaryRecall'].item(),
                            'BinarySpecificity':all_val_metrics['BinarySpecificity'].item(),
                            'BinaryF1Score':all_val_metrics['BinaryF1Score'].item(),
                            'BinaryCohenKappa' :all_val_metrics['BinaryCohenKappa'].item()
                            }
            
        val_AUROC.reset()
        AUROC.reset()
        val_metrics.reset()
        metrics.reset()
        val_AP.reset()
        train_AP.reset()
        

    val_print_results = {'result_auc': val_auc.cpu(), 'result_ap': val_ap.cpu()}
    print("================= summary of fold {} ====================".format(cur))
    print("test auc: {:.4f}, test ap: {:.4f}".format(val_auc, val_ap))

    return best_val_dict, val_print_results