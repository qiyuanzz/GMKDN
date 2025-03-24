import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
from PIL import Image
import pandas as pd
from sklearn.preprocessing import StandardScaler



class classification_dataset(Dataset):
    def __init__(self, wsi_dir=None, topk = 512, pvalues_path_mrna = r'cox_mrna.csv', mrna_path = r'data_mrna.csv.csv', 
                    pvalues_path_cnv = r'cox_cnv.csv', cnv_path = r'data_cnv.csv', Filter = 'top',
                    state ='train', data_splits_csv = r'splits/5foldcv/ER/splits_0.csv', transform=None, mode=r'omic', num_classes=2):
        self.transform = transform
        self.mode = mode
        self.mrna_path = mrna_path
        self.cnv_path = cnv_path
        self.wsi_dir = wsi_dir
        self.topk = topk
        self.pvalues_path_mrna = pvalues_path_mrna
        self.pvalues_path_cnv = pvalues_path_cnv
        self.data_splits_csv = data_splits_csv
        self.state = state
        self.num_classes = num_classes
        self.Filter = Filter

        
        #data splits
        self.data_splits_csv = pd.read_csv(self.data_splits_csv, index_col=0, dtype=str)
        if state == 'train':
            self.names = self.data_splits_csv.loc[:, 'train'].dropna()
            self.labels = self.data_splits_csv.loc[:, 'train_label'].dropna()
        if state == 'val':
            self.names = self.data_splits_csv.loc[:, 'val'].dropna()
            self.labels = self.data_splits_csv.loc[:, 'val_label'].dropna()
        if state == 'test':
            self.names = self.data_splits_csv.loc[:, 'test'].dropna()
            self.labels = self.data_splits_csv.loc[:, 'test_label'].dropna()


        if self.mode == 'mrna' or self.mode == 'multi' or self.mode == 'cnv'or self.mode == 'omic':
        # mrna
            pvalues_mrna = pd.read_csv(self.pvalues_path_mrna, index_col=0)
            if self.Filter == 'top':
                top_k_covariates = pvalues_mrna.nsmallest(self.topk, 'p')['covariate'].values
            elif self.Filter == 'bottom':
                top_k_covariates = pvalues_mrna.nlargest(self.topk,'p')['covariate'].values
            elif self.Filter == 'random':
                top_k_covariates = pvalues_mrna.sample(self.topk)['covariate'].values
            else:
                raise ValueError('Filter must be either top, bottom or random')
            mrna = pd.read_csv(self.mrna_path, index_col=0)
            mrna.reset_index(inplace=True)
            mrna = mrna.drop_duplicates(subset='case_id', keep='first')
            mrna.set_index('case_id', inplace=True)
            self.mrna = mrna[top_k_covariates]
            self.mrna_sizes = self.mrna.shape[1]


        # cnv
            pvalues_cnv = pd.read_csv(self.pvalues_path_cnv, index_col=0)
            if self.Filter == 'top':
                top_k_covariates = pvalues_cnv.nsmallest(self.topk, 'p')['covariate'].values
            elif self.Filter == 'bottom':
                top_k_covariates = pvalues_cnv.nlargest(self.topk,'p')['covariate'].values
            else:
                raise ValueError('Filter must be either top, bottom or random')
            cnv = pd.read_csv(self.cnv_path, index_col=0)
            cnv.reset_index(inplace=True)
            cnv = cnv.drop_duplicates(subset='case_id', keep='first')
            cnv.set_index('case_id', inplace=True)
            self.cnv = cnv[top_k_covariates]
            self.cnv_sizes = self.cnv.shape[1]

            # standardize the data
            # self.scaler_mrna = StandardScaler()
            # self.scaler_cnv = StandardScaler()
            # self.mrna = pd.DataFrame(self.scaler_mrna.fit_transform(self.mrna), index=self.mrna.index, columns=self.mrna.columns)
            # self.cnv = pd.DataFrame(self.scaler_cnv.fit_transform(self.cnv), index=self.cnv.index, columns=self.cnv.columns)

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(np.array(self.labels) == str(i))[0]

        self.case_to_slide = self._create_case_to_slide_mapping()


    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):


        if self.mode == 'mrna' or self.mode == 'cnv'or self.mode == 'omic':
            sample = self.names[idx]
            label = self.labels[idx]
            mrna_feat = torch.tensor(self.mrna.loc[sample].values, dtype=torch.float32)
            cnv_feat = torch.tensor(self.cnv.loc[sample].values, dtype=torch.float32)
            feat = [mrna_feat, cnv_feat]
        elif self.mode == 'path':
            sample = self.names[idx]
            label = self.labels[idx]
            slide_id = self.get_slide_ids(sample)[0]        # Assuming there is only one slide per case
            wsi_path = os.path.join(self.wsi_dir, slide_id)
            wsi_feat =  torch.load(wsi_path)
            wsi_feat = wsi_feat.clone().detach().float()
            feat = [wsi_feat]
        elif self.mode == 'multi':
            sample = self.names[idx]
            # print(idx , sample)
            label = self.labels[idx]
            slide_id = self.get_slide_ids(sample)[0]        # Assuming there is only one slide per case
            wsi_path = os.path.join(self.wsi_dir, slide_id)
            wsi_feat =  torch.load(wsi_path)
            mrna_feat = torch.tensor(self.mrna.loc[sample].values, dtype=torch.float32)
            cnv_feat = torch.tensor(self.cnv.loc[sample].values, dtype=torch.float32)
            feat = [wsi_feat, mrna_feat, cnv_feat]
        else:
            raise ValueError('mode must be either omic, wsi or multi')
        label = int(label)
        return feat, label, idx
    
    def getlabel(self, idx):
        return self.labels[idx]

    def _create_case_to_slide_mapping(self):
        case_to_slide = {}
        pt_files = [f for f in os.listdir(self.wsi_dir) if f.endswith('.pt')]
        for pt_file in pt_files:
            case_id = pt_file[:12]  # Assuming the first 12 characters are the case_id
            slide_id = pt_file  # The entire filename is the slide_id
            if case_id in case_to_slide:
                case_to_slide[case_id].append(slide_id)
            else:
                case_to_slide[case_id] = [slide_id]
        return case_to_slide

    def get_slide_ids(self, case_id):
        return self.case_to_slide.get(case_id, [])

if __name__ == '__main__':
    dataset = classification_dataset()
    for feat, label in dataset:
        print(feat, label)
        break




