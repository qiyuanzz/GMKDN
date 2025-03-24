from __future__ import print_function, division
import math
import os
import pdb
import pickle
import re

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset

from utils.utils import generate_split, nth


class Generic_WSI_classification_Dataset(Dataset):
    def __init__(self, csv_path = 'dataset_csv/ccrcc_clean.csv', label_path='',  mode = 'omic', apply_sig = False, 
        shuffle = False, seed = 7, print_info = True, n_bins = 4, ignore=[],
        patient_strat=False, label_col = None, filter_dict = {}, eps=1e-6, pkl_path='all_HER2.pkl'):
        r"""
        Generic_WSI_classification_Dataset 

        Args:
            csv_file (string): Path to the csv file with annotations.
            shuffle (boolean): Whether to shuffle
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int
            ignore (list): List containing class labels to ignore
        """
        self.custom_test_ids = None
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
        self.data_dir = None
        self.label_path = label_path

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)


        slide_data = pd.read_csv(csv_path, low_memory=False)
        ### new
        missing_slides_ls = ['TCGA-A7-A6VX-01Z-00-DX2.9EE94B59-6A2C-4507-AA4F-DC6402F2B74F.svs',
                            'TCGA-A7-A0CD-01Z-00-DX2.609CED8D-5947-4753-A75B-73A8343B47EC.svs',
                            'TCGA-HT-7483-01Z-00-DX1.7241DF0C-1881-4366-8DD9-11BF8BDD6FBF.svs',
                            'TCGA-06-0882-01Z-00-DX2.7ad706e3-002e-4e29-88a9-18953ba422bf.svs']
        slide_data.drop(slide_data[slide_data['slide_id'].isin(missing_slides_ls)].index, inplace=True)


        if 'case_id' not in slide_data:
            slide_data.index = slide_data.index.str[:12]
            slide_data['case_id'] = slide_data.index
            slide_data = slide_data.reset_index(drop=True)
        

        patients_df = slide_data.drop_duplicates(['case_id']).copy()

        patient_dict = {}
        slide_data = slide_data.set_index('case_id')
        for patient in patients_df['case_id']:
            slide_ids = slide_data.loc[patient, 'slide_id']
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = slide_ids.values
            patient_dict.update({patient:slide_ids})

        self.patient_dict = patient_dict
    
        slide_data = patients_df
        slide_data.reset_index(drop=True, inplace=True)
        slide_data = slide_data.assign(slide_id=slide_data['case_id'])

        self.num_classes = 2
        patients_df = slide_data.drop_duplicates(['case_id'])
        #labels
        with open(pkl_path, 'rb') as f:
            labels = pickle.load(f)
        disc_labels = np.array([labels[id] for id in patients_df['case_id'] if id in labels.keys()])
        patients_df = patients_df[patients_df['case_id'].isin(labels.keys())]
        patients_df.insert(2, 'label', disc_labels.astype(int))

        slide_data = patients_df
        new_cols = list(slide_data.columns[-1:]) + list(slide_data.columns[:-2])
        slide_data = slide_data[new_cols]
        self.slide_data = slide_data
        self.metadata = slide_data.columns[:11]
        self.mode = mode


        ### Signatures
        self.apply_sig = apply_sig
        if self.apply_sig:
            self.signatures = pd.read_csv('./datasets_csv_sig/signatures.csv')
        else:
            self.signatures = None

        if print_info:
            self.summarize()


    def patient_data_prep(self):
        patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
        patient_labels = []
        
        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.slide_data['label'][locations[0]] # get patient label
            patient_labels.append(label)
        
        self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}




    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        else:
            return len(self.slide_data)

    def summarize(self):
        # print("label column: {}".format(self.label_col))
        # print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.num_classes))
        print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
        # for i in range(self.num_classes):
        #     print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
        #     print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))


    def get_split_from_df(self, all_splits: dict, split_key: str='train', scaler=None):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(df_slice, self.label_path,  metadata=self.metadata, mode=self.mode, 
                                    signatures=self.signatures, data_dir=self.data_dir, 
                                    patient_dict=self.patient_dict, num_classes=self.num_classes)
        else:
            split = None
        
        return split


    def return_splits(self, from_id: bool=True, csv_path: str=None):
        if from_id:
            raise NotImplementedError
        else:
            assert csv_path 
            all_splits = pd.read_csv(csv_path)
            train_split = self.get_split_from_df(all_splits=all_splits, split_key='train')
            val_split = self.get_split_from_df(all_splits=all_splits, split_key='val')
            test_split = self.get_split_from_df(all_splits=all_splits, split_key='test')

            ### --> Normalizing Data
            print("****** Normalizing Data ******")
            # scalers = train_split.get_scaler()
            # train_split.apply_scaler(scalers=scalers)
            # val_split.apply_scaler(scalers=scalers)
            #test_split.apply_scaler(scalers=scalers)
            ### <--
        return train_split, val_split, test_split


    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        return None

    def __getitem__(self, idx):
        return None


class Generic_MIL_classification_Dataset(Generic_WSI_classification_Dataset):
    def __init__(self, data_dir, mode: str='omic', **kwargs):
        super(Generic_MIL_classification_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.mode = mode
        self.use_h5 = False

    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def __getitem__(self, idx):
        case_id = self.slide_data['case_id'][idx]
        label = self.slide_data['label'][idx]
        # event_time = self.slide_data[self.label_col][idx]
        # c = self.slide_data['censorship'][idx]
        slide_ids = self.patient_dict[case_id]

        if type(self.data_dir) == dict:
            source = self.slide_data['oncotree_code'][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir
        
        if not self.use_h5:
            if self.data_dir:
                if self.mode == 'path':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path,map_location=torch.device('cpu'))
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    return (path_features, torch.zeros((1,1)), label)

                elif self.mode == 'omic':
                    omic1 = torch.tensor(self.genomic_features[self.omic_names[0]].iloc[idx].values)
                    omic2 = torch.tensor(self.genomic_features[self.omic_names[1]].iloc[idx].values)
                    omic3 = torch.tensor(self.genomic_features[self.omic_names[2]].iloc[idx].values)
                    omic4 = torch.tensor(self.genomic_features[self.omic_names[3]].iloc[idx].values)
                    omic5 = torch.tensor(self.genomic_features[self.omic_names[4]].iloc[idx].values)
                    omic6 = torch.tensor(self.genomic_features[self.omic_names[5]].iloc[idx].values)

                    return (torch.zeros((1,1)), omic1, omic2, omic3, omic4, omic5, omic6, label)
                    # return (torch.zeros((1,1)), genomic_features, label, event_time, c)

                # elif self.mode == 'pathomic':
                #     path_features = []
                #     for slide_id in slide_ids:
                #         wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                #         wsi_bag = torch.load(wsi_path,map_location=torch.device('cpu'))
                #         # wsi_bag = torch.ones((1000, 1024)) / 10
                #         path_features.append(wsi_bag)
                #     path_features = torch.cat(path_features, dim=0)
                #     genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                #     return (path_features, genomic_features, label, event_time, c)

                elif self.mode == 'pathomic':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path,map_location=torch.device('cpu'))
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    omic1 = torch.tensor(self.genomic_features[self.omic_names[0]].iloc[idx].values)
                    omic2 = torch.tensor(self.genomic_features[self.omic_names[1]].iloc[idx].values)
                    omic3 = torch.tensor(self.genomic_features[self.omic_names[2]].iloc[idx].values)
                    omic4 = torch.tensor(self.genomic_features[self.omic_names[3]].iloc[idx].values)
                    omic5 = torch.tensor(self.genomic_features[self.omic_names[4]].iloc[idx].values)
                    omic6 = torch.tensor(self.genomic_features[self.omic_names[5]].iloc[idx].values)

                    return (path_features, omic1, omic2, omic3, omic4, omic5, omic6, label)
                else:
                    raise NotImplementedError('Mode [%s] not implemented.' % self.mode)
                ### <--
            else:
                return slide_ids, label


class Generic_Split(Generic_MIL_classification_Dataset):
    def __init__(self, slide_data, label_path, metadata, mode, signatures=None, data_dir=None, patient_dict=None, num_classes=2):
        self.use_h5 = False
        self.slide_data = slide_data
        self.metadata = metadata
        self.mode = mode
        self.data_dir = data_dir
        self.num_classes = num_classes
        # self.label_col = label_col
        self.label_path = label_path
        self.patient_dict = patient_dict
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

        ### --> Initializing genomic features in Generic Split
        self.genomic_features = self.slide_data.drop(self.metadata, axis=1)
        self.signatures = signatures


        def series_intersection(s1, s2):
            return pd.Series(list(set(s1) & set(s2)))
        
        
        if self.signatures is not None:
            self.omic_names = []
            for col in self.signatures.columns:
                omic = self.signatures[col].dropna().unique()
                # omic = np.concatenate([omic+mode for mode in ['_mut', '_cnv', '_rnaseq']])
                omic = np.concatenate([omic+mode for mode in ['_rnaseq']])
                omic = sorted(series_intersection(omic, self.genomic_features.columns))
                self.omic_names.append(omic)
            self.omic_sizes = [len(omic) for omic in self.omic_names]
        # print("Shape", self.genomic_features.shape)
        ### <--

    def __len__(self):
        return len(self.slide_data)

    ### --> Getting StandardScaler of self.genomic_features
    def get_scaler(self):
        scaler_omic = StandardScaler().fit(self.genomic_features)
        return (scaler_omic,)
    ### <--

    ### --> Applying StandardScaler to self.genomic_features
    def apply_scaler(self, scalers: tuple=None):
        transformed = pd.DataFrame(scalers[0].transform(self.genomic_features))
        transformed.columns = self.genomic_features.columns
        self.genomic_features = transformed
    ### <--