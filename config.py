import torch
import os

class Config(object):
    def __init__(self):
        DB = 'db4_control'
        self.use_cuda = False
        self.threads = 1

        if not self.use_cuda:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:0')
        
        if DB == 'data_atlas':
            # DB info
            self.number_of_class = 73
            self.input_size = 15519
            self.rna_paths = ['data_atlas/atlas_rna_dropseq.npz',
                              'data_atlas/atlas_rna_facs.npz']
            self.rna_labels = ['data_atlas/atlas_rna_dropseq_cellTypes.txt',
                               'data_atlas/atlas_rna_facs_cellTypes.txt']
            self.atac_paths = ['data_atlas/atlas_atac.npz']
            # Optional. If atac_labels are provided, accuracy after knn would be provided.
            self.atac_labels = ['data_atlas/atlas_atac_cellTypes.txt']
            self.rna_protein_paths = []
            self.atac_protein_paths = []

            # Training config
            self.batch_size = 256
            self.lr_stage1 = 0.0001
            self.lr_stage3 = 0.001
            self.lr_decay_epoch = 10
            self.epochs_stage1 = 10
            self.epochs_stage3 = 10
            self.p = 0.8
            self.embedding_size = 64
            self.momentum = 0.9
            self.center_weight = 10
            self.with_crossentorpy = True
            self.seed = 1
            self.checkpoint = ''


        elif DB == "db4_control":
            self.number_of_class = 7  # Number of cell types in CITE-seq data
            # Number of common genes and proteins between CITE-seq data and ASAP-seq
            self.input_size = 17668
            # RNA gene expression from CITE-seq data
            self.rna_paths = ['data/citeseq_control_rna.npz']
            # CITE-seq data cell type labels (coverted to numeric)
            self.rna_labels = ['data/citeseq_control_cellTypes.txt']
            # ATAC gene activity matrix from ASAP-seq data
            self.atac_paths = ['data/asapseq_control_atac.npz']
            # ASAP-seq data cell type labels (coverted to numeric)
            self.atac_labels = ['data/asapseq_control_cellTypes.txt']
            # Protein expression from CITE-seq data
            self.rna_protein_paths = ['data/citeseq_control_adt.npz']
            # Protein expression from ASAP-seq data
            self.atac_protein_paths = ['data/asapseq_control_adt.npz']

            # Training config
            #self.batch_size = 256
            #self.lr_stage1 = 0.01
            #self.lr_stage3 = 0.01
            #self.lr_decay_epoch = 20
            #self.epochs_stage1 = 20
            #self.epochs_stage3 = 20
            #self.p = 0.8
            #self.embedding_size = 64
            #self.momentum = 0.9
            #self.center_weight = 1
            #self.with_crossentorpy = True
            #self.seed = 1
            #self.checkpoint = ''
            self.batch_size = 256
            self.lr_stage1 = 0.01
            self.lr_stage3 = 0.01
            self.lr_decay_epoch = 10
            self.epochs_stage1 = 10
            self.epochs_stage3 = 10
            self.p = 0.9
            self.embedding_size = 64
            self.momentum = 0.99999
            self.center_weight = 1
            self.with_crossentorpy = True
            self.seed = 1
            self.checkpoint = ''	            

        



