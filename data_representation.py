#!/opt/anaconda3/bin/python


# warnings.filterwarnings('ignore')



import os
from datetime import datetime
import calendar
import math
from random import randint
import random
from collections import Counter


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
import matplotlib.pylab as pylab
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch








from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split


from gensim import corpora
import openpyxl
import fasttext
import pickle


from embedding_methods import one_hot_encode_sequences
from embedding_methods import embed_sequences_with_bow_countvect
from embedding_methods import embed_sequences_with_gensim_doc2bow_tfidf
from embedding_methods import embed_sequences_with_keras
from embedding_methods import embed_sequences_with_gensim_word2vec
from embedding_methods import embed_sequences_with_gensim_word2vec_cbow
from embedding_methods import embed_sequences_with_gensim_word2vec_skipgram
from embedding_methods import embed_sequences_with_fasttext_cbow
from embedding_methods import embed_sequences_with_fasttext_skipgram
from embedding_methods import embed_sequences_with_fasttext_class_trained


from sirna_model_building_helper_methods import classify
from sirna_model_building_helper_methods import classify_no_undefined

from sirna_model_building_helper_methods import get_flanking_sequence
from sirna_model_building_helper_methods import get_20mer_from_16mer



import logging




params = {'legend.fontsize': 12,
          'figure.figsize': (6, 4),
          'axes.labelsize': 12,
          'axes.titlesize': 12,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'font.family': 'Arial', # TODO: comment out if running on the cluster!
          }
pylab.rcParams.update(params)

#########################################################################################################################################
#####################################      Constants (Global Variables/Dictionaries)        #############################################
#########################################################################################################################################

training_set_plot_color = '#1494DF' #'#D46F37'
testing_set_plot_color = '#4BB3B1'
external_set_plot_color = '#eb9834'
paramopt_set_plot_color = '#6359A4'



ineff_color = '#3AA6E2'
eff_color = '#F7B531'
undef_color = '#B6B6B7'


# For organizing output files, everything will go into this folder
all_output_dir = 'output_model_fitting/'

# For storing parameter info from data processing to access pre-processed datasets
processed_dataset_parameters_index_file = all_output_dir+'data-processing-param-index.csv'
# For storing parameter info from model fitting
model_fitting_parameters_index_file = all_output_dir+'model-fitting-param-index.csv'

# Directory path to and file holding siRNA data
# input_data_dir = 'new_input_data/'
# #input_data_dir = '/Users/kmonopoli/Dropbox (UMass Medical School)/compiling_sirna_screening_data/'
# # input_data_file = 'cleaned_no-bad-or-duplicate-screens_sirna_screen_data_4392sirnas-bdna-75-genes_JAN-29-2024.csv'
# #input_data_file = 'cleaned_no-bad-or-duplicate-screens_sirna_screen_data_(4392sirnas-bdna|75-genes)_JAN-29-2024.csv'
# input_data_file = 'training_sirna_screen_data_bdna-human-p3_1903-sirnas_MAR-21-2024.csv'

# Holds additional siRNA data for evaluating final models
# external_data_file_dict = {
#     True: 'randomized_sirna_screen_data_777-sirnas_p3-bdna_FEB-28-2024.csv', # randomize_ext_data__ = True
#     False: 'newly_added_sirna_screen_data_777-sirnas|-bdna_FEB-22-2024.csv'  # randomize_ext_data__ = False
#     # False: 'cleaned_no-bad-or-duplicate-screens_sirna_screen_data_4392sirnas-bdna-75-genes_JAN-29-2024.csv'  # randomize_ext_data__ = False
# }



# Dictionaries used for labelling
remove_undefined_label_dict = {True: 'removed undefined siRNAs', False: 'retained undefined siRNAs'}
remove_undefined_label_dict_abbrev = {True: 'rm-u', False: 'kp-u'}

includes_targ_region_dict = {
    'flanking_and_target_regions': True,
    'target_regions_only': True,
    'flanking_regions_only': False
}

param_norm_label_dict = {True: 'normalized', False: 'non-normalized'}

ulabelled_data_file_dict = {
    # TODO: update Unlabelled Data files to reflect training dataset (human P3)
    # Unlabelled siRNA data Generated Randomly (i.e. as random sequences of A/U/C/G)
    #'unlab-randomized': 'new_input_data/' + '/semisupervised_for_cluster_oct-19-2023/' + 'unlabelled_randomized_sirna_data_37050-sirnas_hm_SEP-12-2023_pog.csv',
    # 'unlab-randomized': 'new_input_data/' + 'unlabelled_randomized_sirna_data_38024-sirnas_SEP-12-2023.csv',
    'unlab-randomized': 'new_input_data/' + 'unlabelled_sequences-random_sirna_data_(170000-sirnas)_FEB-29-2024.csv',

    # Unlabelled siRNA Data Generated from Targeted Transcripts - evenly distributed throughout
    'sequences-from-targeted-transcripts': 'new_input_data/' + 'unlabelled_unweighted-sequences-from-targeted-transcripts_sirna_data_(131181-sirnas|human|P3|bDNA|46txs)_FEB-29-2024.csv',

    # Unlabelled siRNA Data Generated from Targeted Transcripts - weighted by representation
    #'weighted-sequences-from-targeted-transcripts': 'new_input_data/' + 'unlabelled_weighted-sequences-from-targeted-transcripts_sirna_data_37928-sirnas_SEP-18-2023.csv',
    'weighted-sequences-from-targeted-transcripts': 'new_input_data/' + 'unlabelled_weighted-sequences-from-targeted-transcripts_sirna_data_(146977-sirnas|human|P3|bDNA|46txs)_FEB-29-2024.csv',

    # Unlabelled siRNA Data Generated from Transcriptome (including untargeted transcripts)
    # 'sequences-from-all-transcripts': 'new_input_data/' + 'unlabelled_sequences-from-species-transcriptomes_sirna_data_39000-sirnas_OCT-5-2023.csv',
    'sequences-from-all-transcripts': 'new_input_data/' + 'unlabelled_sequences-from-transcriptomes_sirna_data_(170000-sirnas|human|P3|bDNA)_FEB-29-2024.csv',
}


chemical_scaffold_dict = {
    '-'.join(['P3', 'P2', 'P5', 'P3 Asymmetric', 'P3 Blunt', 'O-Methyl Rich']): 'p2p3p5p3ap3bome',
    '-'.join(['P3', 'P2', 'P5']): 'p2p3p5',
    '-'.join(['P3']): 'p3',
}

model_type_dict = {
    'random-forest': 'rf',
    'semi-sup-random-forest': 'ssrf',
    'sup-svm':'svm',
    'semi-sup-svm':'tsvm',
    'semi-sup-label-propagation':'sslp',
    'semi-sup-label-spreading':'ssls',
    'linear-classification':'linclf',
    'PARAMOPT':'pomdl',
}




model_dict = { # Dictionary of ACTUAL models for model parameter optimization
    'random-forest': RandomForestClassifier(),
    'semi-sup-random-forest': SelfTrainingClassifier(RandomForestClassifier( max_depth=3)), # NOTE: to speed up fitting set max_depth
    'sup-svm': SVC(kernel="rbf", gamma=0.5, probability=True), # TODO: consider other SVM parameters
    'semi-sup-svm': SelfTrainingClassifier(SVC(kernel="rbf", gamma=0.5, probability=True)), # TODO: consider other SVM parameters
    'semi-sup-label-propagation': SelfTrainingClassifier(LabelPropagation()),
    'semi-sup-label-spreading': SelfTrainingClassifier(LabelSpreading()),
    'linear-classification':LogisticRegression(),
}

param_id_dict = {  # Dictionary containing all possible parameters to optmize mapped to a single unique character to be used for labelling files/directories
    'kmer-size':'k',
    'flank-length':'l',
    'model':'m',
    'window-size':'w',
    'word-frequency-cutoff':'q',
    'ANN_output_dimmension':'d',
    'unlabeled_data_type':'u',
    'unlabeled_data_size':'z',
    'feature_encoding':'e',
    'None':'n',
    # Note: if add more parameters to optimize, add them here
}



feature_encodings_dict = {
    'one-hot':'oh',
    'bow-countvect':'bowcv',
    'bow-gensim':'bowgen',
    # WRONG 'bow-gensim-weights-times-values':'bowgenwtv',
    # WRONG 'bow-gensim-weights-times-values-adjusted':'bowgenwtva',
    'bow-gensim-weights':'bowgenw',
    # WRONG 'bow-gensim-values':'bowgenv',
    # WRONG 'bow-gensim-values-adjusted':'bowgenva',
    'ann-keras':'annk',
    'ann-word2vec-gensim':'w2v',
    'ann-word2vec-gensim-cbow':'w2vcbow',
    'ann-word2vec-gensim-skipgram':'w2vsg',
    'ann-fasttext-cbow':'annftxcbow',
    'ann-fasttext-skipgram':'annftxsg',
    'ann-fasttext-class-trained':'annftlab',

}


feature_encodings_titles_dict = {
    'one-hot':'One-Hot',
    'bow-countvect':'BOW cv',
    'bow-gensim':'BOW g',
    # WRONG 'bow-gensim-weights-times-values': 'BOW g wxv',
    # WRONG 'bow-gensim-weights-times-values-adjusted':  'BOW g wxv adj',
    'bow-gensim-weights':  'BOW g w',
    # WRONG 'bow-gensim-values':  'BOW g v',
    # WRONG 'bow-gensim-values-adjusted': 'BOW g v adj',
    'ann-keras':'ANN',
    'ann-word2vec-gensim':'Word2Vec',
    'ann-word2vec-gensim-cbow':'Word2Vec CBOW',
    'ann-word2vec-gensim-skipgram':'Word2Vec sg',
    'ann-fasttext-cbow':'ANN ftxt CBOW',
    'ann-fasttext-skipgram':'ANN ftxt sg',
    'ann-fasttext-class-trained':'ANN ftxt lab',
}

expr_key_norm_dict = {
    True:'expression_percent_normalized_by_max_min', # 'normalized_expression_percent'
    False:'expression_percent'}

supported_model_types = list(model_type_dict.keys())
unlabelled_data_types = list(ulabelled_data_file_dict.keys())
region_types = list(includes_targ_region_dict.keys())
optimizing_parameters = list(param_id_dict.keys())
encodings_ls = list(feature_encodings_dict.keys())

default_params_to_loop_dict = {
    'kmer-size': [2, 3, 5, 7, 9, 12, 15, 17, 20],
    'flank-length': [0, 5, 10,20, 25, 50, 75, 100],
    'model': supported_model_types,
    'window-size': [1, 2, 3, 4, 5, 7, 10, 15, 20],
    'word-frequency-cutoff': [1, 2, 3, 4, 5, 10],
    'ANN_output_dimmension': [4, 10, 20, 50, 100, 200, 500, 1000],
    'unlabeled_data_type': unlabelled_data_types,
    'unlabeled_data_size': [0.00, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00],  # decimal proportion of whole unlabelled dataset to take (0.50 = 50%, 1.00 = 100%)
    'feature_encoding':encodings_ls,
    'None': [],
    # Note: if add more parameters to optimize, add them here
}





class DataRepresentationBuilder:
    #########################################################################################################################################
    #####################################            DataRepresentationBuilder Class            #############################################
    #########################################################################################################################################

    def __init__(self, parameter_to_optimize__,
                 num_rerurun_model_building__=10,  # 25,
                 custom_parameter_values_to_loop__=[],
                 model_type__='random-forest',
                 flank_len__=50, kmer_size__=9,
                 normalized__=True, effco__=25, ineffco__=60, remove_undefined__=True,
                 region__='flanking_and_target_regions',
                 screen_type_ls__=['bDNA'], species_ls__=['human'],
                 chemical_scaffold_ls__=['P3'],
                 window_size__=1, word_freq_cutoff__=1, output_dimmension__=20,
                 unlabelled_data__='sequences-from-all-transcripts',
                 unlabeled_data_size__=1.00,
                 plot_grid_splits__=False, plot_extra_visuals__=False,
                 run_round_num__=1,
                 encoding_ls__ = ['one-hot', 'ann-word2vec-gensim', 'bow-gensim', 'ann-keras', 'bow-countvect'],
                 metric_used_to_id_best_po__='F-Score',
                 f_beta__ = 0.1,#0.25, #0.5
                 run_param_optimization__ = True,
                 use_existing_processed_dataset__ = False,
                 apply_final_models_to_external_dataset__ = False, # whether or not to use external_data_file to evaluate final models
                 ext_species_ls__=['human'], ext_chemical_scaffold_ls__=['P3'],
                 #randomize_ext_data__ = False, # Not used if using external_data_file__ parameter
                 external_data_file__ = 'newly_added_sirna_screen_data_777-sirnas|-bdna_FEB-22-2024.csv', # NO randomization of extenral data
                 # external_data_file__ = 'randomized_sirna_screen_data_777-sirnas_p3-bdna_FEB-28-2024.csv',  # Randomization of extenral data
                 # external_data_file__  = 'cleaned_no-bad-or-duplicate-screens_sirna_screen_data_4392sirnas-bdna-75-genes_JAN-29-2024.csv',  # NO randomization of extenral data
                 input_data_dir__='new_input_data/',
                 input_data_file__ = 'training_sirna_screen_data_bdna-human-p3_1903-sirnas_MAR-21-2024.csv',
                 include_random_background_comparison__ = False,
                 plot_starting_data_thresholds__ = True,
                 ):
        '''
        #########################################################################################################################################
        #####################################        Dataset Parameters (Instance Variables)        #############################################
        #########################################################################################################################################
        '''
        logging.basicConfig(level=logging.DEBUG, filename="logfile", filemode="a+",
                            format="%(asctime)-15s %(levelname)-8s %(message)s")
        logging.info("\n\n\n ------------------- NEW RUN ------------------- \n\n\n")

        pd.set_option('display.max_columns', None)

        if not run_param_optimization__:
            logging.info("\n\n\nIMPORTANT: run_param_optimization__ is set to ("+str(run_param_optimization__)+") so will not be running parameter optimization"+
                  "(will only be building final models). Any mention of parameter optimization from Constructor can be ignored.\n\n\n" )

        if not run_param_optimization__:
            if parameter_to_optimize__ != 'None':
                raise Exception("ERROR: if run_param_optimization__ is set to (" + str(run_param_optimization__) + " parameter_to_optimize__ must be set to 'None' but is currently (" + str(parameter_to_optimize__) + ")")

        # Clean input data
        if region__ not in region_types:
            raise ValueError("ERROR: Invalid region name. Expected one of: %s" % region_types)
        if parameter_to_optimize__ not in optimizing_parameters:
            raise ValueError("ERROR: Invalid parameter to optimize. Expected one of: %s" % optimizing_parameters)
        if model_type__ not in supported_model_types:
            raise ValueError("ERROR: Invalid model type. Expected one of: %s" % supported_model_types)
        if unlabelled_data__ not in unlabelled_data_types:
            raise ValueError("ERROR: Invalid unlabelled_data__ name. Expected one of: %s" % unlabelled_data_types)
        for e in encoding_ls__:
            if e not in encodings_ls:
                raise ValueError("ERROR: Invalid encoding_ls__ name. Expected one of: %s" % encodings_ls)
        if (parameter_to_optimize__ == 'flank-length') and ('flank' not in region__):
            raise ValueError("ERROR: If optimizing flank-length parameter, region_ must contain flanking regions, inputted region__ = " + str(region__) + " does not!")
        ## flank_len_ cannot zero in cases where considering flanking regions
        if (parameter_to_optimize__ != 'flank-length') and ('flank' in region__) and (flank_len__ == 0):
            raise ValueError("ERROR: flank_len_ cannot be zero with region_ " + str(region__) + "!")
        ## If using dualglo data there are no flanking regions so flank_len_ must equal 0, and region_ must equal 'target_regions_only'
        if ('DualGlo' in screen_type_ls__) and ('flank' in region__):
            raise ValueError("ERROR: If screen_type_ls__ includes DualGlo Data, region__ cannot include flanking regions, inputted region__ = " + str(region__))
        if ('DualGlo' in screen_type_ls__) and (flank_len__ != 0):
            raise ValueError("ERROR: If screen_type_ls__ includes DualGlo Data, flank_len__ must be set to 0, inputted flank_len__ = " + str(flank_len__))

        if ((parameter_to_optimize__ == 'unlabeled_data_type') or (parameter_to_optimize__ == 'unlabeled_data_size')) and ('semi-' not in model_type__):
            raise ValueError(
                "ERROR: If optimizing unlabeled data type or size model type must be semi-supervised, inputted model_type__ = " + str(
                    model_type__))

        if ((unlabeled_data_size__ > 1.0) or (unlabeled_data_size__ < 0.0) or (type(unlabeled_data_size__) != float)):
            raise ValueError("ERROR: invalid unlabeled_data_size__ : " + str(unlabeled_data_size__) + " must be a float between 0.00 and 1.00")

        if ((flank_len__ > 100) or (flank_len__ < 0)):# or type(flank_len__ != int)):
            raise ValueError("ERROR: Invalid flank_length__ : " + str(flank_len__) + " , must be an integer between 0 and 100")

        # If any custom parameter looping values are included, check they are supported
        if custom_parameter_values_to_loop__ != []:
            if parameter_to_optimize__ == 'model':
                # check model list is supported
                for m_ in custom_parameter_values_to_loop__:
                    if m_ not in supported_model_types:
                        raise ValueError("ERROR: Invalid model type in custom_parameter_to_optimize__ list: " + str(m_) + " , Expected one of: %s" % supported_model_types)
            if parameter_to_optimize__ == 'unlabeled_data_size':
                # check parameters are all between 0.00 and 1.00
                for s_ in custom_parameter_values_to_loop__:
                    if (s_ > 1.0) or (s_ < 0.0) or (type(s_) != float):
                        raise ValueError("ERROR: Invalid unlabeled_data_size_ in custom_parameter_to_optimize__ list : " + str(s_) + " , must be a float between 0.00 and 1.00")
            if parameter_to_optimize__ == 'flank-length':
                # check that flank lengths are between 0 and 100
                for l_ in custom_parameter_values_to_loop__:
                    if (l_ > 100) or (l_ < 0) or (type(l_) != int):
                        raise ValueError("ERROR: Invalid flank_length in custom_parameter_to_optimize__ list : " + str(l_) + " , flank_length be an integer between 0 and 100")
            if parameter_to_optimize__ == 'unlabeled_data_type':
                # check unlabeled data type list is supported
                for t_ in custom_parameter_values_to_loop__:
                    if t_ not in unlabelled_data_types:
                        raise ValueError("ERROR: Invalid unlabelled_data_type in custom_parameter_to_optimize__ list : " + str(t_) + " , Expected one of: %s" % unlabelled_data_types)
            if parameter_to_optimize__ == 'feature_encoding':
                # check unlabeled data type list is supported
                for t_ in custom_parameter_values_to_loop__:
                    if t_ not in encodings_ls:
                        raise ValueError("ERROR: Invalid feature_encoding in custom_parameter_to_optimize__ list : " + str(t_) + " , Expected one of: %s" % encodings_ls)
        # Set General parameters
        self.input_data_dir = input_data_dir__ #= 'new_input_data/',
        self.input_data_file = input_data_file__ #= 'training_sirna_screen_data_bdna-human-p3_1903-sirnas_MAR-21-2024.csv',

        self.plot_starting_data_thresholds_ = plot_starting_data_thresholds__
        self.num_rerurun_model_building = num_rerurun_model_building__  # times to rerun building models using INDEPENDENT 80:10:10 datasets
        self.run_round_num = run_round_num__  # NOTE: for running additional looping beyond num_rerurun_model_building__, currently not used
        self.screen_type_ls = screen_type_ls__  # NOTE: DualGlo will not have flanking regions
        self.species_ls = species_ls__
        self.chemical_scaffold_ls = chemical_scaffold_ls__
        self.chemical_scaffold_lab = chemical_scaffold_dict['-'.join(self.chemical_scaffold_ls)]
        self.normalized_ = normalized__
        self.effco_ = effco__
        self.ineffco_ = ineffco__
        self.remove_undefined_ = remove_undefined__
        self.region_ = region__
        self.includes_targ_region_ = includes_targ_region_dict[self.region_]
        self.parameter_to_optimize = parameter_to_optimize__
        self.plot_grid_splits_ = plot_grid_splits__
        self.plot_extra_visuals_ = plot_extra_visuals__
        self.metric_used_to_id_best_po = metric_used_to_id_best_po__  # TODO: change to different metric?
        self.f_beta_ = f_beta__
        self.run_param_optimization_ = run_param_optimization__
        self.use_existing_processed_dataset_ = use_existing_processed_dataset__
        self.apply_final_models_to_external_dataset_ = apply_final_models_to_external_dataset__ # whether or not to use external_data_file to evaluate final models
        # self.external_data_file_ = external_data_file_dict[randomize_ext_data__]
        self.external_data_file_ = external_data_file__
        self.ext_species_ls = ext_species_ls__ # TODO: include this information in data saving parameters and labels
        self.ext_chemical_scaffold_ls = ext_chemical_scaffold_ls__ # TODO: include this information in data saving parameters and labels
        #self.randomize_ext_data_ = randomize_ext_data__
        if ('random' in external_data_file__) or ('shuffle' in external_data_file__):
            self.randomize_ext_data_ = True
        else:
            self.randomize_ext_data_ = False

        self.include_random_background_comparison_ = include_random_background_comparison__

        # Splits for Train:Parameter Opt:Test
        self.test_set_size_pcnt_ = 15
        self.paramopt_set_size_pcnt_ = 10
        self.split_set_size_pcnt_ = self.test_set_size_pcnt_ + self.paramopt_set_size_pcnt_  # 20 # for 80:10:10 #20 #10 # Percentage of the data to be held out when building models (and used to evaluate the final model)
        self.allowed_classification_prop_deviation_pcnt_ = 5  # percentage of data allowed to be off for classification proportions

        self.kmer_size_ = kmer_size__
        self.flank_len_ = flank_len__  # length on each side (e.g. 50 --> 120nt total length: 50mer 5' flank +20mer target region + 50mer 3' flank)
        self.model_type_ = model_type__
        self.window_size_ = window_size__
        self.word_freq_cutoff_ = word_freq_cutoff__  # Number of times a word must occur in the Bag-of-words Corpus --> when word_freq_cutoff = 1 only include words that occur more than once
        self.output_dimmension_ = output_dimmension__  # output dimmensino of ANN embedding
        self.expr_key = expr_key_norm_dict[self.normalized_]
        self.feature_encoding_ls = encoding_ls__

        ## Parameters to Optimize
        # NOTE: if add more parameters to optimize add them to these two set of if statements below
        if len(custom_parameter_values_to_loop__) != 0:
            self.param_values_to_loop_ = custom_parameter_values_to_loop__
        else:
            self.param_values_to_loop_ = default_params_to_loop_dict[self.parameter_to_optimize]
            if self.param_values_to_loop_ == []:
                logging.info("\n\n\nNOTE: running without optimizing any parameters\n\n\n")
            else:
                self.param_opt_working_keys_ls = []

        logging.info("\n\n\nNOTE: parameter_to_optimize = ("+str(self.parameter_to_optimize)+") if this parameter was set in the input, it will be ignored!"+
              "\t Instead will consider these parameter values: [ "+str(', '.join([str(x) for x in self.param_values_to_loop_])) + "]\n\n\n" )

        if self.parameter_to_optimize == 'kmer-size':
            self.kmer_size_ = 'PARAMOPT' 
        elif self.parameter_to_optimize == 'flank-length':
            self.flank_len_ = 'PARAMOPT' 
        elif self.parameter_to_optimize == 'model':
            self.model_type_ = 'PARAMOPT' 
        elif self.parameter_to_optimize == 'window-size':
            self.window_size_ = 'PARAMOPT' 
        elif self.parameter_to_optimize == 'word-frequency-cutoff':
            self.word_freq_cutoff_ = 'PARAMOPT' 
        elif self.parameter_to_optimize == 'ANN_output_dimmension':
            self.output_dimmension_ = 'PARAMOPT' 
        elif self.parameter_to_optimize == 'unlabeled_data_type':
            self.unlabelled_data_ = 'PARAMOPT' 
            self.input_unlabeled_data_file = 'PARAMOPT'   # NOTE: will need to retreive later from ulabelled_data_file_dict
        elif self.parameter_to_optimize == 'unlabeled_data_size':
            self.unlabeled_data_size_ = 'PARAMOPT' 
        elif self.parameter_to_optimize == 'feature_encoding':
            self.feature_encoding_ls = self.param_values_to_loop_
        else:
            self.parameter_to_optimize = 'None'
            if len(custom_parameter_values_to_loop__) != 0:
                raise ValueError("ERROR: custom_parameter_values_to_loop__ is not empty but parameter_to_optimize was:" + str(self.parameter_to_optimize))

        # Set Unlabelled Data Source (for Semisupervised)
        if (('semi-sup' in self.model_type_) or (self.parameter_to_optimize == 'model')):
            if (self.parameter_to_optimize != 'unlabeled_data_type'):
                self.unlabelled_data_ = unlabelled_data__
                self.input_unlabeled_data_file = ulabelled_data_file_dict[self.unlabelled_data_]
            if (self.parameter_to_optimize != 'unlabeled_data_size'):
                self.unlabeled_data_size_ = unlabeled_data_size__
        else:
            self.unlabelled_data_ = 'PARAMOPT' 
            self.input_unlabeled_data_file = None

        # Get date - for labeling and saving files
        self.date_ = calendar.month_abbr[datetime.now().month].upper() + '-' + str(datetime.now().day) + '-' + str(
            datetime.now().year)
        logging.info("Date set successfully!"+str(self.date_))

        #logging.info("Construction complete!")


        existing_dataset_dir__ = ''
        if self.use_existing_processed_dataset_:
            logging.info("\n\n\nSearching for existing pre-processed data with matching parameters...\n")
            # Exclude semi-supervised model building from utilizing pre-loaded data (since embeddings of unlabeled data are not stored and all data must be embedded at the same time for some embedding methods)
            if ('semi-sup-' in self.model_type_):
                raise Exception("ERROR: cannot use existing processed datasets for model type (" + str(self.model_type_) + ") because is semi-supervised learning and embeddigs of unlabeled data are not stored")
            if (self.parameter_to_optimize == 'model') and (np.any(['semi-sup' in x for x in self.param_values_to_loop_])):
                raise Exception("ERROR: cannot use existing processed datasets if optimizing model type when one or more of those models utilize semi-supervised learning" +
                                " (" + str(self.param_values_to_loop_) + ") because embeddings of unlabeled data are not stored")

            # First identify directory of pre-processed dataset matching current parameters to load in (if it exists)
            existing_dataset_dir__ = self.find_existing_processed_datasets()

        # Load in existing pre-processed data (only if it exists)
        if self.use_existing_processed_dataset_ and (existing_dataset_dir__ != ''):
            logging.info("\n\n\nIMPORTANT: use_existing_processed_dataset__ is set to (" + str(use_existing_processed_dataset__) + ") so will use existing processed data to build models\n\n\n")
            logging.info("Loading in pre-processed dataset ("+str(existing_dataset_dir__)+") ...")
            self.load_in_existing_process_datasets(existing_dataset_dir__)

            logging.info("Loading existing pre-processed datasets complete!")

        else:
            logging.info("\nCreating processed datasets...\n")
            # General label describing siRNA data used for model building
            self.all_data_label_str_ = (
                    '-'.join(self.species_ls) +
                    '_' + self.chemical_scaffold_lab +
                    '_' + '-'.join(self.screen_type_ls) +

                    '_' + param_norm_label_dict[self.normalized_] +
                    '_effco-' + str(self.effco_) +
                    '|ineffco-' + str(self.ineffco_) + '-' + remove_undefined_label_dict[self.remove_undefined_]
            ).replace(' ', '_')
            # Abbreviation of General label describing siRNA data used for model building
            self.abbrev_all_data_label_str_ = (
                    '-'.join([x[0] for x in self.species_ls]) +
                    '_' + self.chemical_scaffold_lab +
                    '_' + '-'.join(self.screen_type_ls) +
                    '_' + '-'.join([feature_encodings_dict[e] for e in self.feature_encoding_ls]) +
                    '_' + param_norm_label_dict[self.normalized_].replace('alized', '').replace('-', '') +
                    '_' + str(self.effco_) +
                    '-' + str(self.ineffco_) + '-' + remove_undefined_label_dict_abbrev[self.remove_undefined_]
            ).replace(' ', '_')


            logging.info("\t all_data_label_str_ = "+str(self.all_data_label_str_))
            logging.info("\t abbrev_all_data_label_str_ = " + str(self.abbrev_all_data_label_str_))

            self.create_processed_datasets()
            logging.info("Creating processed datasets complete!\n\n\n")

        logging.info("\nRunning model fittings...")
        self.run_model_fittings()
        logging.info("Model Fittings complete!\n")


        if self.run_param_optimization_:
            logging.info("\nPloting precision-recall curves from Parameter Optimization...")
            self.plot_param_opt_precision_recall_curves()



        if self.apply_final_models_to_external_dataset_:
            logging.info("\nPloting precision-recall curves from Final Model Building evaluated on External Dataset...")
            # self.plot_final_model_precision_recall_curves_on_ext_dataset()
            # self.plot_final_model_top_precision_recall_curves_on_ext_dataset()
            self.plot_final_model_precision_recall_curves_on_ext_dataset_and_test_set()
        else:
            logging.info("\nPloting precision-recall curves from Final Model Building...")
            self.plot_final_model_precision_recall_curves()
            ## self.plot_final_model_top_precision_recall_curves()

        logging.info("\nCurve plotting complete!")

        if self.run_param_optimization_:
            logging.info("\nPlotting box plots from Parameter Optimization...")
            self.plot_param_opt_model_box_plots()

        if self.run_param_optimization_ and self.apply_final_models_to_external_dataset_:
            logging.info("\nPlotting box plots from Final Model Building...")
            self.plot_final_model_box_plots_per_param_val()
            self.plot_final_model_box_plots_per_metric()

            logging.info("\nPlotting box plot for F-score ONLY from Final Model Building evaluated on External Dataset...")
            self.plot_final_model_and_external_data_box_plots_f_score_only()

        
        else:
            if self.apply_final_models_to_external_dataset_:
                logging.info("\nPlotting box plots from Final Model Building evaluated on External Dataset...")
                #*# self.plot_final_model_box_plots_per_metric_on_ext_dataset()
                #*# self.plot_final_model_box_plots_per_param_val_on_ext_dataset()
                self.plot_final_model_and_external_data_box_plots_per_metric()
                self.plot_final_model_and_external_data_box_plots_f_score_only()

            else:
                logging.info("\nPlotting box plots from Final Model Building...")
                self.plot_final_model_box_plots_per_param_val()
                self.plot_final_model_box_plots_per_metric()
            
        
        logging.info("\nBox plotting complete!")

        logging.info("\n\n\nPROCESS FINISHED\n\n\n")
        print('Data information saved to:\n\t', self.all_data_split_dir)
        print('Model information saved to:\n\t',self.output_directory)


        return ## End constructor







    def plot_thresholds(self, df_, figure_label_, output_dir__='', savefig=True):
        fig, ax = plt.subplots()
        fig.set_size_inches(w=5, h=4)


        colors_ls = [x.replace('inefficient', ineff_color).replace('efficient', eff_color).replace('undefined', undef_color) for
                     x in list(df_.sort_values(by=self.expr_key)['class'])]
        ax.bar(
            x=list(range(len(df_))),
            height=df_.sort_values(by=self.expr_key)[self.expr_key],
            color=colors_ls,
            # width=(1.0),
        )

        container2 = ax.errorbar(
            list(range(len(df_))),
            df_.sort_values(by=self.expr_key)[self.expr_key],
            yerr=df_.sort_values(by=self.expr_key)['standard_deviation'],
            lolims=True,
            color='black',
        )

        connector, (caplines,), (vertical_lines,) = container2.lines
        connector.set_alpha(0)
        caplines.set_solid_capstyle('butt')
        try:
            caplines.set_marker(None)
        except:
            pass
        vertical_lines.set_linewidth(1.0)  # 0.5)

        ax.set_ylim(0, max(df_[self.expr_key]) + 0.2 * max(df_[self.expr_key]))
        ax.set_xlim(0, len(df_))
        ax.set_ylabel('Target Gene Expression (%)\nNormalized Per Assay (Gene)')
        ax.set_xlabel('siRNAs (' + str(len(df_)) + ' total)')
        ax.tick_params(axis='x', bottom=False, labelbottom=False)  # remove x-axis ticks and labels

        # Legend
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=eff_color, edgecolor=None,
                  label=('< ' + str(self.effco_) + '% : Efficient (' + str(len(df_[df_['class'] == 'efficient'])) + ' siRNAs)')),
            Patch(facecolor=ineff_color, edgecolor=None, label=('≥ ' + str(self.ineffco_) + '% : Inefficient (' + str(
                len(df_[df_['class'] == 'inefficient'])) + ' siRNAs)')),
            Patch(facecolor=undef_color, edgecolor=None,
                  label=('Undefined (' + str(len(df_[df_['class'] == 'undefined'])) + ' siRNAs)')),
        ]
        ax.legend(handles=legend_elements, loc='upper left', frameon=False, fontsize=9)

        plt.title(figure_label_)
        # plt.title(output_run_file_info_string_.split('effco')[0].replace('_',' ').replace('-',' ')+'\n'+'Thresholds <'+str(effco_)+'% | ≥'+str(ineffco_)+'%',fontsize=12)

        fig.tight_layout()

        if savefig:
            # ** SAVE FIGURE **
            plt.rcParams['svg.fonttype'] = 'none'  # exports text as strings rather than vector paths (images)
            fnm_ = (all_output_dir + output_dir__ + figure_label_.split('\n')[0].replace(' ', '_').lower().replace('%',
                                                                                                                   'pcnt') + '_partition')
            fnm_svg_ = (all_output_dir + output_dir__ + 'svg_figs/' + figure_label_.split('\n')[0].replace(' ',
                                                                                                           '_').lower().replace(
                '%', 'pcnt') + '_partition')

            fig.savefig(fnm_svg_.split('.')[0] + '.svg', format='svg', transparent=True)

            fig.savefig(fnm_.split('.')[0] + '.png', format='png', dpi=300, transparent=False)
            logging.info('Figure saved to:'+str(fnm_) + '.png')

    def plot_proportions_pie(self, figure_label_, output_dir__, df_, undefined_df_, train_split_df_, split_initial_df_,
                             df_train_kfld_, df_test_kfld_, df_paramopt_kfld_, round_ct_, savefig=True):
        # Checking porportions of partitioned data roughly match input parameters

        fig, ax = plt.subplots(1)  # 1,3)
        fig.set_size_inches(w=5, h=5)
        # For Plotting
        import matplotlib.pylab as pylab
        params = {'legend.fontsize': 12,
                  'font.size': 12,
                  }
        pylab.rcParams.update(params)

        ax.set_title('Round ' + str(round_ct_ + 1) + ' Partition\nTotal siRNAs: ' + str(len(train_split_df_)))
        ax.pie(
            [len(df_train_kfld_), len(df_test_kfld_), len(df_paramopt_kfld_)],  # len(split_initial_df_) ],
            autopct='%1.f%%',
            startangle=90,
            colors=['#DB7AC8', '#FECC0A', '#28A18B'],
        )

        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor='#DB7AC8', edgecolor=None, label=(
                    str(100 - self.split_set_size_pcnt_) + '% Round ' + str(round_ct_ + 1) + ' Training Dataset (' + str(
                len(df_train_kfld_)) + ')')),
            Patch(facecolor='#FECC0A', edgecolor=None, label=(
                    str(self.test_set_size_pcnt_) + '% Round ' + str(round_ct_ + 1) + ' Testing Dataset (' + str(
                len(df_test_kfld_)) + ')')),
            Patch(facecolor='#28A18B', edgecolor=None, label=(str(self.paramopt_set_size_pcnt_) + '% Round ' + str(
                round_ct_ + 1) + ' Parameter Optimization Dataset (' + str(len(df_paramopt_kfld_)) + ')')),
        ]

        ax.legend(handles=legend_elements, frameon=False, loc='upper right', bbox_to_anchor=(1.5, 0.1))
        fig.tight_layout()  # NOTE: h and w (above in fig.set_size... MUST be large enough to accomodate legends or will be cut off/squished in output)

        if savefig:
            # ** SAVE FIGURE **
            plt.rcParams['svg.fonttype'] = 'none'  # exports text as strings rather than vector paths (images)
            fnm_ = (all_output_dir + output_dir__ + figure_label_.split('\n')[0].replace(' ', '_').lower().replace('%',
                                                                                                                   'pcnt') + '_partition')
            fnm_svg_ = (all_output_dir + output_dir__ + 'svg_figs/' + figure_label_.split('\n')[0].replace(' ',
                                                                                                           '_').lower().replace(
                '%', 'pcnt') + '_partition')

            fig.savefig(fnm_svg_.split('.')[0] + '.svg', format='svg', transparent=True)

            fig.savefig(fnm_.split('.')[0] + '.png', format='png', dpi=300, transparent=False)
            logging.info('Figure saved to:'+str(fnm_) + '.png')

    def find_existing_processed_datasets(self):
        '''
        Find correct dataset directory name (matching input parameters) for pre-processed (enbedded and split) data
        :return: name of directory holding suitable pre-processed data OR "" if no pre-processed data exist with given parameters
        '''
        # Create string row to use to search processed_dataset_parameters_index_file for an existing pre-processed dataset to load in
        query_data_split_info_dict = {
            ##'all_data_split_dir':self.all_data_split_dir,
            'num_rerurun_model_building': self.num_rerurun_model_building,  # times to rerun building models using INDEPENDENT 80:10:10 datasets
            'run_round_num': self.run_round_num,  # NOTE: for running additional looping beyond num_rerurun_model_building currently not used
            'screen_type_ls': self.screen_type_ls,  # NOTE: DualGlo will not have flanking regions
            'species_ls': self.species_ls,
            'chemical_scaffold_ls': self.chemical_scaffold_ls,
            'chemical_scaffold_lab': self.chemical_scaffold_lab,
            'normalized_': self.normalized_,
            'effco_': self.effco_,
            'ineffco_': self.ineffco_,
            'remove_undefined_': self.remove_undefined_,
            'region_': self.region_,
            'includes_targ_region_': self.includes_targ_region_,
            'parameter_to_optimize': self.parameter_to_optimize,
            'test_set_size_pcnt_': self.test_set_size_pcnt_,
            'paramopt_set_size_pcnt_': self.paramopt_set_size_pcnt_,
            'split_set_size_pcnt_': self.split_set_size_pcnt_,
            'allowed_classification_prop_deviation_pcnt_': self.allowed_classification_prop_deviation_pcnt_,  # percentage of data allowed to be off for classification proportions
            'kmer_size_': self.kmer_size_,
            'flank_len_': self.flank_len_,  # length on each side (e.g. 50 --> 120nt total length: 50mer 5' flank +20mer target region + 50mer 3' flank)
            'window_size_': self.window_size_,
            'word_freq_cutoff_': self.word_freq_cutoff_,  # Number of times a word must occur in the Bag-of-words Corpus --> when word_freq_cutoff' : self.1 only include words that occur more than once
            'output_dimmension_': self.output_dimmension_,  # output dimmensino of ANN embedding
            'expr_key': self.expr_key,
            'feature_encoding_ls': self.feature_encoding_ls,
            'param_values_to_loop_': self.param_values_to_loop_,
            'apply_final_models_to_external_dataset_': self.apply_final_models_to_external_dataset_,

        }
        query_row_string_ = ''
        for k in query_data_split_info_dict.keys():
            if type(query_data_split_info_dict[k]) == list:
                query_row_string_ += (str(';'.join(query_data_split_info_dict[k])) + ',')
            else:
                query_row_string_ += (str(query_data_split_info_dict[k]) + ',')

        # Read processed_dataset_parameters_index_file to identify pre-processed dataset(s) (if any) that match current parameters
        if not os.path.exists(processed_dataset_parameters_index_file):
            # If processed_dataset_parameters_index_file file does not already exist then can't load in data
            logging.info("\n\n\nWARNING: processed_dataset_parameters_index_file (" + str(processed_dataset_parameters_index_file) + ") file containing information to index pre-processed data does not exist, so cannot load in pre-processed dataset \n\n\n")
            return  "" # call create processed datasets since cannot find info to load any existing datasets

        else:
            with open(processed_dataset_parameters_index_file, 'r') as f:
                lines_ = f.readlines()[1:]
            f.close()

            row_strings_ls = []
            for l, indx_ in zip(lines_, range(len(lines_))):
                row_strings_ls.append(str(indx_) + '~~~' + ','.join(l.split('***BREAK***')[0].split(',')[1:]))

            # Identify any values in row_strings_ls that match row_strings_ls
            matching_rows_ls = [x for x in row_strings_ls if x.split('~~~')[-1] == query_row_string_]
            if len(matching_rows_ls) == 0:
                logging.info("\n\n\nWARNING: no data with parameters matching this run found in processed_dataset_parameters_index_file (" + str(processed_dataset_parameters_index_file) + "), cannot load in pre-processed dataset \n\n\n")
                logging.info('\n\nquery_row_string_:')
                logging.info(query_row_string_)
                logging.info('\n\n 1st in row_strings_ls:')
                logging.info(row_strings_ls[0].split('~~~')[-1])
                logging.info('\n\n')
                return "" # call create processed datasets since cannot find info to load any existing datasets
            else:
                # Select directory name based off of matching_rows_ls to return
                logging.info("\n\nFound "+str(len(matching_rows_ls))+ " indicies with data parameters matching those for this run")
                if len(matching_rows_ls) > 1:
                    logging.info("Selecting first occurrence") # TODO: update to pick randomly?
                # If more than one match, picks first occurrence
                indx_match_ = int(matching_rows_ls[0].split('~~~')[0])
                match_dir_ = lines_[indx_match_].split(',')[0]
                logging.info("\nPre-processed data directory selected:\n\t\t",match_dir_,'\n')
                return match_dir_

    def load_in_existing_process_datasets(preproc_data_dir, self):
        '''
        Loads in existing pre-processed (embedded and split) data
        Also updates all class variables/parameters that are set/defined when calling create_processed_datasets()
        Only supported for supervised model building
        '''

        # TODO: Finish writing code to load in and manage utilizing existing dataset:


        raise Exception("TODO: Finish writing code to load in existing pre-processed datasets")



        # TODO: Set parameters normally set when calling create_processed_datasets()

        # TODO: Set parameters normally set when calling perform_feature_embedding()

        # TODO: Set parameters normally set when calling split_train_test_paramopt()

        # TODO: (possibly, might not be necessary since not always called when processing data?) Set parameters normally set when calling plot_data_splits()
        # TODO: (possibly, might not be necessary since not always called when processing data?) Set parameters normally set when calling plot_pie_of_data_splits()
        # TODO: (possibly, might not be necessary since not always called when processing data?) Set parameters normally set when calling plot_bar_data_splits()

        # Set parameters normally set when calling load_in_unlab_data()
        self.indxs_mid_undefined = list(self.df[self.df['numeric_class'] == -1].index)
        self.indxs_labeled_data = list(self.df.index)


    def create_processed_datasets(self):

        logging.info("Creating processed datasets...")

        # Read in Data
        try:
            self.df = pd.read_excel(self.input_data_dir + self.input_data_file)
        except:
            self.df = pd.read_csv(self.input_data_dir + self.input_data_file)
            logging.info("Successfully read in .csv data")

        logging.info("\n\n\nSuccessfully read in - " + str(len(self.df)) + ' siRNAs')
        ########################################################################
        ##                     ~*~ Select Data ~*~                       ##
        ########################################################################
        self.df.drop(columns=['expression_replicate_1', 'expression_replicate_2', 'expression_replicate_3', 'ntc_replicate_1', 'ntc_replicate_2', 'ntc_replicate_3', 'untreated_cells_replicate_1', 'untreated_cells_replicate_2', 'untreated_cells_replicate_3'], inplace=True)

        logging.info(str(self.df['chemical_scaffold'].value_counts()))
        logging.info('')
        logging.info(str(self.df['screen_type'].value_counts()))
        logging.info('')
        logging.info(str(self.df['species'].value_counts()))

        logging.info("Selecting data with screen type:\n"+str(self.screen_type_ls))
        self.df = self.df[self.df['screen_type'].isin(self.screen_type_ls)]
        logging.info("Selecting data with species:\n"+str(self.species_ls))
        self.df = self.df[self.df['species'].isin(self.species_ls)]
        logging.info("Selecting data with chemical scaffold:\n"+str(self.chemical_scaffold_ls))
        self.df = self.df[self.df['chemical_scaffold'].isin(self.chemical_scaffold_ls)]

        logging.info("Training dataset size:"+str(len(self.df)))

        # If Using additional external dataset to evaluate final models Load in
        if self.apply_final_models_to_external_dataset_:
            logging.info("\n\n\nIMPORTANT: apply_final_models_to_external_dataset_ set to ("+str(self.apply_final_models_to_external_dataset_)+") so using additional external dataset to evaluate final models\n\n\n")
            logging.info("\n\n\nLoading in external_data_file_ ("+str(self.external_data_file_)+") along with input_data_file ...\n\n\n")

            try:
                df_ext = pd.read_excel(self.input_data_dir + self.external_data_file_)
                #df_ext = pd.read_excel(self.input_data_dir + self.input_data_file) # for cases where using different data from same dataset (eg chemical scaffolds, species, etc.)

            except:
                df_ext = pd.read_csv(self.input_data_dir + self.external_data_file_)
                #df_ext = pd.read_csv(self.input_data_dir + self.input_data_file) # for cases where using different data from same dataset (eg chemical scaffolds, species, etc.)

            logging.info("Successfully read in external dataset - "+str(len(df_ext))+' siRNAs (NOTE: this includes middle (undefined) efficacy siRNAs')

            ########################################################################
            ##                 ~*~ Select External Data ~*~                       ##
            ########################################################################
            df_ext.drop(columns=['expression_replicate_1', 'expression_replicate_2', 'expression_replicate_3', 'ntc_replicate_1', 'ntc_replicate_2', 'ntc_replicate_3', 'untreated_cells_replicate_1', 'untreated_cells_replicate_2', 'untreated_cells_replicate_3'], inplace=True)


            logging.info("Selecting external data with screen type:\n" +str(self.screen_type_ls))
            df_ext = df_ext[df_ext['screen_type'].isin(self.screen_type_ls)]
            logging.info("Selecting external data with species:\n"+str(self.ext_species_ls))
            df_ext = df_ext[df_ext['species'].isin(self.ext_species_ls)]
            logging.info("Selecting external data with chemical scaffold:\n"+str(self.ext_chemical_scaffold_ls))
            df_ext = df_ext[df_ext['chemical_scaffold'].isin(self.ext_chemical_scaffold_ls)]

            # Randomize expression data
            #if self.randomize_ext_data_:
            # logging.info("\n\n\nWARNING: Randomizing external dataset expression data\n\n\n")
            # #df_ext[self.expr_key] = [float(x) for x in list(np.random.randint(1, high=100, size=len(df_ext)))]

            # ext_expr_ls = list(df_ext[self.expr_key])
            # random.shuffle(ext_expr_ls)
            # df_ext[self.expr_key] = ext_expr_ls

            # # # # # Randomize expression data (from TRAINING dataet)
            # # # # # if self.randomize_ext_data_:
            # # # # logging.info("\n\n\nWARNING: Randomizing TRAINING dataset expression data\n\n\n")
            # # # # #df_ext[self.expr_key] = [float(x) for x in list(np.random.randint(1, high=100, size=len(df_ext)))]
            # # # # train_expr_ls = list(self.df[self.expr_key])
            # # # # random.shuffle(train_expr_ls)
            # # # # self.df[self.expr_key] = train_expr_ls

            # Add additional column to both df and df_ext to keep track of if external data or original data
            self.df['from_external_test_dataset'] = [False] * len(self.df)
            df_ext['from_external_test_dataset'] = [True] * len(df_ext)

            # append external dataset dataframe to original dataset dataframe
            self.df = pd.concat([self.df, df_ext], axis=0)
            self.df.reset_index(inplace=True, drop=True)
            self.df.sort_values(by='from_external_test_dataset', ascending=True, inplace=True)
            self.df.reset_index(inplace=True, drop=True)
            logging.info('Successfully concatenated input and external datasets\n')


        ########################################################################
        ##                         ~*~  Clean Data ~*~                        ##
        ########################################################################

        logging.info('region_ = '+self.region_)
        # Define key to identify column with sequence data used for model building
        self.flank_seq_working_key = 'seq'

        # NOTE: Since only '16mer_complementary_region' and 'flanking_sequence_1' are checked for NaNs in cleaning use 16mer_complementary_region to get 20mer
        self.df['from_16mer_20mer_targeting_region'] = self.df.apply(lambda x: get_20mer_from_16mer(x['16mer_complementary_region'], x['flanking_sequence_1'], x['20mer_targeting_region'],x['mismatch_16mer_for_flanks']), axis=1)
        # Drop sequences missing from_16mer_20mer_targeting_region
        len_before = len(self.df)
        self.df = self.df[self.df['from_16mer_20mer_targeting_region'].notna()]
        self.df.reset_index(drop=True, inplace=True)
        logging.info("Dropped "+str(len_before - len(self.df)) +" siRNAs for missing 20mer targeting region sequence data  ( "+str(len(self.df))+ " Remaining)")

        # If using flanking sequence Remove sequences missing flanking regions
        if 'flank' in self.region_:  # check if using flanking region
            if self.parameter_to_optimize != 'flank-length':
                self.flank_seq_working_key += '_flank-' + str(self.flank_len_) + 'nts'

            # 1) Drop sequences missing flanking sequences
            len_before = len(self.df)
            self.df = self.df[self.df['flanking_sequence_1'].notna()]
            self.df.reset_index(drop=True, inplace=True)
            logging.info("Dropped "+str(len_before - len(self.df)) +" siRNAs for missing flanking sequence data ( "+str(len(self.df))+ " Remaining)")

            # 2) Drop sequences with mismatch in 16mer (when finding flanks) # TODO: UPDATE THIS TO INCLUDE MISMATCH IN 16MERS?
            len_before = len(self.df)
            self.df = self.df[self.df['mismatch_16mer_for_flanks'] == '16mer perfect match to target']
            self.df.reset_index(inplace=True, drop=True)
            logging.info("Dropped" +str(len_before - len(self.df)) + ' siRNAs for having mismatch 16mer')

            # 3) Drop sequences where could not get longest flanking sequence
            if self.parameter_to_optimize == 'flank-length':
                self.longest_flank_len = max(self.param_values_to_loop_)
            else:
                self.longest_flank_len = self.flank_len_

            len_before = len(self.df)
            indxs_too_short_flanks_ = list(self.df[self.df.apply(lambda x: get_flanking_sequence(x['16mer_complementary_region'], x['flanking_sequence_1'], self.longest_flank_len, True), axis=1).isna()].index)
            self.df.drop(index=indxs_too_short_flanks_, inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            logging.info("Dropped "+str(len_before - len(self.df)) +" siRNAs because could not get longest flanking sequence ( "+str(len(self.df))+ " Remaining)")

            # 4) Take longest flank_len_ from flank_lens_to_loop_ and check there aren't any sequences that aren't long enough to extract longest flanking sequence
            if 'True' in [str(x) for x in list(set(self.df.apply(lambda x: get_flanking_sequence(x['16mer_complementary_region'], x['flanking_sequence_1'], self.longest_flank_len, True), axis=1).isna()))]:
                raise Exception("ERROR: some target sequences' surrounding regions are too short to extract longest flanking region")

            if self.parameter_to_optimize != 'flank-length':
                self.longest_flank_len = self.flank_len_
                if 'target' in self.region_:
                    self.flank_seq_working_key += '_target'
                    logging.info("***********")
                    logging.info(self.df.apply(lambda x: get_flanking_sequence(x['16mer_complementary_region'], x['flanking_sequence_1'], self.flank_len_, True), axis=1))
                    self.df[self.flank_seq_working_key] = self.df.apply(lambda x: get_flanking_sequence(x['16mer_complementary_region'], x['flanking_sequence_1'], self.flank_len_, True), axis=1)
                else:
                    self.flank_seq_working_key += '_NO-target'
                    self.df[self.flank_seq_working_key] = self.df.apply(lambda x: get_flanking_sequence(x['16mer_complementary_region'], x['flanking_sequence_1'], self.flank_len_, False), axis=1)
        else:
            if self.parameter_to_optimize != 'flank-length':
                # NOTE: Since only '16mer_complementary_region' and 'flanking_sequence_1' are checked for NaNs in cleaning use 16mer_complementary_region to get 20mer
                self.flank_seq_working_key = 'from_16mer_20mer_targeting_region' # '20mer_targeting_region'





        # If optimizing Flank length parameter generate sequences with different flank lengths
        if self.parameter_to_optimize == 'flank-length':
            self.flank_seq_working_key = None
            for flank_len_ in self.param_values_to_loop_:  # if parameter_to_optimize == 'flank-len':
                if flank_len_ == 0:
                    # NOTE: Since only '16mer_complementary_region' and 'flanking_sequence_1' are checked for NaNs in cleaning use 16mer_complementary_region to get 20mer
                    flank_seq_working_key__ = 'from_16mer_20mer_targeting_region' #'20mer_targeting_region'

                else:
                    flank_seq_working_key__ = 'seq'
                    flank_seq_working_key__ += '_flank-' + str(flank_len_) + 'nts'
                    logging.info("Flanking sequence size (per side):"+ str(flank_len_)+' nts')
                    if 'target' in self.region_:  # If True include target region in flanking sequence
                        flank_seq_working_key__ += '_target'
                        # Get flanking sequence of desired length along with target region (drop sequences that don't have long enough flanking sequences)
                        # and create new column to store sequences
                        self.df[flank_seq_working_key__] = self.df.apply(lambda x: get_flanking_sequence(x['16mer_complementary_region'], x['flanking_sequence_1'], flank_len_, True), axis=1)
                    else:
                        flank_seq_working_key__ += '_NO-target'
                        # Get flanking sequence of desired length WITHOUT target region (drop sequences that don't have long enough flanking sequences)
                        # and create new column to store sequences
                        self.df[flank_seq_working_key__] = self.df.apply(lambda x: get_flanking_sequence(x['16mer_complementary_region'], x['flanking_sequence_1'], flank_len_, False), axis=1)
                try:
                    self.param_opt_working_keys_ls.append(flank_seq_working_key__)
                except:
                    self.param_opt_working_keys_ls = []
                    self.param_opt_working_keys_ls.append(flank_seq_working_key__)
                #for f_ in self.param_opt_working_keys_ls:
                    #logging.info('\n', f_, ' ', list(self.df[f_].apply(lambda x: len(x)).value_counts().index)[0])

        self.df.sort_values(by=[self.expr_key], inplace=True)
        #self.df.reset_index(drop=True, inplace=True)
        logging.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        logging.info((self.df[['chemical_scaffold', 'screen_type', 'species']].value_counts()))
        if self.remove_undefined_:
            self.df['class'] = self.df[self.expr_key].apply(lambda x: classify(x, self.effco_, self.ineffco_))
        else:
            self.df['class'] = self.df[self.expr_key].apply(lambda x: classify_no_undefined(x, self.effco_, self.ineffco_))
        logging.info(self.df['class'].value_counts())

        # Convert classes into form that can be read by kfld splitter (0's and 1's and -1's for UNLABELLED)
        self.df['numeric_class'] = [int(x.replace('inefficient', '0').replace('efficient', '1').replace('undefined', '-1')) for x in list(self.df['class'])]

        # drop an nans

        self.load_in_unlab_data()
        self.perform_feature_embedding()
        self.split_train_test_paramopt()


    def load_in_unlab_data(self):
        logging.info("\nloading in unlabeled Data..")
        ###############################################################################################################
        ###############################  (if applicable) Load in Unlabeled Data   #####################################
        ###############################################################################################################
        if 'semi-sup' in self.model_type_:
            #self.df_unlab = pd.read_csv(self.input_unlabeled_data_file,  encoding='unicode_escape')
            try:
                self.df_unlab = pd.read_csv(self.input_unlabeled_data_file)
            except:
                self.df_unlab = pd.read_excel(self.input_unlabeled_data_file)

            ###################################################################################
            ##                     ~*~ Select & Clean UNLABELLED Data ~*~                    ##
            ###################################################################################
            self.df_unlab['from_16mer_20mer_targeting_region'] = self.df_unlab.apply(lambda x: get_20mer_from_16mer(x['16mer_complementary_region'], x['flanking_sequence_1'], x['20mer_targeting_region'],x['mismatch_16mer_for_flanks']), axis=1)
            # Drop sequences missing from_16mer_20mer_targeting_region
            len_before = len(self.df_unlab)
            self.df_unlab = self.df_unlab[self.df_unlab['from_16mer_20mer_targeting_region'].notna()]
            self.df_unlab.reset_index(drop=True, inplace=True)
            logging.info("Dropped " +str(len_before - len(self.df_unlab))+ " siRNAs for missing 20mer targeting region sequence data  ( "+str(len(self.df_unlab))+ " Remaining)")


            logging.info('region_ =', self.region_)
            # Define key to identify column with sequence data used for model building
            if 'flank' in self.region_:  # check if using flanking region
                # Remove sequences missing flanking regions (not necessary, undefined data all have flanks)
                # Take longest flank_len_ from flank_lens_to_loop_ and check there aren't any sequences that aren't long enough to extract longest flanking sequence
                logging.info("Longest Flanking sequence size (per side):\n" + str(self.longest_flank_len) + ' nts')
                indxs_to_drop_ = list(self.df_unlab[(self.df_unlab.apply(lambda x: get_flanking_sequence(x['16mer_complementary_region'], x['flanking_sequence_1'], self.longest_flank_len, True), axis=1)).isna()].index)
                logging.info("\n\n\nDropping "+ str(len(indxs_to_drop_))+ ' unlabelled siRNAs because could not determine flanking sequence')
                self.df_unlab.drop(index=indxs_to_drop_, inplace=True)
                self.df_unlab.reset_index(inplace=True, drop=True)
                logging.info("Now have "+str(len(self.df_unlab))+' siRNAs in unlabelled dataset')

            # If optimizing Flank length parameter generate sequences with different flank lengths
            if self.parameter_to_optimize == 'flank-length':
                logging.info("\nGetting flanking sequences for unlabeled data...")
                for flank_len_, flank_seq_working_key in zip(self.param_values_to_loop_, self.param_opt_working_keys_ls):  # if parameter_to_optimize == 'flank-len':
                    logging.info("Flanking sequence size (per side): " + str(flank_len_) + ' nts')
                    # Get flanking sequence of desired length along with target region (drop sequences that don't have long enough flanking sequences) and create new column to store sequences
                    if 'target' in self.region_:
                        self.df_unlab[flank_seq_working_key] = self.df_unlab.apply(lambda x: get_flanking_sequence(x['16mer_complementary_region'], x['flanking_sequence_1'], flank_len_, True), axis=1)
                    else:
                        self.df_unlab[flank_seq_working_key] = self.df_unlab.apply(lambda x: get_flanking_sequence(x['16mer_complementary_region'], x['flanking_sequence_1'], flank_len_, False), axis=1)
                # for f_ in self.param_opt_working_keys_ls:
                #     logging.info('\n', f_, ' ', list(self.df_unlab[f_].apply(lambda x: len(x)).value_counts().index)[0])
                for flank_seq_working_key in self.param_opt_working_keys_ls:
                    ## Drop Unlabelled siRNAs where the working sequence could not be determined
                    len_before = len(self.df_unlab)
                    self.df_unlab.drop(index=self.df_unlab[self.df_unlab[flank_seq_working_key].isna()].index, inplace=True)
                    self.df_unlab.reset_index(inplace=True, drop=True)
                    logging.info('Dropped', len_before - len(self.df_unlab), 'unlabelled siRNAs where the working (i.e. flanking) sequence could not be determined for flank_seq_working_key:', flank_seq_working_key)
            else:
                logging.info("\nGetting flanking sequences for unlabeled data...")
                logging.info("Flanking sequence size (per side): " +str(self.flank_len_) +' nts')
                # Get flanking sequence of desired length along with target region (drop sequences that don't have long enough flanking sequences) and create new column to store sequences
                if 'target' in self.region_:
                    self.df_unlab[self.flank_seq_working_key] = self.df_unlab.apply(lambda x: get_flanking_sequence(x['16mer_complementary_region'], x['flanking_sequence_1'], self.flank_len_, True), axis=1)
                else:
                    self.df_unlab[self.flank_seq_working_key] = self.df_unlab.apply(lambda x: get_flanking_sequence(x['16mer_complementary_region'], x['flanking_sequence_1'], self.flank_len_, False), axis=1)

                
                ## Drop Unlabelled siRNAs where the working sequence could not be determined
                len_before = len(self.df_unlab)
                self.df_unlab.drop(index=self.df_unlab[self.df_unlab[self.flank_seq_working_key].isna()].index, inplace=True)
                self.df_unlab.reset_index(inplace=True, drop=True)
                logging.info('\nDropped', len_before - len(self.df_unlab), 'unlabelled siRNAs where the working (i.e. flanking) sequence could not be determined for flank_seq_working_key:', self.flank_seq_working_key)

            logging.info("Selecting data with species:\n" +str(self.species_ls))
            self.df_unlab = self.df_unlab[self.df_unlab['species'].isin(self.species_ls)]

            # TODO: for Parameter Optimization of unlabelled data set SIZE , add option to alter dataset size - BUT DO THIS CAREFULLY SO DON'T HAVE A LOT OF DATA IN RAM
            logging.info('\n\n\nNumber of unlabelled siRNAs loaded:', len(self.df_unlab))

            # Format Unlabelled to match Labelled Data
            # Add missing columns to self.df_unlab
            self.df_unlab['class'] = ['undefined'] * len(self.df_unlab)
            self.df_unlab['numeric_class'] = [-1] * len(self.df_unlab)
            for c in ['experiment_name', 'expression_percent_normalized_by_max_min', 'expression_percent_normalized_by_z_score', 'standard_deviation_normalized_by_subtracting_mean', 'cleaned_bdna_p2p3p5_human-mouse']:
                self.df_unlab[c] = np.nan

            if self.apply_final_models_to_external_dataset_:
                self.df_unlab['from_external_test_dataset'] = [False] * len(self.df_unlab)

            self.df_unlab = self.df_unlab[list(self.df.columns)] # reorder columns to match labelled self.df
            # Combine unlabelled and labelled data into a single dataframe
            #self.df_before_adding_u = self.df.copy() # make a backup of self.df

            if self.apply_final_models_to_external_dataset_:
                # only include data from initial dataset in indxs_mid_undefined  and indxs_labeled_data
                self.indxs_labeled_data = list(self.df[~self.df['from_external_test_dataset']].index)
                self.indxs_mid_undefined = list(self.df[(self.df['numeric_class'] == -1) & (~self.df['from_external_test_dataset'])].index)
                self.indxs_ext_mid_undefined = list(self.df[(self.df['numeric_class'] == -1) & (self.df['from_external_test_dataset'])].index)

            else:
                self.indxs_labeled_data = list(self.df.index)
                self.indxs_mid_undefined = list(self.df[ self.df['numeric_class'] == -1 ].index)

            self.df = pd.concat(
                [self.df, self.df_unlab],  # NOTE: ORDER HERE MATTERS self.df MUST COME FIRST (code below uses indicies)
                axis=0,
                join="outer",
                ignore_index=True,
                keys=None,
                levels=None,
                names=None,
                verify_integrity=False,
                copy=True,
            )
        else:
            logging.info(str(self.model_type_) +" Does not use unlabeled data")
            if self.apply_final_models_to_external_dataset_:
                # only include data from initial dataset in indxs_mid_undefined  and indxs_labeled_data
                # self.indxs_mid_undefined = list(self.df[(self.df['numeric_class'] == -1) & (~self.df['from_external_test_dataset'])].index)
                self.indxs_labeled_data = list(self.df[~self.df['from_external_test_dataset']].index)
                # self.indxs_ext_mid_undefined = list(self.df[(self.df['numeric_class'] == -1) & (self.df['from_external_test_dataset'])].index)

            else:
                # self.indxs_mid_undefined = list(self.df[self.df['numeric_class'] == -1].index)
                self.indxs_labeled_data = list(self.df.index)



    def perform_feature_embedding(self):
        logging.info("\n\nPerforming feature embedding...")
        ###############################################################################################################
        ###############################          Perform Feature Embedding        #####################################
        ###############################################################################################################

        # TODO: update to work for semisupervised? ~ seem to be having a problem when running semi-sup encoding data

        # Shuffle data so not sorted by efficacy (For troubleshooting)
        shuffled_indx_ls = list(range(len(self.df)))
        random.shuffle(shuffled_indx_ls)
        self.df['temp_index'] = shuffled_indx_ls
        self.df.sort_values(by=['temp_index'],inplace=True)
        self.df.drop(columns=['temp_index'],inplace=True)



        #['one-hot', 'bow-countvect', 'bow-gensim', 'ann-keras', 'ann-word2vec-gensim']


        prnt_kmer_size_ = False
        prnt_flank_seq_working_key_ = False
        prnt_window_size_ = False
        prnt_word_freq_cutoff_ = False

        if self.parameter_to_optimize == 'kmer-size':
            kmer_sizes_ls = self.param_values_to_loop_
            prnt_kmer_size_ = True
        else:
            kmer_sizes_ls = [self.kmer_size_]

        if self.parameter_to_optimize == 'flank-length':
            flank_seq_working_key__ls = self.param_opt_working_keys_ls
            prnt_flank_seq_working_key_ = True
        else:
            flank_seq_working_key__ls = [self.flank_seq_working_key]


        for kmer_ in kmer_sizes_ls:
            if prnt_kmer_size_:
                logging.info('\nkmer size: '+str(kmer_))
            for flank_seq_working_key__ in flank_seq_working_key__ls:
                if prnt_flank_seq_working_key_:
                    logging.info('\nflank_seq_working_key: '+str(flank_seq_working_key__))
                for encoding_ in self.feature_encoding_ls:
                    if (self.parameter_to_optimize == 'window-size'):
                        logging.info('PARAMOPT - encoding: '+encoding_+' '+self.parameter_to_optimize)
                        window_size_ls_ = self.param_values_to_loop_
                        word_freq_cutoff_ls_ = [self.word_freq_cutoff_]
                        prnt_window_size_ = True

                    elif (self.parameter_to_optimize == 'word-frequency-cutoff'):
                        logging.info('PARAMOPT - encoding:'+str(encoding_)+' '+str(self.parameter_to_optimize))
                        window_size_ls_ = [self.window_size_]
                        word_freq_cutoff_ls_ = self.param_values_to_loop_
                        prnt_word_freq_cutoff_ = True

                    else:
                        logging.info('encoding:'+str(encoding_))
                        window_size_ls_ = [self.window_size_]
                        word_freq_cutoff_ls_ = [self.word_freq_cutoff_]

                    for wndwsz_ in window_size_ls_:
                        if prnt_window_size_:
                            logging.info('\nwindow_size_ = '+str(wndwsz_))
                        if wndwsz_ > kmer_:
                            raise Exception("ERROR: window_size ("+str(wndwsz_)+") cannot exceed kmer_size ("+str(kmer_)+") ")
                        for wfco_ in word_freq_cutoff_ls_:
                            if prnt_word_freq_cutoff_:
                                logging.info('\nword_freq_cutoff_ = '+str(wfco_))
                            if encoding_ == 'one-hot': ### One-Hot Encoding ###
                                self.df['one-hot_encoded_' + flank_seq_working_key__ + '_kmer-' + str(kmer_)+ '_windw-'+str(wndwsz_)+'-wfreq-'+str(wfco_)] = one_hot_encode_sequences(
                                    list(self.df[flank_seq_working_key__]))

                            elif encoding_ == 'bow-countvect':### Bag-of-Words Embedding with Sklearn CountVectorizer###
                                self.df['bow-countvect_encoded_' + flank_seq_working_key__ + '_kmer-' + str(kmer_) + '_windw-'+str(wndwsz_)+'-wfreq-'+str(wfco_)] = embed_sequences_with_bow_countvect(
                                    list(self.df[flank_seq_working_key__]), kmer_size_=kmer_, window_size_=wndwsz_,word_freq_cutoff_=wfco_)  # , output_directory = output_directory)



                            elif encoding_ == 'bow-gensim':### Bag-of-Words Embedding with Gensim Doc2bow ###
                                self.df['bow-gensim_encoded_' + flank_seq_working_key__ + '_kmer-' + str(kmer_)+ '_windw-'+str(wndwsz_)+'-wfreq-'+str(wfco_)] = embed_sequences_with_gensim_doc2bow_tfidf(
                                    list(self.df[flank_seq_working_key__]), kmer_size_=kmer_, window_size_=wndwsz_,word_freq_cutoff_=wfco_)  # , output_directory = output_directory)



                            # elif encoding_ == 'bow-gensim-weights-times-values':  ### Bag-of-Words Embedding with Gensim Doc2bow ###
                            #     self.df['bow-gensim-weights-times-values_encoded_' + flank_seq_working_key__ + '_kmer-' + str(kmer_)+ '_windw-'+str(wndwsz_)+'-wfreq-'+str(wfco_)] = embed_sequences_with_gensim_doc2bow_tfidf(
                            #         list(self.df[flank_seq_working_key__]), kmer_size_=kmer_, window_size_=wndwsz_, word_freq_cutoff_=wfco_, vector_output_='weights-times-values',
                            #         # TODO: below for troubleshooting, remove later
                            #         data_expr_pcnts_ = list(self.df[self.expr_key]),
                            #         data_numeric_classes_ =  list(self.df['numeric_class']),
                            #         data_classes_ =  list(self.df['class']),
                            #         data_source_ = list(self.df['from_external_test_dataset']),
                            #         data_oligo_names_ = list(self.df['oligo_name']),
                            #         data_expr_pcnts_norm_ = list(self.df['expression_percent_normalized_by_max_min']),
                            #     )  # , output_directory = output_directory)
                            #
                            #     #raise Exception('Troubleshooting stopping...')


                            elif encoding_ == 'bow-gensim-weights':  ### Bag-of-Words Embedding with Gensim Doc2bow ###
                                self.df['bow-gensim-weights_encoded_' + flank_seq_working_key__ + '_kmer-' + str(kmer_) + '_windw-'+str(wndwsz_)+'-wfreq-'+str(wfco_)]= embed_sequences_with_gensim_doc2bow_tfidf(
                                    list(self.df[flank_seq_working_key__]), kmer_size_=kmer_, window_size_=wndwsz_, word_freq_cutoff_=wfco_, vector_output_='weights',
                                    data_expr_pcnts_=list(self.df[self.expr_key]), data_numeric_classes_=list(self.df['numeric_class']), data_classes_=list(self.df['class']),  # TODO: for troubleshooting, remove later
                                )  # , output_directory = output_directory)

                            # elif encoding_ == 'bow-gensim-values':  ### Bag-of-Words Embedding with Gensim Doc2bow ###
                            #     self.df['bow-gensim-values_encoded_' + flank_seq_working_key__ + '_kmer-' + str(kmer_) + '_windw-'+str(wndwsz_)+'-wfreq-'+str(wfco_)] = embed_sequences_with_gensim_doc2bow_tfidf(
                            #         list(self.df[flank_seq_working_key__]), kmer_size_=kmer_, window_size_=wndwsz_, word_freq_cutoff_=wfco_, vector_output_='values',
                            #         data_expr_pcnts_=list(self.df[self.expr_key]), data_numeric_classes_=list(self.df['numeric_class']), data_classes_=list(self.df['class']),  # TODO: for troubleshooting, remove later
                            #     )  # , output_directory = output_directory)
                            #
                            # elif encoding_ == 'bow-gensim-weights-times-values-adjusted':  ### Bag-of-Words Embedding with Gensim Doc2bow ###
                            #     self.df['bow-gensim-weights-times-values-adjusted_encoded_' + flank_seq_working_key__ + '_kmer-' + str(kmer_) + '_windw-'+str(wndwsz_)+'-wfreq-'+str(wfco_)] = embed_sequences_with_gensim_doc2bow_tfidf(
                            #         list(self.df[flank_seq_working_key__]), kmer_size_=kmer_, window_size_=wndwsz_, word_freq_cutoff_=wfco_, vector_output_='weights-times-values-adjusted')  # , output_directory = output_directory)
                            #
                            # elif encoding_ == 'bow-gensim-values-adjusted':  ### Bag-of-Words Embedding with Gensim Doc2bow ###
                            #     self.df['bow-gensim-values-adjusted_encoded_' + flank_seq_working_key__ + '_kmer-' + str(kmer_) + '_windw-'+str(wndwsz_)+'-wfreq-'+str(wfco_)]= embed_sequences_with_gensim_doc2bow_tfidf(
                            #         list(self.df[flank_seq_working_key__]), kmer_size_=kmer_, window_size_=wndwsz_, word_freq_cutoff_=wfco_, vector_output_='values-adjusted')  # , output_directory = output_directory)



                            elif encoding_ == 'ann-keras': ### Deep Embedding with ANN - Keras ###
                                self.df['ann-keras_encoded_' + flank_seq_working_key__ + '_kmer-' + str(kmer_) + '_windw-'+str(wndwsz_)+'-wfreq-'+str(wfco_)] = embed_sequences_with_keras(
                                    list(self.df[flank_seq_working_key__]), kmer_size_=kmer_, window_size_=wndwsz_,output_dimmension_=self.output_dimmension_)  # , output_directory = output_directory)

                            elif encoding_ == 'ann-word2vec-gensim':### Deep Embedding with Word2Vec - Gensim (TODO: old remove?)  ###
                                self.df['ann-word2vec-gensim_encoded_' + flank_seq_working_key__ + '_kmer-' + str(kmer_) + '_windw-'+str(wndwsz_)+'-wfreq-'+str(wfco_)] = embed_sequences_with_gensim_word2vec(
                                    list(self.df[flank_seq_working_key__]), kmer_size_=kmer_,window_size_=wndwsz_,word_freq_cutoff_=wfco_)  # , output_directory = output_directory)

                            elif encoding_ == 'ann-word2vec-gensim-skipgram':  ### Deep Embedding with Word2Vec - Gensim SkipGram ###
                                self.df['ann-word2vec-gensim-skipgram_encoded_' + flank_seq_working_key__ + '_kmer-' + str(kmer_) + '_windw-' + str(wndwsz_) + '-wfreq-' + str(wfco_)] = embed_sequences_with_gensim_word2vec_skipgram(
                                    list(self.df[flank_seq_working_key__]), kmer_size_=kmer_, window_size_=wndwsz_, word_freq_cutoff_=wfco_,)

                            elif encoding_ == 'ann-word2vec-gensim-cbow':  ### Deep Embedding with Word2Vec - Gensim CBOW ###
                                self.df['ann-word2vec-gensim-cbow_encoded_' + flank_seq_working_key__ + '_kmer-' + str(kmer_) + '_windw-' + str(wndwsz_) + '-wfreq-' + str(wfco_)] = embed_sequences_with_gensim_word2vec_cbow(
                                    list(self.df[flank_seq_working_key__]), kmer_size_=kmer_, window_size_=wndwsz_, word_freq_cutoff_=wfco_,)

                            elif encoding_ == 'ann-fasttext-skipgram':  ### Deep Embedding with Fasttext - SkipGram ###
                                self.df['ann-fasttext-skipgram_encoded_' + flank_seq_working_key__ + '_kmer-' + str(kmer_) + '_windw-' + str(wndwsz_) + '-wfreq-' + str(wfco_)] = embed_sequences_with_fasttext_skipgram(
                                    list(self.df[flank_seq_working_key__]), kmer_size_=kmer_, window_size_=wndwsz_, word_freq_cutoff_=wfco_, )

                            elif encoding_ == 'ann-fasttext-cbow':  ### Deep Embedding with Fasttext - CBOW ###
                                self.df['ann-fasttext-cbow_encoded_' + flank_seq_working_key__ + '_kmer-' + str(kmer_) + '_windw-' + str(wndwsz_) + '-wfreq-' + str(wfco_)] = embed_sequences_with_fasttext_cbow(
                                    list(self.df[flank_seq_working_key__]), kmer_size_=kmer_, window_size_=wndwsz_, word_freq_cutoff_=wfco_, )

                            elif encoding_ == 'ann-fasttext-class-trained':  ### Deep Embedding with Fasttext - CBOW ###
                                self.df['ann-fasttext-class-trained_encoded_' + flank_seq_working_key__ + '_kmer-' + str(kmer_) + '_windw-' + str(wndwsz_) + '-wfreq-' + str(wfco_)] = embed_sequences_with_fasttext_class_trained(
                                    list(self.df[flank_seq_working_key__]), kmer_size_=kmer_, window_size_=wndwsz_, word_freq_cutoff_=wfco_,
                                    data_classes_ = list(self.df['class'])  , indxs_ext_data_ = list(self.df['from_external_test_dataset']) ,
                                )




                            else:
                                raise ValueError('ERROR: encoding '+str(encoding_)+' is not supported')

        ## Print out Encoded Vector Lengths
        prnt_vector_lens_start_ls = []
        prnt_vector_lens_end_ls = []
        for kmer_ in kmer_sizes_ls:
            for flank_seq_working_key__ in flank_seq_working_key__ls:
                for encoding_ in self.feature_encoding_ls:
                    for wndwsz_ in window_size_ls_:
                        for wfco_ in word_freq_cutoff_ls_:
                            v_len_start_ = len(self.df.iloc[0][encoding_ + '_encoded_' + flank_seq_working_key__ + '_kmer-' + str(kmer_) + '_windw-'+str(wndwsz_)+'-wfreq-'+str(wfco_)])
                            v_len_end_ = len(self.df.iloc[-1][encoding_+'_encoded_' + flank_seq_working_key__ + '_kmer-' + str(kmer_) + '_windw-'+str(wndwsz_)+'-wfreq-'+str(wfco_)])
                            if encoding_ == 'one-hot':
                                prnt_vector_lens_start_ls.append(str(v_len_start_) + ' \t')
                                prnt_vector_lens_end_ls.append(str(v_len_end_)+' \t')
                            else:
                                prnt_vector_lens_start_ls.append(str(v_len_start_) + ' \t\t')
                                prnt_vector_lens_end_ls.append(str(v_len_end_) + ' \t\t')
        logging.info('\n\nEncoded vector lengths:\n\n')
        logging.info('\t '.join(encodings_ls))
        logging.info(' '.join(prnt_vector_lens_start_ls))
        logging.info(' '.join(prnt_vector_lens_end_ls))
        # logging.info(len(self.df.iloc[0]['one-hot_encoded_' + self.flank_seq_working_key + '_kmer-' + str(kmer_)]), '\t',
        #       len(self.df.iloc[0]['bow-countvect_encoded_' + self.flank_seq_working_key + '_kmer-' + str(kmer_)]), '\t\t',
        #       len(self.df.iloc[0]['bow-gensim_encoded_' + self.flank_seq_working_key + '_kmer-' + str(kmer_)]), '\t\t',
        #       len(self.df.iloc[0]['ann-keras_encoded_' + self.flank_seq_working_key + '_kmer-' + str(kmer_)]), '\t\t',
        #       len(self.df.iloc[0]['ann-word2vec-gensim_encoded_' + self.flank_seq_working_key + '_kmer-' + str(kmer_)]))
        #
        # logging.info(len(self.df.iloc[-1]['one-hot_encoded_' + self.flank_seq_working_key + '_kmer-' + str(kmer_)]), '\t',
        #       len(self.df.iloc[-1]['bow-countvect_encoded_' + self.flank_seq_working_key + '_kmer-' + str(kmer_)]), '\t\t',
        #       len(self.df.iloc[-1]['bow-gensim_encoded_' + self.flank_seq_working_key + '_kmer-' + str(kmer_)]), '\t\t',
        #       len(self.df.iloc[-1]['ann-keras_encoded_' + self.flank_seq_working_key + '_kmer-' + str(kmer_)]), '\t\t',
        #       len(self.df.iloc[-1]['ann-word2vec-gensim_encoded_' + self.flank_seq_working_key + '_kmer-' + str(kmer_)]))



    def split_train_test_paramopt(self):
        logging.info("\nSplitting Datasets into train/test/paramopt...")
        ##############################################################################################################
        #################################    Split Dataset into Train:Paramopt:Test    ###############################
        ##############################################################################################################
        # Create a unique self.datasplit_id_ for data splitting so can re-run with same data

        # self.datasplit_id_ = 'SUPk'+str(randint(100, 999) ) # ID used to find output data file
        self.datasplit_id_ = model_type_dict[self.model_type_] + self.parameter_to_optimize[0].upper() + str(randint(10000, 99999))  # ID used to find output data file
        # Check that datasplit_id_ doesn't exist already
        while self.datasplit_id_ in [x[0:7] for x in os.listdir(all_output_dir)]:
            # self.datasplit_id_ = 'SUPk'+str(randint(100, 999) ) # ID used to find output data file
            self.datasplit_id_ = model_type_dict[self.model_type_] + self.parameter_to_optimize[0].upper() + str(randint(10000, 99999))  # ID used to find output data file
            #logging.info("self.datasplit_id_ already exists, generating new ID...")
            #logging.info("NEW self.datasplit_id_ | Randomized " + str(len(self.datasplit_id_)) + "-digit ID for this Set of Rounds:\t " + self.datasplit_id_)
        logging.info("self.datasplit_id_ | Randomized " + str(len(self.datasplit_id_)) + "-digit ID for this Set of Rounds:\t " + self.datasplit_id_)
        self.all_data_split_dir = 'data-' + self.datasplit_id_ + '_' + self.abbrev_all_data_label_str_.replace('|', '-') + '/'

        if not os.path.exists(all_output_dir + self.all_data_split_dir):
            os.makedirs(all_output_dir + self.all_data_split_dir)
            logging.info("All datasets will be stored in:\n"+os.getcwd() + all_output_dir + '\n\t' + self.all_data_split_dir)
            if not os.path.exists(all_output_dir + self.all_data_split_dir + 'figures/'):
                os.makedirs(all_output_dir + self.all_data_split_dir + 'figures/')
            if not os.path.exists(all_output_dir + self.all_data_split_dir + 'figures/svg_figs/'):
                os.makedirs(all_output_dir + self.all_data_split_dir + 'figures/svg_figs/')
            if not os.path.exists(all_output_dir + self.all_data_split_dir + 'datasets/'):
                os.makedirs(all_output_dir + self.all_data_split_dir + 'datasets/')

        else:
            raise Exception(
                'ERROR: directory with name ' + self.all_data_split_dir + ' exists. Check that self.datasplit_id_ is being randomized correctly')

        # Append processed dataset parameter information to processed_dataset_parameters_index_file to enable loading in processed data later
        # information to include:
        # directory name for the data splitting: self.all_data_split_dir
        # Information that would make that dataset applicable for a given run's parameters

        data_split_info_dict = {
            'all_data_split_dir':self.all_data_split_dir,
            'num_rerurun_model_building': self.num_rerurun_model_building,  # times to rerun building models using INDEPENDENT 80:10:10 datasets
            'run_round_num': self.run_round_num,  # NOTE: for running additional looping beyond num_rerurun_model_building currently not used
            'screen_type_ls': self.screen_type_ls,  # NOTE: DualGlo will not have flanking regions
            'species_ls': self.species_ls,
            'chemical_scaffold_ls': self.chemical_scaffold_ls,
            'chemical_scaffold_lab': self.chemical_scaffold_lab,
            'normalized_': self.normalized_,
            'effco_': self.effco_,
            'ineffco_': self.ineffco_,
            'remove_undefined_': self.remove_undefined_,
            'region_': self.region_,
            'includes_targ_region_': self.includes_targ_region_,
            'parameter_to_optimize': self.parameter_to_optimize,
            'test_set_size_pcnt_': self.test_set_size_pcnt_,
            'paramopt_set_size_pcnt_': self.paramopt_set_size_pcnt_,
            'split_set_size_pcnt_': self.split_set_size_pcnt_,
            'allowed_classification_prop_deviation_pcnt_': self.allowed_classification_prop_deviation_pcnt_,  # percentage of data allowed to be off for classification proportions
            'kmer_size_': self.kmer_size_,
            'flank_len_': self.flank_len_,  # length on each side (e.g. 50 --> 120nt total length: 50mer 5' flank +20mer target region + 50mer 3' flank)
            'window_size_': self.window_size_,
            'word_freq_cutoff_': self.word_freq_cutoff_,  # Number of times a word must occur in the Bag-of-words Corpus --> when word_freq_cutoff' : self.1 only include words that occur more than once
            'output_dimmension_': self.output_dimmension_,  # output dimmensino of ANN embedding
            'expr_key': self.expr_key,
            'feature_encoding_ls': self.feature_encoding_ls,
            'param_values_to_loop_': self.param_values_to_loop_,

            'apply_final_models_to_external_dataset_' : self.apply_final_models_to_external_dataset_,
            'randomize_ext_data_': self.randomize_ext_data_,
            'external_data_file_':self.external_data_file_,
            'ext_species_ls_': self.ext_species_ls,
            'ext_chemical_scaffold_ls_': self.ext_chemical_scaffold_ls,

            'BREAK_PLACEHOLDER_': '***BREAK***',
            # Below parameters do not need to match with parameters for loading in pre-processed data in the future
            'model_type_': self.model_type_,  # NOTE: cannot load pre-processed data for semi-supervised models
            'date_': self.date_,
            'all_data_label_str_': self.all_data_label_str_,
            'abbrev_all_data_label_str_': self.abbrev_all_data_label_str_,
        }
        row_string_ = ''
        for k in data_split_info_dict.keys():
            if type(data_split_info_dict[k]) == list:
                row_string_ += (str(';'.join([str(x) for x in data_split_info_dict[k]])) + ',')
            else:
                row_string_ += (str(data_split_info_dict[k]) + ',')
        # if processed_dataset_parameters_index_file file does not already exist, make a new one and label the columns
        if not os.path.exists(processed_dataset_parameters_index_file ):
            header_string_ = ','.join(list(data_split_info_dict.keys()))
            with open(processed_dataset_parameters_index_file, 'w') as f:
                f.write(header_string_ + '\n')
            f.close()
            logging.info("Created file for storing data processing parameter data for this and future runs: \n\t"+str(processed_dataset_parameters_index_file))

        # Append run parameter info to processed_dataset_parameters_index_file
        with open(processed_dataset_parameters_index_file,'a') as f:
            f.write(row_string_+'\n')
        f.close()
        logging.info("Data processing parameter data appeneded to: \n\t"+str(processed_dataset_parameters_index_file))

        # Name and Plot Entire Dataset (excluding unlabelled data used for semi-supervised)
        all_data_label = "All siRNA Data"
        if self.plot_starting_data_thresholds_:
            self.plot_thresholds(self.df.iloc[self.indxs_labeled_data], all_data_label, self.all_data_split_dir + 'figures/')



        if self.apply_final_models_to_external_dataset_:
            # Name and Plot External Dataset
            ext_data_label = "External Test siRNA Data"
            if self.plot_starting_data_thresholds_:
                self.plot_thresholds(self.df[self.df['from_external_test_dataset']], ext_data_label, self.all_data_split_dir + 'figures/')



        # Relabel undefined data as nonfunctional if remove_undefined_ = False (so won't be removed in next part)
        if not self.remove_undefined_:
            logging.info('\nNOTE: remove_undefined_ set to ' + str(self.remove_undefined_) + ' so Undefined data will not be excluded from either Training or External datasets!')

            # Exclude external undefined data (if added)

            def reclassify_undefined_numeric(x):
                if x < 0:
                    return 0
                else:
                    return x

            def reclassify_undefined_label(x):
                if x == 'undefined':
                    return 'inefficient'
                else:
                    return x

            self.df['numeric_class'] = self.df['numeric_class'].apply(lambda x: reclassify_undefined_numeric(x))
            self.df['class'] = self.df['class'].apply(lambda x: reclassify_undefined_label(x))




        # Exclude external undefined data (if added)
        if self.apply_final_models_to_external_dataset_:
            self.indxs_mid_undefined = list(self.df[(self.df['numeric_class'] == -1) & (~self.df['from_external_test_dataset'])].index)
            self.mid_undef_df = self.df.iloc[self.indxs_mid_undefined].copy()

            self.df_noundef = self.df[ (self.df['numeric_class'] != -1) & (~self.df['from_external_test_dataset']) ].copy()
            self.df_noundef.reset_index(inplace=True, drop=False)


            # Save external noundefined dataset
            self.ext_df_noundef = self.df[ (self.df['numeric_class'] != -1) & (self.df['from_external_test_dataset']) ].copy()
            self.ext_noundef_df_fnm = all_output_dir + self.all_data_split_dir + 'labeled_data_from_external_test_dataset' + '_partition.csv'
            self.ext_df_noundef.to_csv(self.ext_noundef_df_fnm, index=False)
            logging.info("External Labeled (i.e. no middle/undefined data) Dataset saved to:\n\t"+ str(self.ext_noundef_df_fnm))

            # Save external undefined (middle) dataset
            self.indxs_ext_mid_undefined = list(self.df[(self.df['numeric_class'] == -1) & (self.df['from_external_test_dataset'])].index)
            self.ext_mid_undef_df = self.df.iloc[self.indxs_ext_mid_undefined].copy()
            self.ext_mid_undef_df_fnm = all_output_dir + self.all_data_split_dir + 'undefined_data_from_external_test_dataset' + '_partition.csv'
            self.ext_mid_undef_df.to_csv(self.ext_mid_undef_df_fnm, index=False)
            logging.info("External Undefined Dataset saved to:\n\t"+str(self.ext_mid_undef_df_fnm))

        else:
            self.indxs_mid_undefined = list(self.df[(self.df['numeric_class'] == -1)].index)
            self.mid_undef_df = self.df.iloc[self.indxs_mid_undefined].copy()

            self.df_noundef = self.df[self.df['numeric_class'] != -1].copy()
            self.df_noundef.reset_index(inplace=True, drop=False)



        if self.remove_undefined_: # note: if don't remove undefined data (i.e. remove_undefined_ = False) below code will not work
            # Name and Plot Undefined (excluded) Dataset
            undefined_label = "Undefined Data"
            if self.plot_starting_data_thresholds_:
                self.plot_thresholds(self.mid_undef_df, undefined_label + "\nexcluded from training and evaluation - for now",self.all_data_split_dir + 'figures/')

            # Save Undefined dataset
            self.mid_undef_df_fnm = all_output_dir + self.all_data_split_dir + undefined_label.replace(' ', '_').replace('%','pcnt') + '_partition.csv'
            self.mid_undef_df.to_csv(self.mid_undef_df_fnm, index=False)
            logging.info("Undefined Dataset saved to:\n\t"+str(self.mid_undef_df_fnm))

        # Name and Plot Dataset excluding undefined data
        all_data_label = "After Remove Undef - All siRNA Data"
        if self.plot_starting_data_thresholds_:
            self.plot_thresholds(self.df_noundef, all_data_label, self.all_data_split_dir + 'figures/')

        if self.apply_final_models_to_external_dataset_:
            # Name and Plot External Dataset  excluding undefined data
            ext_data_label = "After Remove Undef - External Test siRNA Data"
            if self.plot_starting_data_thresholds_:
                self.plot_thresholds(self.ext_df_noundef , ext_data_label, self.all_data_split_dir + 'figures/')



        # Create 80:10:10 splits INDEPENDENTLY on the labelled (efficnet/inefficent) siRNA data
        logging.info('Allowed percentage classification proportiond deviation: ' + str(self.allowed_classification_prop_deviation_pcnt_) + '% (' + str( int(np.round(len(self.df_noundef) * (self.allowed_classification_prop_deviation_pcnt_ / 100), 0))) + ' siRNAs)')
        for n_ in range(self.num_rerurun_model_building):  # [0:1]:
            need_to_resplit = True
            resplit_round_ct_ = 0
            while need_to_resplit:
                # logging.info('Round:',n_+1,'/',num_rerurun_model_building)

                ## 1) Create Training set first --> split_initial and train_split_ dataframes for:
                #    80:10:10 --> train (80) : paramopt (10) : train (10)
                #    NOTE: split_initial will not contain unlabelled data


                # Make train Set First
                train_split_df, split_initial_indxs, _, _ = train_test_split(
                    list(self.df_noundef.index),
                    self.df_noundef['numeric_class'],
                    train_size=((100 - (self.test_set_size_pcnt_ + self.paramopt_set_size_pcnt_)) / 100),
                    shuffle=True,
                    random_state=None
                )

                # split_initial_df  holds data that will be further split into Training and Parameter Optimization Sets
                split_initial_df = self.df_noundef.iloc[split_initial_indxs]
                split_initial_df = split_initial_df.copy()
                split_initial_df.reset_index(inplace=True, drop=True)

                # train_split_df holds data that will be used in training
                not_in_index_list = ~self.df_noundef.index.isin(
                    split_initial_indxs)  # train_split_df contains everything NOT in split_initial
                train_split_df = self.df_noundef.iloc[not_in_index_list]
                train_split_df = train_split_df.copy()

                ## 2) Further Split split_initial_df into test_split_df and paramopt_test_split_df
                split_paramopt_indxs, split_test_indxs, _, _ = train_test_split(
                    list(split_initial_df.index),
                    split_initial_df['numeric_class'],
                    test_size=self.test_set_size_pcnt_ / (self.test_set_size_pcnt_ + self.paramopt_set_size_pcnt_),
                    shuffle=True,
                    random_state=None
                )
                # test_split_df contains data used for testing (model evaluation)
                test_split_df = split_initial_df.iloc[split_test_indxs]
                test_split_df = test_split_df.copy()

                # paramop_split_df contains data used for Parameter Optmization
                not_in_index_list = ~split_initial_df.index.isin(
                    split_test_indxs)  # paramop_split_df contains everything NOT in test_split_df
                paramop_split_df = split_initial_df.iloc[not_in_index_list]
                paramop_split_df = paramop_split_df.copy()


                # test_split_df = pd.concat([test_split_df, test_split_df_unlab_only], axis=0)
                # test_split_df.reset_index(inplace=True,drop=True)

                # Check that proportions of Train/Test/Paramopt Datasets are consistent with starting dataset, if not re-split
                starting_dataset_eff_proportion_ = np.round(
                    100 * len(self.df_noundef[self.df_noundef['numeric_class'] == 1]) / len(self.df_noundef), 0)
                starting_dataset_ineff_proportion_ = np.round(
                    100 * len(self.df_noundef[self.df_noundef['numeric_class'] == 0]) / len(self.df_noundef), 0)

                train_dataset_eff_proportion_ = np.round(
                    100 * len(train_split_df[train_split_df['numeric_class'] == 1]) / len(train_split_df), 0)
                train_dataset_ineff_proportion_ = np.round(
                    100 * len(train_split_df[train_split_df['numeric_class'] == 0]) / len(train_split_df), 0)

                test_dataset_eff_proportion_ = np.round(
                    100 * len(test_split_df[test_split_df['numeric_class'] == 1]) / len(test_split_df), 0)
                test_dataset_ineff_proportion_ = np.round(
                    100 * len(test_split_df[test_split_df['numeric_class'] == 0]) / len(test_split_df), 0)

                paramop_dataset_eff_proportion_ = np.round(
                    100 * len(paramop_split_df[paramop_split_df['numeric_class'] == 1]) / len(paramop_split_df), 0)
                paramop_dataset_ineff_proportion_ = np.round(
                    100 * len(paramop_split_df[paramop_split_df['numeric_class'] == 0]) / len(paramop_split_df), 0)

                if (train_dataset_eff_proportion_ > (
                        starting_dataset_eff_proportion_ + self.allowed_classification_prop_deviation_pcnt_)) or (
                        train_dataset_eff_proportion_ < (
                        starting_dataset_eff_proportion_ - self.allowed_classification_prop_deviation_pcnt_)):
                    # logging.info('ERROR: Need to resplit data due to Training set proportions (Round',resplit_round_ct_+1,')')
                    # logging.info('** Dataset Excluding Undefined:',len(self.df_noundef),'siRNAs Total \t(',int(np.round(100*len(self.df_noundef)/len(self.df_noundef),0)),'% )','\n\t0:',self.df_noundef['numeric_class'].value_counts()[0],'siRNAs','\t\t(',int(starting_dataset_ineff_proportion_),'% )',' \n\t1:',self.df_noundef['numeric_class'].value_counts()[1],'siRNAs ','\t\t(',int(starting_dataset_eff_proportion_),'% )\n')
                    # logging.info('** Training:',len(train_split_df),'siRNAs Total \t\t\t(',int(np.round(100*len(train_split_df)/len(self.df_noundef),0)),'% )','\n\t0:',train_split_df['numeric_class'].value_counts()[0],'siRNAs','\t\t(',int(train_dataset_ineff_proportion_),'% )',' \n\t1:',train_split_df['numeric_class'].value_counts()[1],'siRNAs ','\t\t(',int(train_dataset_eff_proportion_),'% )\n')
                    resplit_round_ct_ += 1

                elif (test_dataset_eff_proportion_ > (
                        starting_dataset_eff_proportion_ + self.allowed_classification_prop_deviation_pcnt_)) or (
                        test_dataset_eff_proportion_ < (
                        starting_dataset_eff_proportion_ - self.allowed_classification_prop_deviation_pcnt_)):
                    # logging.info('ERROR: Need to resplit data due to Testing set proportions (Round',resplit_round_ct_+1,')')
                    # logging.info('** Dataset Excluding Undefined:',len(self.df_noundef),'siRNAs Total \t(',int(np.round(100*len(self.df_noundef)/len(self.df_noundef),0)),'% )','\n\t0:',self.df_noundef['numeric_class'].value_counts()[0],'siRNAs','\t\t(',int(starting_dataset_ineff_proportion_),'% )',' \n\t1:',self.df_noundef['numeric_class'].value_counts()[1],'siRNAs ','\t\t(',int(starting_dataset_eff_proportion_),'% )\n')
                    # logging.info('** Testing:',len(test_split_df),'siRNAs Total \t\t\t(',int(np.round(100*len(test_split_df)/len(self.df_noundef),0)),'% )','\n\t0:',test_split_df['numeric_class'].value_counts()[0],'siRNAs','\t\t(',int(test_dataset_ineff_proportion_),'% )',' \n\t1:',test_split_df['numeric_class'].value_counts()[1],'siRNAs ','\t\t(',int(test_dataset_eff_proportion_),'% )\n')
                    resplit_round_ct_ += 1

                elif (paramop_dataset_eff_proportion_ > (
                        starting_dataset_eff_proportion_ + self.allowed_classification_prop_deviation_pcnt_)) or (
                        paramop_dataset_eff_proportion_ < (
                        starting_dataset_eff_proportion_ - self.allowed_classification_prop_deviation_pcnt_)):
                    # logging.info('ERROR: Need to resplit data due to Parameter Optimization set proportions (Round',resplit_round_ct_+1,')')
                    # logging.info('** Dataset Excluding Undefined:',len(self.df_noundef),'siRNAs Total \t(',int(np.round(100*len(self.df_noundef)/len(self.df_noundef),0)),'% )','\n\t0:',self.df_noundef['numeric_class'].value_counts()[0],'siRNAs','\t\t(',int(starting_dataset_ineff_proportion_),'% )',' \n\t1:',self.df_noundef['numeric_class'].value_counts()[1],'siRNAs ','\t\t(',int(starting_dataset_eff_proportion_),'% )\n')
                    # logging.info('** Parameter Optimization::',len(paramop_split_df),'siRNAs Total \t(',int(np.round(100*len(paramop_split_df)/len(self.df_noundef),0)),'% )','\n\t0:',paramop_split_df['numeric_class'].value_counts()[0],'siRNAs','\t\t(',int(paramop_dataset_ineff_proportion_),'% )',' \n\t1:',paramop_split_df['numeric_class'].value_counts()[1],'siRNAs ','\t\t(',int(paramop_dataset_eff_proportion_),'% )\n')
                    resplit_round_ct_ += 1

                else:
                    need_to_resplit = False

                    logging.info('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Round ' + str(n_ + 1) + ' / ' + str(
                        self.num_rerurun_model_building) + ' Complete! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                    logging.info('** Resplit '+str(resplit_round_ct_)+' times')

                    # logging.info('Dataset Excluding Undefined: '+str(len(self.df_noundef))+' siRNAs Total \t( '+
                    #       str(int(np.round(100 * len(self.df_noundef) / len(self.df_noundef), 0)))+ '% )'+'\n\t0:'
                    #       str(self.df_noundef['numeric_class'].value_counts()[0])+' siRNAs '+'\t\t('+
                    #       str(int(starting_dataset_ineff_proportion_)) + '% )'+' \n\t1:'+
                    #       str(self.df_noundef['numeric_class'].value_counts()[1])+' siRNAs '+'\t\t('+
                    #       str(int(starting_dataset_eff_proportion_))+ '% )\n')
                    #
                    # logging.info('Training:', len(train_split_df), 'siRNAs Total \t\t\t(',
                    #       int(np.round(100 * len(train_split_df) / len(self.df_noundef), 0)), '% )', '\n\t0:',
                    #       train_split_df['numeric_class'].value_counts()[0], 'siRNAs', '\t\t(',
                    #       int(train_dataset_ineff_proportion_), '% )', ' \n\t1:',
                    #       train_split_df['numeric_class'].value_counts()[1], 'siRNAs ', '\t\t(',
                    #       int(train_dataset_eff_proportion_), '% )\n')
                    #
                    # logging.info('Testing:', len(test_split_df), 'siRNAs Total \t\t\t(',
                    #       int(np.round(100 * len(test_split_df) / len(self.df_noundef), 0)), '% )', '\n\t0:',
                    #       test_split_df['numeric_class'].value_counts()[0], 'siRNAs', '\t\t(',
                    #       int(test_dataset_ineff_proportion_), '% )', ' \n\t1:',
                    #       test_split_df['numeric_class'].value_counts()[1], 'siRNAs ', '\t\t(',
                    #       int(test_dataset_eff_proportion_), '% )\n')
                    #
                    # logging.info('Parameter Optimization::', len(paramop_split_df), 'siRNAs Total \t(',
                    #       int(np.round(100 * len(paramop_split_df) / len(self.df_noundef), 0)), '% )', '\n\t0:',
                    #       paramop_split_df['numeric_class'].value_counts()[0], 'siRNAs', '\t\t(',
                    #       int(paramop_dataset_ineff_proportion_), '% )', ' \n\t1:',
                    #       paramop_split_df['numeric_class'].value_counts()[1], 'siRNAs ', '\t\t(',
                    #       int(paramop_dataset_eff_proportion_), '% )\n')

                    if (-1 in list(train_split_df['numeric_class'].value_counts().index)):
                        raise Exception("ERROR: train_split_df contains undefined siRNAs when it should not")
                    if (-1 in list(test_split_df['numeric_class'].value_counts().index)):
                        raise Exception("ERROR: test_split_df contains undefined siRNAs when it should not")
                    if (-1 in list(paramop_split_df['numeric_class'].value_counts().index)):
                        raise Exception("ERROR: paramop_split_df contains undefined siRNAs when it should not")

                    ## Name and Plot Data

                    # Name and Plot Training Dataset
                    train_split__label = 'ROUND-' + str(n_ + 1) + " Training Data " + str(
                        100 - (self.test_set_size_pcnt_ + self.paramopt_set_size_pcnt_)) + "%"
                    if self.num_rerurun_model_building < 2:
                        if self.plot_starting_data_thresholds_:
                            self.plot_thresholds(train_split_df, train_split__label + "\n of labeled (not undefined) data",
                                            self.all_data_split_dir + 'figures/')

                    # Name and Plot Testing Dataset
                    testing_data_label = 'ROUND-' + str(n_ + 1) + " Testing Data " + str(self.test_set_size_pcnt_) + "%"
                    if self.num_rerurun_model_building < 2:
                        if self.plot_starting_data_thresholds_:
                            self.plot_thresholds(test_split_df, testing_data_label + "\n of  evaluation (testing) dataset",
                                            self.all_data_split_dir + 'figures/')

                    # Name and Plot Parameter Optimization Dataset
                    paramopt_data_label = 'ROUND-' + str(n_ + 1) + " Parameter Optimization Data " + str(
                        self.paramopt_set_size_pcnt_) + "%"
                    if self.num_rerurun_model_building < 2:
                        if self.plot_starting_data_thresholds_:
                            self.plot_thresholds(paramop_split_df, paramopt_data_label + "\n of  evaluation (testing) dataset",
                                            self.all_data_split_dir + 'figures/')

                    ## Save Datasets

                    # Save  Training Dataset
                    train_split_df_fnm = all_output_dir + self.all_data_split_dir + 'datasets/' + train_split__label.replace(' ',
                                                                                                                             '_').replace(
                        '%', 'pcnt') + '_partition.csv'
                    train_split_df.to_csv(train_split_df_fnm, index=False)
                    logging.info("split_initial Dataset saved to:\n\t"+str(train_split_df_fnm))

                    # Save  Testing Dataset
                    testing_df_fnm = all_output_dir + self.all_data_split_dir + 'datasets/' + testing_data_label.replace(' ',
                                                                                                                         '_').replace(
                        '%', 'pcnt') + '_partition.csv'
                    test_split_df.to_csv(testing_df_fnm, index=False)
                    logging.info("Testing Dataset saved to:\n\t"+str(testing_df_fnm))

                    # Save   Parameter Optimization Dataset
                    paramopt_df_fnm = all_output_dir + self.all_data_split_dir + 'datasets/' + paramopt_data_label.replace(' ',
                                                                                                                           '_').replace(
                        '%', 'pcnt') + '_partition.csv'
                    paramop_split_df.to_csv(paramopt_df_fnm, index=False)
                    logging.info("Parameter Optimization Dataset saved to:\n\t"+str(paramopt_df_fnm))

                    # Name and Plot Partitions
                    partition_label = 'ROUND-' + str(n_ + 1) + '_pie'
                    if self.num_rerurun_model_building < 2:
                        self.plot_proportions_pie(partition_label, self.all_data_split_dir + 'figures/', self.df, self.mid_undef_df, train_split_df, split_initial_df, train_split_df, test_split_df, paramop_split_df, round_ct_=n_, savefig=True)

    #########################################################################################################################################
    ##################################################     Plot All Split Data (Optional)     ###############################################
    #########################################################################################################################################

    def plot_data_splits(self):
        if (self.plot_grid_splits_):
            logging.info("Plotting data splits as a grid...")
            # Plot Data Splitting for each round in a single figure
            if self.num_rerurun_model_building > 2:


                train_col = training_set_plot_color
                test_col = testing_set_plot_color
                paramopt_col = paramopt_set_plot_color


                # sns.palplot([train_col,test_col,paramopt_col])

                # find nearest square for plotting


                rows_cols_compiled_datasplit_fig = math.ceil(math.sqrt(self.num_rerurun_model_building))

                data_fnm_dict = {'Train': 'training_data', 'Paramopt': 'Parameter_Optimization_Data', 'Test': 'Testing_Data'}
                data_pcnt_sz_dict = {'Train': 100 - (self.test_set_size_pcnt_ + self.paramopt_set_size_pcnt_),
                                     'Paramopt': self.paramopt_set_size_pcnt_, 'Test': self.test_set_size_pcnt_}

                figure_label_dict = {
                    'Train': 'data-' + str(self.datasplit_id_) + '\n' + 'ALL-ROUNDS-' + str(
                        self.num_rerurun_model_building) + " Training Data " + str(
                        100 - (
                                self.test_set_size_pcnt_ + self.paramopt_set_size_pcnt_)) + "%" + "\n of labeled (not undefined) data",
                    'Test': 'data-' + str(self.datasplit_id_) + '\n' + 'ALL-ROUNDS-' + str(
                        self.num_rerurun_model_building) + " Testing Data " + str(
                        self.test_set_size_pcnt_) + "%" + "\n of  evaluation (testing) dataset",
                    'Paramopt': 'data-' + str(self.datasplit_id_) + '\n' + 'ALL-ROUNDS-' + str(
                        self.num_rerurun_model_building) + " Parameter Optimization Data " + str(
                        self.paramopt_set_size_pcnt_) + "%" + "\n of  evaluation (testing) dataset",
                }

                title_dict = {
                    'Train': 'Training Datasets',
                    'Test': 'Testing Datasets',
                    'Paramopt': 'Parameter Optimization Datasets'
                }

                pie_wedge_index_dict = {'Train': 0, 'Test': 1, 'Paramopt': 2}

                # Build dictionary of plot coordinates
                plot_coord_dict = {}  # axs[row_,col_]
                max_rows_ = -1
                ct_ = 0
                for row_ in range(rows_cols_compiled_datasplit_fig):  # num rows
                    for col_ in range(rows_cols_compiled_datasplit_fig):  # num cols -1 (to skip last column)
                        if ct_ < self.num_rerurun_model_building:
                            # logging.info(ct_,':',[row_,col_])
                            plot_coord_dict[ct_] = (row_, col_)
                            max_rows_ = row_ + 1
                        ct_ += 1

                output_dir__ = self.all_data_split_dir + 'figures/'

                data_sz_dict = {}

                # Loop through each dataset type for the first round of each round, getting dataset sizes
                for dataset_ in ['Train', 'Paramopt', 'Test']:
                    # Get dataset for given round
                    data_fnm_ = all_output_dir + self.all_data_split_dir + 'datasets/' + 'ROUND-' + str(1) + '_' + data_fnm_dict[
                        dataset_] + '_' + str(data_pcnt_sz_dict[dataset_]) + 'pcnt_partition.csv'
                    self.df_ = pd.read_csv(data_fnm_)
                    data_sz_dict[dataset_] = len(self.df_)

                # Plot a single curve for each embedding type
                for dataset_ in ['Train', 'Paramopt', 'Test']:  # [2:]:
                    figure_label_ = dataset_ + ' ' + figure_label_dict[dataset_]

                    # fig,axs = plt.subplots(rows_cols_compiled_datasplit_fig, rows_cols_compiled_datasplit_fig+1)
                    fig, axs = plt.subplots(max_rows_, rows_cols_compiled_datasplit_fig + 1)

                    fig.set_size_inches(w=13, h=8.5, )  # NOTE: h and w must be large enough to accomodate any legends

                    # Remove last column of axes
                    # gs = axs[0, rows_cols_compiled_datasplit_fig].get_gridspec()
                    gs = axs[0, max_rows_].get_gridspec()
                    # remove the underlying axes
                    for ax in axs[0:, -1]:
                        ax.remove()

                    # Add axis to the right to hold legend
                    # axbig = fig.add_subplot(gs[0:, -1])
                    axlegend = fig.add_subplot(gs[0, -1])

                    # Add axis to the right to hold pie plot
                    axpie = fig.add_subplot(gs[1, -1])

                    # Legend


                    legend_elements = [
                        Patch(facecolor=eff_color, edgecolor=None, label=('< ' + str(self.effco_) + '% : Efficient')),
                        # ('+str(len(self.df_[self.df_['class'] == 'efficient']))+' siRNAs)')),
                        Patch(facecolor=ineff_color, edgecolor=None, label=('≥ ' + str(
                            self.ineffco_) + '% : Inefficient')),
                        # ('+str(len(self.df_[self.df_['class'] == 'inefficient']))+' siRNAs)')),
                        Patch(facecolor=undef_color, edgecolor=None, label=('Undefined')),
                        # ('+str(len(self.df_[self.df_['class'] == 'undefined']))+' siRNAs)')),
                    ]
                    axlegend.legend(handles=legend_elements, loc='upper left', frameon=False, fontsize=12)
                    axlegend.axis('off')

                    # Add Pie Plot to right side
                    data_ = [data_sz_dict['Train'], data_sz_dict['Test'], data_sz_dict['Paramopt']]

                    def piefn(pct, allvals):
                        absolute = int(np.round(pct / 100. * np.sum(allvals)))
                        return f"{pct:.1f}%\n({absolute:d})"

                    wedges, texts, autotexts = axpie.pie(
                        data_,
                        autopct=lambda pct: piefn(pct, data_),
                        pctdistance=0.75,
                        startangle=90,
                        textprops=dict(color="black", fontsize=12),  # , fontweight='bold'),
                        colors=[train_col, test_col, paramopt_col],  # paramopt_col],
                        radius=1.5,
                        wedgeprops=dict(width=0.7),
                    )

                    wedges[pie_wedge_index_dict[dataset_]].set_edgecolor('black')
                    wedges[pie_wedge_index_dict[dataset_]].set_linewidth(3)
                    wedges[pie_wedge_index_dict[dataset_]].set_zorder(99999)

                    pie_legend_elements = [
                        Patch(facecolor=train_col, edgecolor=None, label='Training'),  # ('+str(data_sz_dict['Train'])+')')),
                        Patch(facecolor=test_col, edgecolor=None, label='Testing'),  # ('+str(data_sz_dict['Test'])+')')),
                        Patch(facecolor=paramopt_col, edgecolor=None, label='Param.\n Opt.'),
                        # '\n  Optimization'),# ('+str(data_sz_dict['Paramopt'])+')')),
                    ]
                    pie_legend_elements[pie_wedge_index_dict[dataset_]].set_edgecolor('black')
                    pie_legend_elements[pie_wedge_index_dict[dataset_]].set_linewidth(1)

                    axpie.legend(handles=pie_legend_elements, frameon=False, loc='lower left', handleheight=0.5, handlelength=0.75,
                                 bbox_to_anchor=(0.2, 0.2), fontsize=12)

                    # Remove axes that are not needed (based on number of total parameter optmimization rounds)
                    ct_datasplit_rounds_to_plot_ = 1
                    for row_ in range(len(axs)):  # num rows
                        for col_ in range(len(axs[0]) - 1):  # num cols -1 (to skip last column)
                            # logging.info(row_,col_,ct_datasplit_rounds_to_plot_)
                            if ct_datasplit_rounds_to_plot_ > self.num_rerurun_model_building:
                                # remove axis from figure
                                axs[row_][col_].remove()
                            ct_datasplit_rounds_to_plot_ += 1

                    # Loop through each round, getting data and plotting it
                    for n_ in range(self.num_rerurun_model_building):
                        # Get dataset for given round
                        data_fnm_ = all_output_dir + self.all_data_split_dir + 'datasets/' + 'ROUND-' + str(n_ + 1) + '_' + \
                                    data_fnm_dict[dataset_] + '_' + str(data_pcnt_sz_dict[dataset_]) + 'pcnt_partition.csv'
                        self.df_ = pd.read_csv(data_fnm_)

                        # Plot data for each round
                        colors_ls = [
                            x.replace('inefficient', ineff_color).replace('efficient', eff_color).replace('undefined', undef_color) for
                            x in list(self.df_.sort_values(by=self.expr_key)['class'])]

                        # Get axis to plot on
                        ax = axs[plot_coord_dict[n_]]

                        ax.bar(
                            x=list(range(len(self.df_))),
                            height=self.df_.sort_values(by=self.expr_key)[self.expr_key],
                            color=colors_ls,
                            # width=(1.0),
                        )

                        container2 = ax.errorbar(
                            list(range(len(self.df_))),
                            self.df_.sort_values(by=self.expr_key)[self.expr_key],
                            yerr=self.df_.sort_values(by=self.expr_key)['standard_deviation'],
                            lolims=True,
                            color='black',
                        )

                        connector, (caplines,), (vertical_lines,) = container2.lines
                        connector.set_alpha(0)
                        caplines.set_solid_capstyle('butt')
                        caplines.set_marker(None)
                        vertical_lines.set_linewidth(1.0)  # 0.5)

                        ax.set_ylim(0, max(self.df[self.expr_key]) + 0.2 * max(self.df[self.expr_key]))
                        ax.set_xlim(0, len(self.df_))
                        if n_ == 0:
                            ax.set_ylabel('Target Gene Expression (%)\nNormalized Per Assay (Gene)')

                        ax.set_xlabel('siRNAs (' + str(len(self.df_)) + ' total)')

                        ax.tick_params(axis='x', bottom=False, labelbottom=False)  # remove x-axis ticks and labels

                        # Legend

                        legend_elements = [
                            Patch(facecolor=eff_color, edgecolor=None, label=(str(len(self.df_[self.df_['class'] == 'efficient'])))),
                            # +' siRNAs')),
                            Patch(facecolor=ineff_color, edgecolor=None, label=(str(len(self.df_[self.df_['class'] == 'inefficient'])))),
                            # +' siRNAs')),
                            # Patch(facecolor=undef_color, edgecolor=None, label=('Undefined ('+str(len(self.df_[self.df_['class'] == 'undefined']))+' siRNAs)')),
                        ]
                        ax.legend(handles=legend_elements, loc='upper left', frameon=False, fontsize=12,
                                  handleheight=0.5, handlelength=1.5)

                        ax.set_title('Split Round ' + str(n_ + 1))

                    plt.suptitle(title_dict[dataset_] + '\n' + figure_label_.replace('\n', ' '), fontsize=12, fontweight='bold')

                    fig.tight_layout()  # NOTE: h and w (above in fig.set_size... MUST be large enough to accomodate legends or will be cut off/squished in output)

                    # ** SAVE FIGURE **
                    plt.rcParams['svg.fonttype'] = 'none'  # exports text as strings rather than vector paths (images)
                    fnm_ = (all_output_dir + output_dir__ + figure_label_.split('\n')[0].replace(' ', '_').lower().replace('%',
                                                                                                                           'pcnt') + '_partition')
                    fnm_svg_ = (all_output_dir + output_dir__ + 'svg_figs/' + figure_label_.split('\n')[0].replace(' ',
                                                                                                                   '_').lower().replace(
                        '%', 'pcnt') + '_partition')

                    fig.savefig(fnm_svg_.split('.')[0] + '.svg', format='svg', transparent=True)
                    # logging.info('Figure saved to:',fnm_svg_+'.svg')

                    fig.savefig(fnm_.split('.')[0] + '.png', format='png', dpi=300, transparent=False)
                    logging.info('Figure saved to: '+ fnm_ + '.png')
            else:
                logging.info("Not enough splits performed to plot as a grid \nwith 'num_rerurun_model_building' set to: "+
                      str(self.num_rerurun_model_building))
        else:
            logging.info("plot_grid_splits_ set to :" + str(self.plot_grid_splits_))

    def plot_pie_of_data_splits(self):
        if self.plot_extra_visuals_:
            # Plot Data Splitting Pie
            data_fnm_dict = {'Train': 'training_data', 'Paramopt': 'Parameter_Optimization_Data', 'Test': 'Testing_Data'}
            data_pcnt_sz_dict = {'Train': 100 - (self.test_set_size_pcnt_ + self.paramopt_set_size_pcnt_),
                                 'Paramopt': self.paramopt_set_size_pcnt_, 'Test': self.test_set_size_pcnt_}
            data_sz_dict = {}
            for dataset_ in ['Train', 'Paramopt', 'Test']:
                # Get dataset for given round
                data_fnm_ = all_output_dir + self.all_data_split_dir + 'datasets/' + 'ROUND-' + str(1) + '_' + data_fnm_dict[
                    dataset_] + '_' + str(data_pcnt_sz_dict[dataset_]) + 'pcnt_partition.csv'
                self.df_ = pd.read_csv(data_fnm_)
                data_sz_dict[dataset_] = len(self.df_)


            train_col = training_set_plot_color
            test_col = testing_set_plot_color
            paramopt_col = paramopt_set_plot_color

            # sns.palplot([train_col,test_col,paramopt_col])

            pie_wedge_index_dict = {'Train': 0, 'Test': 1, 'Paramopt': 2}
            fig, axpie = plt.subplots()
            fig.set_size_inches(w=5, h=4, )  # NOTE: h and w must be large enough to accomodate any legends
            # Add Pie Plot to right side
            data_ = [data_sz_dict['Train'], data_sz_dict['Test'], data_sz_dict['Paramopt']]

            def piefn(pct, allvals):
                absolute = int(np.round(pct / 100. * np.sum(allvals)))
                return f"{pct:.1f}%\n({absolute:d})"

            wedges, texts, autotexts = axpie.pie(
                data_,
                autopct=lambda pct: piefn(pct, data_),
                pctdistance=0.75,
                startangle=90,
                textprops=dict(color="black", fontsize=12),  # , fontweight='bold'),
                colors=[train_col, test_col, paramopt_col],  # paramopt_col],
                radius=1.5,
                wedgeprops=dict(width=0.7),
            )


            pie_legend_elements = [
                Patch(facecolor=train_col, edgecolor=None, label='Training'),  # ('+str(data_sz_dict['Train'])+')')),
                Patch(facecolor=test_col, edgecolor=None, label='Testing'),  # ('+str(data_sz_dict['Test'])+')')),
                Patch(facecolor=paramopt_col, edgecolor=None, label='Parameter\nOptimization'),
                # '\n  Optimization'),# ('+str(data_sz_dict['Paramopt'])+')')),
            ]

            axpie.legend(handles=pie_legend_elements, frameon=False, loc='lower left', handleheight=0.5, handlelength=0.75,
                         bbox_to_anchor=(0.25, 0.35), fontsize=12)
            dataset_ = 'Paramopt'  # 'Test'
            wedges[pie_wedge_index_dict[dataset_]].set_edgecolor('black')
            wedges[pie_wedge_index_dict[dataset_]].set_linewidth(3)
            # wedges[pie_wedge_index_dict[dataset_]].set_zorder(99999)
            output_dir__ = self.all_data_split_dir + 'figures/'

            # ** SAVE FIGURE **
            plt.rcParams['svg.fonttype'] = 'none'  # exports text as strings rather than vector paths (images)
            fnm_ = (all_output_dir + output_dir__ + 'pie_plot_' + dataset_)  # all_partitions')
            fnm_svg_ = (all_output_dir + output_dir__ + 'svg_figs/' + 'pie_plot_' + dataset_)  # all_partitions')

            fig.savefig(fnm_svg_.split('.')[0] + '.svg', format='svg', transparent=True)
            fig.savefig(fnm_.split('.')[0] + '.png', format='png', dpi=300, transparent=False)
            logging.info('Figure saved to:'+str(fnm_) + '.png')
        else:
            logging.info("plot_extra_visuals_ set to :" + str(self.plot_extra_visuals_))

    def plot_bar_data_splits(self):
        if self.plot_extra_visuals_:
            logging.info("Plotting data bar splits...")
            if self.num_rerurun_model_building > 2:
                # fig,axs = plt.subplots(6, )

                try:
                    len(self.df_mid_undefined)  # TODO: define this above for semi-supervised data
                except:
                    self.df_mid_undefined = self.df.iloc[list(self.df[self.df['numeric_class'] == -1].index)].copy()
                try:
                    len(self.df_no_unlab)  # TODO: define this above for semi-supervised data
                except:
                    self.df_no_unlab = self.df[self.df['numeric_class'] != -1].copy()

                ineff_col = ineff_color
                eff_col = eff_color
                undef_col = undef_color
                data_fnm_dict = {'Train': 'training_data', 'Paramopt': 'Parameter_Optimization_Data', 'Test': 'Testing_Data'}
                data_pcnt_sz_dict = {'Train': 100 - (self.test_set_size_pcnt_ + self.paramopt_set_size_pcnt_),
                                     'Paramopt': self.paramopt_set_size_pcnt_, 'Test': self.test_set_size_pcnt_}
                fig = plt.figure(layout="constrained")
                # fig.set_size_inches(w=10,h=6.5)
                fig.set_size_inches(w=11.6, h=6.2)
                spec = fig.add_gridspec(nrows=6, ncols=6, hspace=0.35, wspace=0.05)

                ax_top_all = fig.add_subplot(spec[0, :])
                ax_no_undef = fig.add_subplot(spec[1, 0:5])

                # row_ = 2
                # ax_train_label = fig.add_subplot(spec[row_, 0:4])
                # ax_po_label = fig.add_subplot(spec[row_, 4:5])
                # ax_test_label = fig.add_subplot(spec[row_, 5:6])
                # ax_train_label.axis('off')
                # ax_po_label.axis('off')
                # ax_test_label.axis('off')

                row_ = 2
                ax_train_example = fig.add_subplot(spec[row_, 0:4])
                ax_po_example = fig.add_subplot(spec[row_, 4:5])
                ax_test_example = fig.add_subplot(spec[row_, 5:6])

                row_ = row_ + 1
                ax_train_1 = fig.add_subplot(spec[row_, 0:4])
                ax_po_1 = fig.add_subplot(spec[row_, 4:5])
                ax_test_1 = fig.add_subplot(spec[row_, 5:6])

                row_ = row_ + 1
                ax_train_2 = fig.add_subplot(spec[row_, 0:4])
                ax_po_2 = fig.add_subplot(spec[row_, 4:5])
                ax_test_2 = fig.add_subplot(spec[row_, 5:6])

                row_ = row_ + 1
                ax_train_3 = fig.add_subplot(spec[row_, 0:4])
                ax_po_3 = fig.add_subplot(spec[row_, 4:5])
                ax_test_3 = fig.add_subplot(spec[row_, 5:6])

                ax_train_dict_ = {0: ax_train_1, 1: ax_train_2, 2: ax_train_3, self.num_rerurun_model_building - 1: ax_train_3}
                ax_po_dict_ = {0: ax_po_1, 1: ax_po_2, 2: ax_po_3, self.num_rerurun_model_building - 1: ax_po_3}
                ax_test_dict_ = {0: ax_test_1, 1: ax_test_2, 2: ax_test_3, self.num_rerurun_model_building - 1: ax_test_3}

                # 1) Top Bar
                ax_curr_ = ax_top_all
                ax_curr_.barh([0], [len(self.df_no_unlab[self.df_no_unlab['class'] == 'efficient'])], color=eff_col)
                ax_curr_.barh([0], [len(self.df_mid_undefined)], left=[len(self.df_no_unlab[self.df_no_unlab['class'] == 'efficient'])],
                              color=undef_col)
                ax_curr_.barh([0], [len(self.df_no_unlab[self.df_no_unlab['class'] == 'inefficient'])],
                              left=[len(self.df_no_unlab[self.df_no_unlab['class'] == 'efficient']) + len(self.df_mid_undefined)], color=ineff_col)
                ax_curr_.xlims = (0, 1000)
                ax_curr_.axis('off')
                for container in ax_curr_.containers:
                    ax_curr_.bar_label(container, label_type='center', rotation=0, color='black', fontweight='bold', fontsize=12)

                # 2) Excluded Undefined Bar
                ax_curr_ = ax_no_undef
                ax_curr_.barh([0], [len(self.df_no_unlab[self.df_no_unlab['class'] == 'efficient'])], color=eff_col)
                ax_curr_.barh([0], [len(self.df_no_unlab[self.df_no_unlab['class'] == 'inefficient'])],
                              left=[len(self.df_no_unlab[self.df_no_unlab['class'] == 'efficient'])], color=ineff_col)
                ax_curr_.xlims = (0, 1000)
                ax_curr_.axis('off')
                for container in ax_curr_.containers:
                    ax_curr_.bar_label(container, labels=[str(container.datavalues[0]) + '  (' + str(
                        int(np.round(100 * (container.datavalues[0] / len(self.df_no_unlab)), 0))) + '%)'], label_type='center',
                                       rotation=0, color='black', fontweight='bold', fontsize=12)

                # 3) Data splitting Bar Example
                # Labels
                ax_train_example.set_title(
                    'Training\n' + str(int((100 - (
                            self.test_set_size_pcnt_ + self.paramopt_set_size_pcnt_)))) + '% (~' +
                    str((int(len(self.df_no_unlab[self.df_no_unlab['class'] == 'efficient']) * (
                            (100 - (self.test_set_size_pcnt_ + self.paramopt_set_size_pcnt_)) / 100))) + (
                            int(len(self.df_no_unlab[self.df_no_unlab['class'] == 'inefficient']) * (
                                    (100 - (self.test_set_size_pcnt_ + self.paramopt_set_size_pcnt_)) / 100)))) +
                    ' siRNAs)')

                ax_po_example.set_title('Parameter\nOptimization\n' + str(self.paramopt_set_size_pcnt_) + '% (~' +
                                        str((int(len(self.df_no_unlab[self.df_no_unlab['class'] == 'inefficient']) * (
                                                self.paramopt_set_size_pcnt_ / 100))) + (
                                                int(len(self.df_no_unlab[self.df_no_unlab['class'] == 'efficient']) * (
                                                        self.paramopt_set_size_pcnt_ / 100)))) +
                                        ' siRNAs)')

                ax_test_example.set_title('Testing\n' + str(self.test_set_size_pcnt_) + '% (~' +
                                          str((int(len(self.df_no_unlab[self.df_no_unlab['class'] == 'inefficient']) * (
                                                  self.test_set_size_pcnt_ / 100))) + (
                                                  int(len(self.df_no_unlab[self.df_no_unlab['class'] == 'efficient']) * (
                                                          self.test_set_size_pcnt_ / 100)))) +
                                          ' siRNAs)')

                # Bars
                ax_curr_ = ax_train_example
                ax_curr_.barh([0], [int(len(self.df_no_unlab[self.df_no_unlab['class'] == 'efficient']) * (
                        (100 - (self.test_set_size_pcnt_ + self.paramopt_set_size_pcnt_)) / 100))], color=eff_col)
                ax_curr_.barh([0], [int(len(self.df_no_unlab[self.df_no_unlab['class'] == 'inefficient']) * (
                        (100 - (self.test_set_size_pcnt_ + self.paramopt_set_size_pcnt_)) / 100))], left=[
                    int(len(self.df_no_unlab[self.df_no_unlab['class'] == 'efficient']) * (
                            (100 - (self.test_set_size_pcnt_ + self.paramopt_set_size_pcnt_)) / 100))], color=ineff_col)
                ax_curr_.xlims = (0, 1000)
                ax_curr_.axis('off')
                for container in ax_curr_.containers:
                    ax_curr_.bar_label(container, label_type='center', rotation=0, color='black', fontweight='bold', fontsize=12)

                ax_curr_ = ax_po_example
                ax_curr_.barh([0], [int(len(self.df_no_unlab[self.df_no_unlab['class'] == 'efficient']) * (
                        self.paramopt_set_size_pcnt_ / 100))],
                              color=eff_col)
                ax_curr_.barh([0], [int(len(self.df_no_unlab[self.df_no_unlab['class'] == 'inefficient']) * (
                        self.paramopt_set_size_pcnt_ / 100))],
                              left=[int(len(self.df_no_unlab[self.df_no_unlab['class'] == 'efficient']) * (
                                      self.paramopt_set_size_pcnt_ / 100))],
                              color=ineff_col)
                ax_curr_.xlims = (0, 1000)
                ax_curr_.axis('off')
                for container in ax_curr_.containers:
                    ax_curr_.bar_label(container, label_type='center', rotation=0, color='black', fontweight='bold', fontsize=12)

                ax_curr_ = ax_test_example
                ax_curr_.barh([0], [int(len(self.df_no_unlab[self.df_no_unlab['class'] == 'efficient']) * (
                        self.test_set_size_pcnt_ / 100))],
                              color=eff_col)
                ax_curr_.barh([0], [int(len(self.df_no_unlab[self.df_no_unlab['class'] == 'inefficient']) * (
                        self.test_set_size_pcnt_ / 100))],
                              left=[int(len(self.df_no_unlab[self.df_no_unlab['class'] == 'efficient']) * (
                                      self.test_set_size_pcnt_ / 100))],
                              color=ineff_col)
                ax_curr_.xlims = (0, 1000)
                ax_curr_.axis('off')
                for container in ax_curr_.containers:
                    ax_curr_.bar_label(container, label_type='center', rotation=0, color='black', fontweight='bold', fontsize=12)

                # 4) Random Splitting (all rounds)
                # Train
                dataset_ = 'Train'
                for n_ in [0, 1, self.num_rerurun_model_building - 1]:  # range(num_rerurun_model_building)[0:1]:
                    # Get dataset for given round
                    data_fnm_ = all_output_dir + self.all_data_split_dir + 'datasets/' + 'ROUND-' + str(n_ + 1) + '_' + data_fnm_dict[
                        dataset_] + '_' + str(data_pcnt_sz_dict[dataset_]) + 'pcnt_partition.csv'
                    self.df_ = pd.read_csv(data_fnm_)
                    self.df_['class'].value_counts()['efficient']

                    eff_ct_curr_ = self.df_['class'].value_counts()['efficient']
                    ineff_ct_curr_ = self.df_['class'].value_counts()['inefficient']

                    ax_curr_ = ax_train_dict_[n_]  # ax_train_1
                    ax_curr_.barh([0], [eff_ct_curr_], color=eff_col)
                    ax_curr_.barh([0], [ineff_ct_curr_], left=[eff_ct_curr_], color=ineff_col)
                    ax_curr_.xlims = (0, 1000)
                    ax_curr_.axis('off')
                    for container in ax_curr_.containers:
                        ax_curr_.bar_label(container, label_type='center', rotation=0, color='black', fontweight='bold',
                                           fontsize=12)

                # Paramopt
                dataset_ = 'Paramopt'
                for n_ in [0, 1, 2]:  # range(num_rerurun_model_building)[0:1]:
                    # Get dataset for given round
                    data_fnm_ = all_output_dir + self.all_data_split_dir + 'datasets/' + 'ROUND-' + str(n_ + 1) + '_' + data_fnm_dict[
                        dataset_] + '_' + str(data_pcnt_sz_dict[dataset_]) + 'pcnt_partition.csv'
                    self.df_ = pd.read_csv(data_fnm_)
                    self.df_['class'].value_counts()['efficient']

                    eff_ct_curr_ = self.df_['class'].value_counts()['efficient']
                    ineff_ct_curr_ = self.df_['class'].value_counts()['inefficient']

                    ax_curr_ = ax_po_dict_[n_]  # ax_po_1
                    ax_curr_.barh([0], [eff_ct_curr_], color=eff_col)
                    ax_curr_.barh([0], [ineff_ct_curr_], left=[eff_ct_curr_], color=ineff_col)
                    ax_curr_.xlims = (0, 1000)
                    ax_curr_.axis('off')
                    for container in ax_curr_.containers:
                        ax_curr_.bar_label(container, label_type='center', rotation=0, color='black', fontweight='bold',
                                           fontsize=12)

                # Test
                dataset_ = 'Test'
                for n_ in [0, 1, 2]:  # range(num_rerurun_model_building)[0:1]:
                    # Get dataset for given round
                    data_fnm_ = all_output_dir + self.all_data_split_dir + 'datasets/' + 'ROUND-' + str(n_ + 1) + '_' + data_fnm_dict[
                        dataset_] + '_' + str(data_pcnt_sz_dict[dataset_]) + 'pcnt_partition.csv'
                    self.df_ = pd.read_csv(data_fnm_)
                    self.df_['class'].value_counts()['efficient']

                    eff_ct_curr_ = self.df_['class'].value_counts()['efficient']
                    ineff_ct_curr_ = self.df_['class'].value_counts()['inefficient']

                    ax_curr_ = ax_test_dict_[n_]  # ax_test_1
                    ax_curr_.barh([0], [eff_ct_curr_], color=eff_col)
                    ax_curr_.barh([0], [ineff_ct_curr_], left=[eff_ct_curr_], color=ineff_col)
                    ax_curr_.xlims = (0, 1000)
                    ax_curr_.axis('off')

                    for container in ax_curr_.containers:
                        ax_curr_.bar_label(container, label_type='center', rotation=0, color='black', fontweight='bold',
                                           fontsize=12)

                ax_train_1.set_title('      ', rotation=90, fontweight='bold', fontsize=12)#, fontfamily='Times New Roman')
                ax_po_1.set_title('      ', rotation=90, fontweight='bold', fontsize=12)#, fontfamily='Times New Roman')
                ax_test_1.set_title('      ', rotation=90, fontweight='bold', fontsize=12)#, fontfamily='Times New Roman')

                ax_train_3.set_title('  ...', rotation=90, fontweight='bold', fontsize=12)#, fontfamily='Times New Roman')
                ax_po_3.set_title('  ...', rotation=90, fontweight='bold', fontsize=12)#, fontfamily='Times New Roman')
                ax_test_3.set_title('  ...', rotation=90, fontweight='bold', fontsize=12)#, fontfamily='Times New Roman')

                # ** SAVE FIGURE **
                plt.rcParams['svg.fonttype'] = 'none'  # exports text as strings rather than vector paths (images)
                output_dir__ = self.all_data_split_dir + 'figures/'
                fnm_ = (all_output_dir + output_dir__ + '_split_cartoon_')
                fnm_svg_ = fnm_

                fig.savefig(fnm_svg_.split('.')[0] + '.svg', format='svg', transparent=True)
                # logging.info('Figure saved to:',fnm_svg_+'.svg')

                fig.savefig(fnm_.split('.')[0] + '.png', format='png', dpi=300, transparent=True)
                logging.info('Figure saved to:'+str(fnm_) + '.png')

            else:
                logging.info("Not enough splits performed to construct a graphic \n\twith 'num_rerurun_model_building' set to: "+
                      str(self.num_rerurun_model_building))

        else:
            logging.info("plot_extra_visuals_ set to :" + str(self.plot_extra_visuals_))

        #########################################################################################################################################
        ###################################                    Run Model Fitting                     ############################################
        ###################################      (Parameter Optimization & Final Model Building)     ############################################
        #########################################################################################################################################

    #########################################################################################################################################
    ##################################################           Run Model Fittings           ###############################################
    #########################################################################################################################################


    def run_model_fittings(self):
        '''
        Calls methods to build models
         * If self.run_param_optimization_ == True:
            * will run parameter optimiztion AND build final models
        * If self.run_param_optimization_ == False:
            * Will run separate final model building method
        '''

        # Create a unique self.modeltrain_id_
        # self.modeltrain_id_ = 'SUPk'+str(randint(10000, 99999) ) # ID used to find output data filie
        self.modeltrain_id_ = model_type_dict[self.model_type_] + param_id_dict[self.parameter_to_optimize].upper() + str(randint(10000, 99999))  # ID used to find output data file
        # be sure modeltrain_id_ doesn't exist already
        while self.modeltrain_id_ in [x[0:9] for x in os.listdir(all_output_dir)]:
            self.modeltrain_id_ = model_type_dict[self.model_type_] + param_id_dict[self.parameter_to_optimize].upper() + str(randint(10000, 99999))  # ID used to find output data file
            # logging.info("self.modeltrain_id_ already exists, generating new ID...")
            # logging.info("NEW self.modeltrain_id_ | Randomized " + str(len(self.modeltrain_id_)) + "-digit ID for this Set of Rounds:\t " + self.modeltrain_id_)
        logging.info("self.modeltrain_id_ | Randomized " + str(len(self.modeltrain_id_)) + "-digit ID for this Set of Rounds:\t " + self.modeltrain_id_)
        self.output_directory = 'popt-' + str(self.modeltrain_id_) + '_' + model_type_dict[self.model_type_] +'_'+ self.parameter_to_optimize + '_' + str(self.num_rerurun_model_building) + '-rounds_'+self.date_+'/'

        # if not os.path.exists(all_output_dir + self.output_directory):
        #     os.makedirs(all_output_dir + self.output_directory)
        #     logging.info("Output for all " + str(self.run_round_num) + " Rounds stored in:\n" + os.getcwd() + '/'+ all_output_dir + self.output_directory)

        #self.output_directory = 'output_' + model_type_dict[self.model_type_] + '_run_' + str(self.modeltrain_id_) + '_' + self.date_
        #self.output_directory += '/'
        self.output_directory = all_output_dir + self.output_directory# + self.output_directory

        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
            logging.info("Output data stored in:\n" + os.getcwd() + self.output_directory)
            if not os.path.exists(self.output_directory + 'figures/'):
                os.makedirs(self.output_directory + 'figures/')
            if not os.path.exists(self.output_directory + 'figures/svg_figs/'):
                os.makedirs(self.output_directory + 'figures/svg_figs/')
            if not os.path.exists(self.output_directory + 'models/'):
                os.makedirs(self.output_directory + 'models/')
            if not os.path.exists(self.output_directory + 'paramopt_' + self.parameter_to_optimize + '/'):
                os.makedirs(self.output_directory + 'paramopt_' + self.parameter_to_optimize + '/')
            if not os.path.exists(self.output_directory + 'paramopt_' + self.parameter_to_optimize + '/svg_figs/'):
                os.makedirs(self.output_directory + 'paramopt_' + self.parameter_to_optimize + '/svg_figs/')
            if not os.path.exists(self.output_directory + 'data/'):
                os.makedirs(self.output_directory + 'data/')
            if self.apply_final_models_to_external_dataset_:
                if not os.path.exists(self.output_directory + 'figures/ext_data/'):
                    os.makedirs(self.output_directory + 'figures/ext_data/')
                if not os.path.exists(self.output_directory + 'figures/ext_data/svg_figs/'):
                    os.makedirs(self.output_directory + 'figures/ext_data/svg_figs/')
        else:
            raise Exception(
                'ERROR: folder with name "' + self.output_directory.replace(self.all_output_dir,
                                                                            '') + '" already exists in ' + os.getcwd() + self.output_directory +
                '\n - To re-run and build models with same conditions rename existing output folder' + '\n\n' +
                'If continue will OVERWRITE data in this folder')

        ## File info string used so can label figures from parameter optimization with all appropriate IDs even before parameters have been optimized
        ##    NOTE: comment out parameter(s) to be optimized and replace value with 'PARAMOPT':
        self.output_run_file_info_string_ = (
                'data-' + str(self.datasplit_id_) + '_popt-' + str(self.modeltrain_id_) +

                '_' + '-'.join(self.species_ls) +
                '_' + self.chemical_scaffold_lab +
                '_' + '-'.join(self.screen_type_ls) +

                '_' + str(param_norm_label_dict[self.normalized_]) +
                '_effco-' + str(self.effco_) +
                '|ineffco-' + str(self.ineffco_) + '-' + str(remove_undefined_label_dict[self.remove_undefined_]) +
                '_PARAMOPT-'+str(self.parameter_to_optimize).replace('_', '-').replace(' ','-') +

                '_' + self.region_.replace('_', '-') +
                '_flank-len-' + str(self.flank_len_) +
                '_kmer-size-' + str(self.kmer_size_) +
                '_model-'+self.model_type_.replace('_', '-').replace(' ','-') +
                '_' + '-'.join([x.replace('_', '-').replace(' ','-') for x in self.feature_encoding_ls]) +

                '_window-size-' + str(self.window_size_) +
                '_word-freq-cutoff-' + str(self.word_freq_cutoff_) +

                '_n-rounds-' + str(self.num_rerurun_model_building) +
                '_hldout-' + str(self.test_set_size_pcnt_) +

                '_' + self.unlabelled_data_ +
                '_paramopt_pcnt_sz-' + str(self.paramopt_set_size_pcnt_) +
                '_' + self.date_
        )
        ######################################
        # Append model fitting parameter information to model_fitting_parameters_index_file to enable loading in processed data later
        # information to include:
        # directory name for the data splitting: self.all_data_split_dir
        # Information that would make that dataset applicable for a given run's parameters

        model_training_info_dict = {
            'paramopt_dir': self.output_directory.replace(all_output_dir,''),

            'use_existing_processed_dataset_': self.use_existing_processed_dataset_,
            'data_dir': self.all_data_split_dir,

            'num_rerurun_model_building': self.num_rerurun_model_building,  # times to rerun building models using INDEPENDENT 80:10:10 datasets
            'run_round_num': self.run_round_num,

            'test_set_size_pcnt_': self.test_set_size_pcnt_,
            'paramopt_set_size_pcnt_': self.paramopt_set_size_pcnt_,

            'run_param_optimization':self.run_param_optimization_,
            'paramopt': self.parameter_to_optimize,
            'param_values_to_loop_': self.param_values_to_loop_,

            'kmer_size_': self.kmer_size_,
            'flank_len_': self.flank_len_,  # length on each side (e.g. 50 --> 120nt total length: 50mer 5' flank +20mer target region + 50mer 3' flank)
            'window_size_': self.window_size_,
            'word_freq_cutoff_': self.word_freq_cutoff_,  # Number of times a word must occur in the Bag-of-words Corpus --> when word_freq_cutoff' : self.1 only include words that occur more than once
            'ann_output_dimmension_': self.output_dimmension_,  # output dimmensino of ANN embedding
            'feature_encoding_ls': self.feature_encoding_ls,

            'apply_final_models_to_external_dataset_': self.apply_final_models_to_external_dataset_,
            'randomize_external_dataset':self.randomize_ext_data_,
            'external_data_file_':self.external_data_file_,
            'ext_species_ls_':self.ext_species_ls,
            'ext_chemical_scaffold_ls_':self.ext_chemical_scaffold_ls,

            #'BREAK_PLACEHOLDER_': '***BREAK***',
            'model_type_': self.model_type_,
            'date_': self.date_,
            'output_run_file_info_string_': self.output_run_file_info_string_,
            #'abbrev_all_data_label_str_': self.abbrev_all_data_label_str_,
        }
        row_string_ = ''
        for k in model_training_info_dict.keys():
            if type(model_training_info_dict[k]) == list:
                row_string_ += (str(';'.join([str(x) for x in model_training_info_dict[k]])) + ',')
            else:
                row_string_ += (str(model_training_info_dict[k]) + ',')
        # if model_fitting_parameters_index_file file does not already exist, make a new one and label the columns
        if not os.path.exists(model_fitting_parameters_index_file):
            header_string_ = ','.join(list(model_training_info_dict.keys()))
            with open(model_fitting_parameters_index_file, 'w') as f:
                f.write(header_string_ + '\n')
            f.close()
            logging.info("Created file for storing model fitting parameter data for this and future runs: \n\t"+str(model_fitting_parameters_index_file))

        # Append run parameter info to model_fitting_parameters_index_file
        with open(model_fitting_parameters_index_file, 'a') as f:
            f.write(row_string_ + '\n')
        f.close()
        logging.info("Model fitting parameter data appeneded to: \n\t"+(model_fitting_parameters_index_file))

        ######################################

        if self.run_param_optimization_:
            logging.info("Running model fittings for both Parameter Optimization and after Final Model Building..")
            # Perform Parameter Optimization first
            self.parameter_optimization()
            # Build Final Models
            self.build_final_models()
            return

        else:
            logging.info("Running model fittings ONLY for final models (no parameter optimization)")
            self.output_run_file_info_string_ = self.output_run_file_info_string_.split('_PARAMOPT')[0] + '_FINAL-MODELS-ONLY_' + '_'.join(self.output_run_file_info_string_.split('_PARAMOPT')[-1].split('_')[1:])
            logging.info('output_run_file_info_string = '+self.output_run_file_info_string_)
            # Build Final Models
            self.build_final_models()
            return




    def parameter_optimization(self):
        logging.info("Running parameter optimization...")


        self.top_param_val_per_round_dict = {}
        self.paramop_performance_metrics_encodings_dict = {}
        self.paramop_performance_curves_encodings_dict = {}
        self.paramop_models_encodings_dict = {}
        self.paramop_key_ls = []
        for n_ in range(self.num_rerurun_model_building):
            logging.info('Building Parameter Optimization ' + str(self.model_type_) + ' models for Round: '+str( n_ + 1)+ ' / '+str(self.num_rerurun_model_building))
            print('Building Parameter Optimization ' + str(self.model_type_) + ' models for Round: '+str( n_ + 1)+ ' / '+str(self.num_rerurun_model_building))

            train_data_fnm_ = all_output_dir + self.all_data_split_dir + 'datasets/' + 'ROUND-' + str(n_ + 1) + '_Training_Data_' + str(100 - (self.test_set_size_pcnt_ + self.paramopt_set_size_pcnt_)) + 'pcnt_partition.csv'
            paramop_data_fnm_ = all_output_dir + self.all_data_split_dir + 'datasets/' + 'ROUND-' + str(n_ + 1) + '_Parameter_Optimization_Data_' + str(self.paramopt_set_size_pcnt_) + 'pcnt_partition.csv'
            # Load in Training and Parameter Optimization Datasets
            logging.info("Loading Training data...")
            logging.info("1) all_output_dir:\n"+all_output_dir+"\n")
            logging.info("2) self.all_data_split_dir:\n"+self.all_data_split_dir+"\n")
            logging.info("3) train_data_fnm_:\n"+train_data_fnm_+"\n")
            self.df_train = pd.read_csv(train_data_fnm_)
            logging.info("Training data loaded successfully")
            logging.info("Loading Parameter optimization data...")
            self.df_paramopt = pd.read_csv(paramop_data_fnm_)
            logging.info("Parameter optimization data loaded successfully")

            ## Loop through Parameter Optimization
            # ['kmer-size', 'flank-length', 'model', 'window-size', 'word-frequency-cutoff', 'ANN_output_dimmension',
            # 'unlabeled_data_type', 'unlabeled_data_size', 'feature_encoding', None]

            if self.parameter_to_optimize == 'kmer-size':
                kmer_sizes_ls = self.param_values_to_loop_
            else:
                kmer_sizes_ls = [self.kmer_size_]

            if self.parameter_to_optimize == 'flank-length':
                flank_seq_working_key__ls = self.param_opt_working_keys_ls
            else:
                flank_seq_working_key__ls = [self.flank_seq_working_key]

            if self.parameter_to_optimize == 'model':
                model_type_ls = self.param_values_to_loop_
            else:
                model_type_ls = [self.model_type_]

            if self.parameter_to_optimize == 'window-size':
                window_size_ls_ = self.param_values_to_loop_
            else:
                window_size_ls_ = [self.window_size_]

            if self.parameter_to_optimize == 'word-frequency-cutoff':
                word_freq_cutoff_ls = self.param_values_to_loop_
            else:
                word_freq_cutoff_ls = [self.word_freq_cutoff_]

            param_ = 'X'
            for kmer_ in kmer_sizes_ls:
                for flank_seq_working_key__ in flank_seq_working_key__ls:
                    if flank_seq_working_key__ == 'from_16mer_20mer_targeting_region':#'20mer_targeting_region':
                        flank_len__ = 0
                    else:
                        flank_len__ = flank_seq_working_key__.split('-')[-1].split('nts')[0]
                    for e in self.feature_encoding_ls: # ['one-hot', 'bow-countvect', 'bow-gensim', 'ann-keras', 'ann-word2vec-gensim']
                        logging.info('\tLoop Embedding: '+str(e)) # TODO: delete this line
                        for wndwsz_ in window_size_ls_:
                            for wfco_ in word_freq_cutoff_ls:
                                for m_ in model_type_ls:
                                    # Train Parameter Optimization Models
                                    if self.parameter_to_optimize == 'kmer-size':
                                        logging.info('  ** Parameter Optimization -- '+str(self.parameter_to_optimize)+' : '+str(kmer_)+ ' **  ')
                                        param_ = kmer_
                                    if self.parameter_to_optimize == 'flank-length':
                                        logging.info('  ** Parameter Optimization -- ' + str(self.parameter_to_optimize) + ' : '+str(flank_len__)+ ' **  ')
                                        param_ = flank_len__
                                    if self.parameter_to_optimize == 'feature_encoding':
                                        logging.info('  ** Parameter Optimization -- ' + str(self.parameter_to_optimize) + ' : '+str(e)+ ' **  ')
                                        param_ = e
                                    if self.parameter_to_optimize == 'model':
                                        logging.info('  ** Parameter Optimization -- ' + str(self.parameter_to_optimize) + ' : '+str(m_)+ ' **  ')
                                        param_ = m_
                                    if self.parameter_to_optimize == 'window-size':
                                        logging.info('  ** Parameter Optimization -- ' + str(self.parameter_to_optimize) + ' : '+str(wndwsz_)+ ' **  ')
                                        param_ = wndwsz_
                                    if self.parameter_to_optimize == 'word-frequency-cutoff':
                                        logging.info('  ** Parameter Optimization -- ' + str(self.parameter_to_optimize) + ' : '+str(wfco_)+ ' **  ')
                                        param_ = wfco_

                                    if m_ == 'PARAMOPT':
                                        clf_po = model_dict['random-forest']
                                    else:
                                        clf_po = model_dict[m_]

                                    # if  e == 'bow-gensim':
                                    # #if e == 'one-hot' or e == 'bow-gensim':
                                    #     X_train_ = [[float(y) for y in x.replace('[', '').replace(']', '').replace('\n', '').split(' ') if y != ''] for x in self.df_train[e + '_encoded_' + flank_seq_working_key__ + '_kmer-' + str(kmer_)]]
                                    # else:


                                    X_train_ = [[float(y) for y in x.replace('[', '').replace(']', '').replace(' ', '').split(',')] for x in self.df_train[e + '_encoded_' + flank_seq_working_key__ + '_kmer-' + str(kmer_)  + '_windw-'+str(wndwsz_)+'-wfreq-'+str(wfco_) ]]
                                    Y_train_ = list(self.df_train['numeric_class'])

                                    if 'semi-sup' in self.model_type_:
                                        # Add undefined data
                                        X_train_u_ = [list(x) for x in list(self.df.loc[(self.indxs_labeled_data[-1]+1):][e + '_encoded_' + flank_seq_working_key__ + '_kmer-' + str(kmer_) + '_windw-'+str(wndwsz_)+'-wfreq-'+str(wfco_) ])]
                                        Y_train_u_ = list(self.df.loc[(self.indxs_labeled_data[-1]+1):]['numeric_class'])
                                        X_train_ = X_train_u_ + X_train_
                                        Y_train_ = np.array(Y_train_u_ + Y_train_)

                                    # if e == 'bow-gensim':
                                    # # if e == 'one-hot' or e == 'bow-gensim':
                                    #     X_paramopt_ = [[float(y) for y in x.replace('[', '').replace(']', '').replace('\n', '').split(' ') if y != ''] for x in self.df_paramopt[e + '_encoded_' + flank_seq_working_key__ + '_kmer-' + str(kmer_)]]
                                    # else:
                                    X_paramopt_ = [[float(y) for y in x.replace('[', '').replace(']', '').replace(' ', '').split(',')] for x in self.df_paramopt[e + '_encoded_' + flank_seq_working_key__ + '_kmer-' + str(kmer_) + '_windw-'+str(wndwsz_)+'-wfreq-'+str(wfco_) ]]

                                    Y_paramopt_ = np.array(self.df_paramopt['numeric_class'])
                                    logging.info("Training paramopt model...")
                                    clf_po.fit(X_train_, Y_train_)
                                    logging.info("Predicting with paramopt model...")
                                    preds_po = clf_po.predict_proba(X_paramopt_)[:, 1]
                                    preds_binary_po = clf_po.predict(X_paramopt_)



                                    # Pickle parameter optimization models (per round, per embedding)
                                    # logging.info("Pickling model " + str(n_ + 1) + ' / ' + str(self.num_rerurun_model_building) + '...')
                                    fnm_ = self.output_directory + 'models/' + 'paramopt_' + model_type_dict[m_] + '_model_rnd-' + str(n_ + 1) + '_' + feature_encodings_dict[e] + '.pickle'
                                    with open(fnm_, 'wb') as pickle_file:
                                        pickle.dump(clf_po, pickle_file)
                                    pickle_file.close()
                                    logging.info('\n\n\nParamopt model ' + str(n_ + 1) + ' with encoding ' + str(e) + ' saved to: '+fnm_.replace(self.output_directory, '~/\n\n'))

                                    ## Evaluate Parameter Optimization Model Performance
                                    logging.info("Evaluating model performance of paramopt model...")
                                    logging.info("\tModel type: "+str(m_))
                                    logging.info('\tEmbedding: '+e)
                                    from sklearn.metrics import precision_recall_curve
                                    #logging.info('\n\n\npreds_po :',set(preds_po),'\n\n\n')

                                    ## In cases where is LabelSpreading or LabelPropagation model and One-Hot Encoding,
                                    #     will have an NaN when trying to compute precision-recall
                                    # TODO: fix for final model p-r curve making too
                                    if not (((m_ == 'semi-sup-label-propagation') or (m_ == 'semi-sup-label-spreading')) and (e == 'one-hot')):
                                        p_po_, r_po_, ts_po_ = precision_recall_curve(Y_paramopt_, preds_po)
                                        aucpr_po_ = metrics.auc(r_po_, p_po_)

                                    from sklearn.metrics import f1_score
                                    fscore_po_ = f1_score(Y_paramopt_, preds_binary_po)  # , average=None)

                                    from sklearn.metrics import fbeta_score
                                    fbetascore_po_ = fbeta_score(Y_paramopt_, preds_binary_po, beta=self.f_beta_)  # , average=None)
                                    logging.info("Computing fbeta_score with beta = "+str(self.f_beta_))

                                    from sklearn.metrics import accuracy_score
                                    accuracy_po_ = accuracy_score(Y_paramopt_, preds_binary_po)

                                    from sklearn.metrics import matthews_corrcoef
                                    mcc_po_ = matthews_corrcoef(Y_paramopt_, preds_binary_po)

                                    ## In cases where is LabelSpreading or LabelPropagation model and One-Hot Encoding,
                                    #     will have an NaN when trying to compute precision-recall
                                    # TODO: fix for final model p-r curve making too
                                    if not(((m_ == 'semi-sup-label-propagation' ) or (m_ == 'semi-sup-label-spreading')) and (e == 'one-hot')) :
                                        p_po_, r_po_, ts_po_ = precision_recall_curve(Y_paramopt_, preds_po)
                                        aucpr_po_ = metrics.auc(r_po_, p_po_)

                                        # Compute Unacheiveable Region
                                        class_dist_po = p_po_[0]  # class distribution (Pr=1)
                                        unach_recalls_po = list(r_po_)[::-1]
                                        # compute y (precision) values from the x (recall) values
                                        unach_precs_po = []
                                        for x in unach_recalls_po:
                                            y = (class_dist_po * x) / ((1 - class_dist_po) + (class_dist_po * x))
                                            unach_precs_po.append(y)
                                        unach_p_r_po_ = [unach_precs_po, unach_recalls_po]
                                        auc_unach_adj_po_ = metrics.auc(r_po_, p_po_) - metrics.auc(unach_recalls_po, unach_precs_po)

                                        paramop_performance_curves_dict_ = {
                                            'Precision_Recall_Curve': [p_po_, r_po_, ts_po_],
                                            'Unacheivable_Region_Curve': [unach_precs_po, unach_recalls_po],
                                        }
                                        aucpr_po_adj_ = aucpr_po_ - p_po_[0]


                                    else:
                                        logging.info("WARNING: because model-type is "+str(m_)+' and encoding type is '+str(e)+' Precision-Recall curves cannot be created for this model')
                                        paramop_performance_curves_dict_ = {
                                            'Precision_Recall_Curve': [[], [], []],
                                            'Unacheivable_Region_Curve': [[], []],
                                        }
                                        aucpr_po_ = 0
                                        aucpr_po_adj_ = 0
                                        auc_unach_adj_po_ = 0

                                    paramop_performance_metrics_dict_ = {
                                        'AUCPR': aucpr_po_,
                                        'AUCPR-adj': aucpr_po_adj_,
                                        'AUCPR-unach-adj': auc_unach_adj_po_,
                                        'F-Score': fscore_po_,
                                        'Fbeta-Score':fbetascore_po_,
                                        'Accuracy': accuracy_po_,
                                        'MCC': mcc_po_,
                                    }
                                    key__ = str(e) + '-' + self.parameter_to_optimize + '-' + str(param_)+'_round_' + str(n_) #str(kmer_) +'-'+ str(flank_len__)+'-' + m_ +'_' + str(n_)
                                    self.paramop_models_encodings_dict[key__] = clf_po
                                    self.paramop_performance_metrics_encodings_dict[ key__] = paramop_performance_metrics_dict_
                                    self.paramop_performance_curves_encodings_dict[  key__] = paramop_performance_curves_dict_
                                    self.paramop_key_ls.append( key__)



            ### Select top parameter value (per round)
            metric_used_to_id_best_po_ = self.metric_used_to_id_best_po
            max_score_po_ = max(pd.DataFrame(self.paramop_performance_metrics_encodings_dict).loc[metric_used_to_id_best_po_])
            best_po_id_ = pd.DataFrame(pd.DataFrame(self.paramop_performance_metrics_encodings_dict).loc[metric_used_to_id_best_po_])[
                pd.DataFrame(pd.DataFrame(self.paramop_performance_metrics_encodings_dict).loc[metric_used_to_id_best_po_])[
                    metric_used_to_id_best_po_] == max_score_po_].index[0]
            try:
                param_val_ = int(best_po_id_.split((self.parameter_to_optimize) + '-')[-1].split('_')[0])
            except:
                param_val_ = str(best_po_id_.split((self.parameter_to_optimize) + '-')[-1].split('_')[0])
            logging.info(
                "Identifying Best Parameter Value in Parameter Optimization:" + '\n\t Metric: ' + metric_used_to_id_best_po_ + '\n\t Score: ' +
                str(max_score_po_) + '\n\t Parameter ID: ' + best_po_id_ + '\n\t ' + str(self.parameter_to_optimize) + ': ' + str(
                    param_val_))

            self.output_run_file_info_string_.replace('PARAMOPT', str(param_val_))

            logging.info('\n\n\n\n\n' +
                  '************************************************************************************\n'
                  '****\t\t\t\t\t\t\t\t\t\t****\n' +
                  '****\t\t\t\t' +

                  'For Round: '+str(n_)+'\n'+
                  '  Selected '+str(self.parameter_to_optimize)+': ' + str(param_val_) +

                  '\t\t\t\t****\n' +
                  '****\t\t\t\t\t\t\t\t\t\t****\n' +
                  '************************************************************************************\n' +
                  '\n\n\n\n\n')

            self.top_param_val_per_round_dict[n_] = param_val_
            # TODO: (maybe don't need to?) Update value for the parameter optimized
            # if self.parameter_to_optimize == 'kmer-size':
            #     self.kmer_size_ = param_val_
            # if self.parameter_to_optimize == 'flank-length':
            #     self.flank_len_ = param_val_
            #     self.flank_seq_working_key = 'seq_flank-'+str(param_val_)+'nts_target'
            # if self.parameter_to_optimize == 'feature_encoding':
            #     self.feature_encoding_ls = [param_val_]
            # if self.parameter_to_optimize == 'model':
            #     self.model_type_ = param_val_

        # Export top parameters for each round of parameter optimization
        fnm_ = self.output_directory + 'data/' + 'best_param_per_' + str(self.num_rerurun_model_building) + '-rnds_paramop.csv'
        with open(fnm_,'w+') as f:
            f.write('round, '+str(self.parameter_to_optimize) + ',\n')
            for n_ in range(self.num_rerurun_model_building):
                f.write(str(n_)+', '+str(self.top_param_val_per_round_dict[n_])+',\n')
        f.close()
        logging.info('Top parameters per round of paramopt model building saved to:\n\t'+fnm_)



    def build_final_models(self):
        logging.info("Building Final Models...")
        self.final_models_encodings_dict = {}
        self.final_performance_metrics_encodings_dict = {}
        self.final_detailed_performance_metrics_encodings_dict = {}
        self.final_performance_curves_encodings_dict = {}

        # add randomized background
        if self.include_random_background_comparison_:
            self.randombackground_final_models_encodings_dict = {}
            self.randombackground_final_performance_metrics_encodings_dict = {}
            self.randombackground_final_detailed_performance_metrics_encodings_dict = {}
            self.randombackground_final_performance_curves_encodings_dict = {}

        if self.apply_final_models_to_external_dataset_:
            self.ext_final_performance_metrics_encodings_dict = {}
            self.ext_final_detailed_performance_metrics_encodings_dict = {}
            self.ext_final_performance_curves_encodings_dict = {}


        self.final_key_ls = []
        self.final_model_params_ls = []

        #self.flank_seq_working_key = 'seq_flank-20nts_target' # TODO: delete line (used only for testing)
        #self.all_data_split_dir = 'data-TEST-rfF26194_h_p3_bDNA_oh-bowcv_norm_25-60-rm-u/' # TODO: delete line (used only for testing)

        all_preds_dict = {} # For exporting predictions to .csv file
        if self.include_random_background_comparison_:
            all_preds_dict_randombackground = {} # For exporting predictions to .csv file

        if self.apply_final_models_to_external_dataset_:
            all_preds_dict_ext = {} # For exporting predictions to .csv file

        # TODO: flank-seq-working key might not work for cases where don't have targeting region or just have target region alone
        for n_ in range(self.num_rerurun_model_building):
            if not self.run_param_optimization_:
                param_val_ = 'noParamVal'
            else:
                param_val_ = self.top_param_val_per_round_dict[n_]

            if self.parameter_to_optimize == 'kmer-size':
                kmer_size___ = param_val_
            else:
                kmer_size___ = self.kmer_size_

            if self.parameter_to_optimize == 'flank-length':
                if param_val_ == 0:
                    flank_seq_working_key___ = 'from_16mer_20mer_targeting_region'#'20mer_targeting_region'
                else:
                    flank_seq_working_key___ = 'seq_flank-'+str(param_val_)+'nts_target'
            else:
                flank_seq_working_key___ = self.flank_seq_working_key

            if self.parameter_to_optimize == 'model':
                model_type___ = param_val_
            else:
                model_type___ = self.model_type_

            if self.parameter_to_optimize == 'window-size':
                window_size___ = param_val_
            else:
                window_size___ = self.window_size_

            if self.parameter_to_optimize == 'word-frequency-cutoff':
                word_freq_cutoff___ = param_val_
            else:
                word_freq_cutoff___ = self.word_freq_cutoff_


            logging.info('Building Final ' + str(model_type___) + ' models for Round: '+str(n_ + 1) +' / ' +str(self.num_rerurun_model_building))
            print('Building Final ' + str(model_type___) + ' models for Round: ' + str(n_ + 1) + ' / ' + str(self.num_rerurun_model_building))
            train_data_fnm_ = all_output_dir + self.all_data_split_dir + 'datasets/' + 'ROUND-' + str(n_ + 1) + '_Training_Data_' + str(100 - (self.test_set_size_pcnt_ + self.paramopt_set_size_pcnt_)) + 'pcnt_partition.csv'
            test_data_fnm_ = all_output_dir + self.all_data_split_dir + 'datasets/' + 'ROUND-' + str(n_ + 1) + '_Testing_Data_' + str(self.test_set_size_pcnt_) + 'pcnt_partition.csv'

            # Load in Training and Testing Datasets
            self.df_train = pd.read_csv(train_data_fnm_)
            self.df_test = pd.read_csv(test_data_fnm_)
            if self.apply_final_models_to_external_dataset_:
                ext_data_fnm_ = self.ext_noundef_df_fnm
                logging.info("Including additional evaluation on external dataset: "+ ext_data_fnm_)
                # TODO: include undefined data in external test dataset evaluation
                df_ext = pd.read_csv(ext_data_fnm_)


            for e in self.feature_encoding_ls:
                # Train Final  Models
                clf_final = model_dict[model_type___]

                # if e == 'bow-gensim':
                # # if e == 'one-hot' or e == 'bow-gensim':
                #     X_train_ = [[float(y) for y in x.replace('[', '').replace(']', '').replace('\n', '').split(' ') if y != ''] for x in self.df_train[e + '_encoded_' + flank_seq_working_key___ + '_kmer-' + str(kmer_size___ )]]
                # else:
                X_train_ = [[float(y) for y in x.replace('[', '').replace(']', '').replace(' ', '').split(',')] for x in self.df_train[e + '_encoded_' + flank_seq_working_key___ + '_kmer-' + str(kmer_size___ )  + '_windw-'+str(window_size___)+'-wfreq-'+str(word_freq_cutoff___) ]]
                Y_train_ = list(self.df_train['numeric_class'])

                if 'semi-sup' in model_type___:
                    # Add undefined data
                    X_train_u_ = [list(x) for x in list(self.df.loc[(self.indxs_labeled_data[-1]+1):][e + '_encoded_' + flank_seq_working_key___ + '_kmer-' + str(kmer_size___ ) + '_windw-'+str(window_size___)+'-wfreq-'+str(word_freq_cutoff___)  ])]
                    Y_train_u_ = list(self.df.loc[(self.indxs_labeled_data[-1]+1):]['numeric_class'])
                    X_train_ = X_train_u_ + X_train_
                    Y_train_ = np.array(Y_train_u_ + Y_train_)

                # if e == 'bow-gensim':
                # # if e == 'one-hot' or e == 'bow-gensim':
                #     X_test_ = [[float(y) for y in x.replace('[', '').replace(']', '').replace('\n', '').split(' ') if y != ''] for x in self.df_test[e + '_encoded_' + flank_seq_working_key___ + '_kmer-' + str(kmer_size___ )]]
                # else:
                X_test_ = [[float(y) for y in x.replace('[', '').replace(']', '').replace(' ', '').split(',')] for x in self.df_test[e + '_encoded_' + flank_seq_working_key___ + '_kmer-' + str(kmer_size___ )  + '_windw-'+str(window_size___)+'-wfreq-'+str(word_freq_cutoff___)  ]]

                Y_test_ = np.array(self.df_test['numeric_class'])

                if self.apply_final_models_to_external_dataset_:
                    # if e == 'bow-gensim':
                    # # if e == 'one-hot' or e == 'bow-gensim':
                    #     X_ext_ = [[float(y) for y in x.replace('[', '').replace(']', '').replace('\n', '').split(' ') if y != ''] for x in df_ext[e + '_encoded_' + flank_seq_working_key___ + '_kmer-' + str(kmer_size___)]]
                    # else:
                    X_ext_ = [[float(y) for y in x.replace('[', '').replace(']', '').replace(' ', '').split(',')] for x in df_ext[e + '_encoded_' + flank_seq_working_key___ + '_kmer-' + str(kmer_size___)   + '_windw-'+str(window_size___)+'-wfreq-'+str(word_freq_cutoff___) ]]

                    Y_ext_ = np.array(df_ext['numeric_class'])

                if self.include_random_background_comparison_:
                    logging.info("Including additional evaluation on randomized background dataset")
                    ## NOTE: if using X_test_/Y_test_.copy() may not include undefined data (so performance might be higher than expected)

                    X_randombackground_ = X_train_.copy()#[[float(y) for y in x.replace('[', '').replace(']', '').replace(' ', '').split(',')] for x in self.df_test[e + '_encoded_' + flank_seq_working_key___ + '_kmer-' + str(kmer_size___) + '_windw-' + str(window_size___) + '-wfreq-' + str(word_freq_cutoff___)]]
                    Y_randombackground_ = Y_train_.copy()#list(self.df_train_['numeric_class'])

                    # To add undefined data back in (if it was removed)
                    # if self.remove_undefined_:
                    #     X_randombackground_undef_ = [list(x) for x in list(self.mid_undef_df[e + '_encoded_' + flank_seq_working_key___ + '_kmer-' + str(kmer_size___ ) + '_windw-'+str(window_size___)+'-wfreq-'+str(word_freq_cutoff___)  ])]
                    #     X_randombackground_ = X_randombackground_ + X_randombackground_undef_
                    #
                    #     Y_randombackground_undef_ = len(self.mid_undef_df)*[0]
                    #     Y_randombackground_ = np.array(Y_randombackground_ + Y_randombackground_undef_)

                    # X_randombackground_ = X_ext_.copy()
                    # Y_randombackground_ = Y_ext_.copy()


                    import random
                    random.shuffle(X_randombackground_)
                    random.shuffle(Y_randombackground_)


                #logging.info("Fitting model "+str(n_ + 1)+' / '+str(self.num_rerurun_model_building)+'...')
                clf_final.fit(X_train_, Y_train_)
                #logging.info("\tfitting complete!")


                preds_final = clf_final.predict_proba(X_test_)[:, 1]
                preds_final_inv = clf_final.predict_proba(X_test_)[:, 0]
                preds_binary_final = clf_final.predict(X_test_)



                if self.apply_final_models_to_external_dataset_:
                    ext_preds_final = clf_final.predict_proba(X_ext_)[:, 1]
                    ext_preds_final_inv = clf_final.predict_proba(X_ext_)[:, 0]
                    ext_preds_binary_final = clf_final.predict(X_ext_)

                if self.include_random_background_comparison_:
                    # randombackground_preds_final = preds_final.copy()
                    # randombackground_preds_final_inv = preds_final_inv.copy()
                    # randombackground_preds_binary_final = preds_binary_final.copy()

                    randombackground_preds_final =  clf_final.predict_proba(X_randombackground_)[:, 1]
                    randombackground_preds_final_inv =  clf_final.predict_proba(X_randombackground_)[:, 0]
                    randombackground_preds_binary_final = clf_final.predict(X_randombackground_)

                #logging.info("Evaluating performance of model " + str(n_ + 1) + ' / ' + str(self.num_rerurun_model_building) + '...')

                ## Evaluate Parameter Optimization Model Performance
                from sklearn.metrics import precision_recall_curve

                # NOTE: no p-r curves for label-propagation/spreading with one-hot encoding
                if not (((model_type___ == 'semi-sup-label-propagation') or (model_type___ == 'semi-sup-label-spreading')) and (e == 'one-hot')):
                    p_final_, r_final_, ts_final_ = precision_recall_curve(Y_test_, preds_final)
                    aucpr_final_ = metrics.auc(r_final_, p_final_)

                from sklearn.metrics import f1_score

                fscore_final_ = f1_score(Y_test_, preds_binary_final)  # , average=None)
                from sklearn.metrics import accuracy_score

                from sklearn.metrics import fbeta_score
                fbetascore_final_ = fbeta_score(Y_test_, preds_binary_final, beta=self.f_beta_)  # , average=None)
                logging.info("\nComputing Final fbeta_score with beta = "+str(self.f_beta_))

                accuracy_final_ = accuracy_score(Y_test_, preds_binary_final)
                from sklearn.metrics import matthews_corrcoef

                mcc_final_ = matthews_corrcoef(Y_test_, preds_binary_final)

                #logging.info("Constructing precision-recall curves for model " + str(n_ + 1) + ' / ' + str(self.num_rerurun_model_building) + '...')
                # NOTE: no p-r curves for label-propagation/spreading with one-hot encoding
                if not (((model_type___ == 'semi-sup-label-propagation') or (model_type___ == 'semi-sup-label-spreading')) and (e == 'one-hot')):
                    # Compute Unacheiveable Region
                    class_dist_final = p_final_[0]  # class distribution (Pr=1)
                    unach_recalls_final = list(r_final_)[::-1]
                    # compute y (precision) values from the x (recall) values
                    unach_precs_final = []
                    for x in unach_recalls_final:
                        y = (class_dist_final * x) / ((1 - class_dist_final) + (class_dist_final * x))
                        unach_precs_final.append(y)
                    unach_p_r_final_ = [unach_precs_final, unach_recalls_final]
                    auc_unach_adj_final_ = metrics.auc(r_final_, p_final_) - metrics.auc(unach_recalls_final, unach_precs_final)

                    final_performance_curves_dict_ = {
                        'Precision_Recall_Curve': [p_final_, r_final_, ts_final_],
                        'Unacheivable_Region_Curve': [unach_precs_final, unach_recalls_final],
                    }
                    aucpr_adj_final_ = aucpr_final_ - p_final_[0]
                else:
                    logging.info("\n\nWARNING: because model-type is " + str(model_type___) + ' and encoding type is ' + str(e) + ' Precision-Recall curves cannot be created for this model\n\n')
                    paramop_performance_curves_dict_ = {
                        'Precision_Recall_Curve': [[], [], []],
                        'Unacheivable_Region_Curve': [[], []],
                    }
                    aucpr_final_ = 0
                    aucpr_adj_final_ = 0
                    auc_unach_adj_final_ =0

                final_performance_metrics_dict_ = {
                    'AUCPR': aucpr_final_,
                    'AUCPR-adj': aucpr_adj_final_,
                    'AUCPR-unach-adj': auc_unach_adj_final_,
                    'F-Score': fscore_final_,
                    'Fbeta-Score': fbetascore_final_,
                    'Accuracy': accuracy_final_,
                    'MCC': mcc_final_,
                }

                key2__ = str(e) + '-'+self.parameter_to_optimize+'-'+str(param_val_) + '_round_'+ str(n_)
                key__ = str(e) + '_' + str(n_)
                self.final_models_encodings_dict[key__] = clf_final
                self.final_performance_metrics_encodings_dict[key__] = final_performance_metrics_dict_
                self.final_detailed_performance_metrics_encodings_dict[key2__] = final_performance_metrics_dict_
                self.final_performance_curves_encodings_dict[key__] = final_performance_curves_dict_
                self.final_key_ls.append(key__)
                # final_model_params_ls --> list of parameter VALUES in all final models
                #     - Same as the  list of top parameters from each round of parameter optimization
                #     - If parameter to optimize is Kmer will just contain integer kmer sizes ex: [3, 9, 12]
                self.final_model_params_ls.append(param_val_)

                if self.apply_final_models_to_external_dataset_:

                    logging.info("\nEvaluating performance on external dataset for model " + str(n_ + 1) + ' / ' + str(self.num_rerurun_model_building) + '...\n')

                    ## Evaluate Parameter Optimization Model Performance
                    # NOTE: no p-r curves for label-propagation/spreading with one-hot encoding
                    if not (((model_type___ == 'semi-sup-label-propagation') or (model_type___ == 'semi-sup-label-spreading')) and (e == 'one-hot')):
                        ext_p_final_, ext_r_final_, ext_ts_final_ = precision_recall_curve(Y_ext_, ext_preds_final)
                        ext_aucpr_final_ = metrics.auc(ext_r_final_, ext_p_final_)

                    ext_fscore_final_ = f1_score(Y_ext_, ext_preds_binary_final)  # , average=None)

                    ext_fbetascore_final_ = fbeta_score(Y_ext_, ext_preds_binary_final, beta=self.f_beta_)  # , average=None)
                    logging.info("\nComputing Final fbeta_score with beta = "+str(self.f_beta_))

                    ext_accuracy_final_ = accuracy_score(Y_ext_, ext_preds_binary_final)

                    ext_mcc_final_ = matthews_corrcoef(Y_ext_, ext_preds_binary_final)
                    # NOTE: no p-r curves for label-propagation/spreading with one-hot encoding
                    if not (((model_type___ == 'semi-sup-label-propagation') or (model_type___ == 'semi-sup-label-spreading')) and (e == 'one-hot')):
                        # Compute Unacheiveable Region
                        ext_class_dist_final = ext_p_final_[0]  # class distribution (Pr=1)
                        ext_unach_recalls_final = list(ext_r_final_)[::-1]
                        # compute y (precision) values from the x (recall) values
                        ext_unach_precs_final = []
                        for x in ext_unach_recalls_final:
                            ext_y = (ext_class_dist_final * x) / ((1 - ext_class_dist_final) + (ext_class_dist_final * x))
                            ext_unach_precs_final.append(ext_y)
                        ext_unach_p_r_final_ = [ext_unach_precs_final, ext_unach_recalls_final]
                        ext_auc_unach_adj_final_ = metrics.auc(ext_r_final_, ext_p_final_) - metrics.auc(ext_unach_recalls_final, ext_unach_precs_final)

                        ext_final_performance_curves_dict_ = {
                            'Precision_Recall_Curve': [ext_p_final_, ext_r_final_, ext_ts_final_],
                            'Unacheivable_Region_Curve': [ext_unach_precs_final, ext_unach_recalls_final],
                        }
                        ext_aucpr_adj_final_ = ext_aucpr_final_ - ext_p_final_[0]
                    else:
                        logging.info("\n\nWARNING: because model-type is " + str(model_type___) + ' and encoding type is ' + str(e) + ' Precision-Recall curves cannot be created for this model\n\n')
                        ext_paramop_performance_curves_dict_ = {
                            'Precision_Recall_Curve': [[], [], []],
                            'Unacheivable_Region_Curve': [[], []],
                        }
                        ext_aucpr_final_ = 0
                        ext_aucpr_adj_final_ = 0
                        ext_auc_unach_adj_final_ = 0

                    ext_final_performance_metrics_dict_ = {
                        'AUCPR': ext_aucpr_final_,
                        'AUCPR-adj': ext_aucpr_adj_final_,
                        'AUCPR-unach-adj': ext_auc_unach_adj_final_,
                        'F-Score': ext_fscore_final_,
                        'Fbeta-Score': ext_fbetascore_final_,
                        'Accuracy': ext_accuracy_final_,
                        'MCC': ext_mcc_final_,
                    }

                    self.ext_final_performance_metrics_encodings_dict[key__] = ext_final_performance_metrics_dict_
                    self.ext_final_detailed_performance_metrics_encodings_dict[key2__] = ext_final_performance_metrics_dict_
                    self.ext_final_performance_curves_encodings_dict[key__] = ext_final_performance_curves_dict_


                if self.include_random_background_comparison_:
                    logging.info("\nEvaluating performance on Randomized (shuffled) Background Data for model " + str(n_ + 1) + ' / ' + str(self.num_rerurun_model_building) + '...\n')


                    ## Evaluate Parameter Optimization Model Performance
                    # NOTE: no p-r curves for label-propagation/spreading with one-hot encoding
                    if not (((model_type___ == 'semi-sup-label-propagation') or (model_type___ == 'semi-sup-label-spreading')) and (e == 'one-hot')):
                        randombackground_p_final_, randombackground_r_final_, randombackground_ts_final_ = precision_recall_curve(Y_randombackground_, randombackground_preds_final)
                        randombackground_aucpr_final_ = metrics.auc(randombackground_r_final_, randombackground_p_final_)

                    randombackground_fscore_final_ = f1_score(Y_randombackground_, randombackground_preds_binary_final)  # , average=None)

                    randombackground_fbetascore_final_ = fbeta_score(Y_randombackground_, randombackground_preds_binary_final, beta=self.f_beta_)  # , average=None)
                    logging.info("\nComputing Final fbeta_score with beta = "+str(self.f_beta_))

                    randombackground_accuracy_final_ = accuracy_score(Y_randombackground_, randombackground_preds_binary_final)

                    randombackground_mcc_final_ = matthews_corrcoef(Y_randombackground_, randombackground_preds_binary_final)
                    # NOTE: no p-r curves for label-propagation/spreading with one-hot encoding
                    if not (((model_type___ == 'semi-sup-label-propagation') or (model_type___ == 'semi-sup-label-spreading')) and (e == 'one-hot')):
                        # Compute Unacheiveable Region
                        randombackground_class_dist_final = randombackground_p_final_[0]  # class distribution (Pr=1)
                        randombackground_unach_recalls_final = list(randombackground_r_final_)[::-1]
                        # compute y (precision) values from the x (recall) values
                        randombackground_unach_precs_final = []
                        for x in randombackground_unach_recalls_final:
                            randombackground_y = (randombackground_class_dist_final * x) / ((1 - randombackground_class_dist_final) + (randombackground_class_dist_final * x))
                            randombackground_unach_precs_final.append(randombackground_y)
                        randombackground_unach_p_r_final_ = [randombackground_unach_precs_final, randombackground_unach_recalls_final]
                        randombackground_auc_unach_adj_final_ = metrics.auc(randombackground_r_final_, randombackground_p_final_) - metrics.auc(randombackground_unach_recalls_final, randombackground_unach_precs_final)

                        randombackground_final_performance_curves_dict_ = {
                            'Precision_Recall_Curve': [randombackground_p_final_, randombackground_r_final_, randombackground_ts_final_],
                            'Unacheivable_Region_Curve': [randombackground_unach_precs_final, randombackground_unach_recalls_final],
                        }
                        randombackground_aucpr_adj_final_ = randombackground_aucpr_final_ - randombackground_p_final_[0]
                    else:
                        logging.info("\n\nWARNING: because model-type is " + str(model_type___) + ' and encoding type is ' + str(e) + ' Precision-Recall curves cannot be created for this model\n\n')
                        randombackground_paramop_performance_curves_dict_ = {
                            'Precision_Recall_Curve': [[], [], []],
                            'Unacheivable_Region_Curve': [[], []],
                        }
                        randombackground_aucpr_final_ = 0
                        randombackground_aucpr_adj_final_ = 0
                        randombackground_auc_unach_adj_final_ = 0

                    randombackground_final_performance_metrics_dict_ = {
                        'AUCPR': randombackground_aucpr_final_,
                        'AUCPR-adj': randombackground_aucpr_adj_final_,
                        'AUCPR-unach-adj': randombackground_auc_unach_adj_final_,
                        'F-Score': randombackground_fscore_final_,
                        'Fbeta-Score': randombackground_fbetascore_final_,
                        'Accuracy': randombackground_accuracy_final_,
                        'MCC': randombackground_mcc_final_,
                    }

                    self.randombackground_final_performance_metrics_encodings_dict[key__] = randombackground_final_performance_metrics_dict_
                    self.randombackground_final_detailed_performance_metrics_encodings_dict[key2__] = randombackground_final_performance_metrics_dict_
                    self.randombackground_final_performance_curves_encodings_dict[key__] = randombackground_final_performance_curves_dict_





                # Pickle final models (per round, per embedding)
                #logging.info("Pickling model " + str(n_ + 1) + ' / ' + str(self.num_rerurun_model_building) + '...')
                fnm_ = self.output_directory + 'models/' + 'final_'+model_type_dict[model_type___]+'_model_rnd-' + str(n_+1) + '_'+feature_encodings_dict[e]+'.pickle'

                with open(fnm_, 'wb') as pickle_file:
                    pickle.dump(clf_final, pickle_file)
                pickle_file.close()

                logging.info('\n\n\nFinal model '+str(n_+1)+' with encoding '+str(e)+' saved to: '+fnm_.replace(self.output_directory, '~/\n\n'))

                # with open(fnm_, 'wb') as pickle_file:
                #     clf__ = pickle.load(pickle_file)
                # pickle_file.close()

                # For exporting predictions to .csv file

                all_preds_dict[str(n_) + '_' + key2__ + '_actual'] = list(Y_test_)
                all_preds_dict[str(n_) + '_' + key2__ + '_preds_final'] = list(preds_final)
                all_preds_dict[str(n_) + '_' + key2__ + '_preds_final_inv'] = list(preds_final_inv)
                all_preds_dict[str(n_) + '_' + key2__ + '_preds_binary_final'] = list(preds_binary_final)


                if self.apply_final_models_to_external_dataset_:
                    all_preds_dict_ext[str(n_) + '_on-ext-data_' + key2__ + '_actual'] = list(Y_ext_)
                    all_preds_dict_ext[str(n_) + '_on-ext-data_' + key2__ + '_preds_final'] = list(ext_preds_final)
                    all_preds_dict_ext[str(n_) + '_on-ext-data_' + key2__ + '_preds_final_inv'] = list(ext_preds_final_inv)
                    all_preds_dict_ext[str(n_) + '_on-ext-data_' + key2__ + '_preds_binary_final'] = list(ext_preds_binary_final)

                if self.include_random_background_comparison_:
                    all_preds_dict_randombackground[str(n_) + '_on-randombackground-data_' + key2__ + '_actual'] = list(Y_randombackground_)
                    all_preds_dict_randombackground[str(n_) + '_on-randombackground-data_' + key2__ + '_preds_final'] = list(randombackground_preds_final)
                    all_preds_dict_randombackground[str(n_) + '_on-randombackground-data_' + key2__ + '_preds_final_inv'] = list(randombackground_preds_final_inv)
                    all_preds_dict_randombackground[str(n_) + '_on-randombackground-data_' + key2__ + '_preds_binary_final'] = list(randombackground_preds_binary_final)

        # Export predictions to .csv file
        fnm_ = self.output_directory + 'data/' + 'predictions_final_models.csv'
        all_preds_dict_df = pd.DataFrame(all_preds_dict)
        all_preds_dict_df.to_csv(fnm_)
        logging.info('\n\n\nPredictions from final model saved to: '+fnm_.replace(self.output_directory, '~/\n\n'))

        if self.apply_final_models_to_external_dataset_:
            fnm_ = self.output_directory + 'data/' + 'predictions_final_models_ext_dataset.csv'
            all_preds_dict_ext_df = pd.DataFrame(all_preds_dict_ext)
            all_preds_dict_ext_df.to_csv(fnm_)
            logging.info('\n\n\nPredictions on External Dataset from final model saved to: '+fnm_.replace(self.output_directory, '~/\n\n'))

        if self.include_random_background_comparison_:
            fnm_ = self.output_directory + 'data/' + 'predictions_final_models_randombackground_dataset.csv'
            all_preds_dict_randombackground_df = pd.DataFrame(all_preds_dict_randombackground)
            all_preds_dict_randombackground_df.to_csv(fnm_)
            logging.info('\n\n\nPredictions on Random Background Dataset from final model saved to: '+fnm_.replace(self.output_directory, '~/\n\n'))






    def plot_param_opt_precision_recall_curves(self):
        logging.info("\nPlotting P-R curves for parameter optimization...")
        ## Plot compiled Parameter Optimization Precision-Recall curves as a single figure
        sup_title_id_info = ('' +  # str(num_rerurun_model_building*run_round_num)+' rounds'+'\n'+
                             self.output_run_file_info_string_.replace('_', ' ').replace(self.region_.replace('_', '-'), self.region_.replace('_', '-') + '\n'))


        pr_curves_keys_ = self.paramop_key_ls
        title_st_ = sup_title_id_info + 'Parameter Optimization ' + str(self.parameter_to_optimize) + '\n' + str(self.num_rerurun_model_building) + ' Rounds ' + str(len(self.param_values_to_loop_)) + ' Parameter Values'

        ## Find nearest square for plotting grid
        rows_cols_compiled_po_fig = math.ceil(math.sqrt(self.num_rerurun_model_building))

        param_sizes_to_loop_ = self.param_values_to_loop_


        param_col_ls = list(sns.color_palette("hls", len(param_sizes_to_loop_)).as_hex())
        greys_col_ls = list(sns.color_palette("Greys", len(param_sizes_to_loop_)).as_hex())

        # Create dictionary of parameter colors based on parameter
        param_col_ls_dict = {}
        for i in range(len(param_sizes_to_loop_)):
            param_col_ls_dict[param_sizes_to_loop_[i]] = param_col_ls[i]

        greys_col_ls_dict = {}
        for i in range(len(param_sizes_to_loop_)):
            greys_col_ls_dict[param_sizes_to_loop_[i]] = greys_col_ls[i]

        embd_color_dict = {}
        for e in self.feature_encoding_ls:
            embd_color_dict[e] = param_col_ls_dict


        # Paramopt keys: str(e) + '-' + self.parameter_to_optimize + '-' + str(param_) + '_' + str(n_)
        # Final model keys: str(e) + '_' + str(n_)

        # Plot a single figure for each embedding type
        for e in self.feature_encoding_ls: # different figures
            fig, axs = plt.subplots(rows_cols_compiled_po_fig, rows_cols_compiled_po_fig + 1)
            fig.set_size_inches(w=9, h=9, )  # NOTE: h and w must be large enough to accomodate any legends

            # Remove last column
            gs = axs[0, rows_cols_compiled_po_fig].get_gridspec()
            # remove the underlying axes
            for ax in axs[0:, -1]:
                ax.remove()
            axbig = fig.add_subplot(gs[0:, -1])

            # Remove axes that are not needed (based on number of total parameter optmimization rounds)
            ct_po_rounds_to_plot_ = 1
            for row_ in range(len(axs)):  # num rows
                for col_ in range(len(axs[0]) - 1):  # num cols -1 (to skip last column)
                    # logging.info(row_,col_,ct_po_rounds_to_plot_)
                    if ct_po_rounds_to_plot_ > self.num_rerurun_model_building:#len(p_r_t_po__ls_):
                        # remove axis from figure
                        axs[row_][col_].remove()
                    ct_po_rounds_to_plot_ += 1
            # get best parameter value per round
            best_aucprs_dict = {}

            all_prts_dict = {} # For exporting Precision-Recall curve numeric data as a .csv file
            longest_prt_ = -1 # For exporting Precision-Recall curve numeric data as a .csv file
            for n in range(self.num_rerurun_model_building): # loop through rounds
                best_round_aucpr_ = -9999
                best_round_param_ = ''
                for p in self.param_values_to_loop_:  # different colors
                    k__ =str(e) + '-' + self.parameter_to_optimize + '-' + str(p) + '_round_' + str(n)
                    if k__ in pr_curves_keys_:

                        # NOTE: no p-r curves for label-propagation/spreading with one-hot encoding
                        if not (((self.parameter_to_optimize == 'model') and ('label' in p) and (e == 'one-hot')) or (
                                ('label' in self.model_type_) and (e == 'one-hot'))):
                                prt_ls = self.paramop_performance_curves_encodings_dict[k__]['Precision_Recall_Curve']
                                prt_ls = [list(x) for x in prt_ls]
                                all_prts_dict[str(n)+'_'+k__] = prt_ls
                                for v in prt_ls:
                                    if longest_prt_ < len(v): # keep track of longest Precision, Recall, Threshold curve list to pad later
                                        longest_prt_ = len(v)
                                aucpr_ = metrics.auc(prt_ls[1], prt_ls[0])
                                if aucpr_ > best_round_aucpr_:
                                    best_round_aucpr_ = aucpr_
                                    best_round_param_ = p
                        else: # case where label-spreading/label-propag and one-hot encoding
                            aucpr_ = 0
                best_aucprs_dict[n] = [best_round_param_,best_round_aucpr_,]

            # Export Precision-Recall curve numeric data as a .csv file
            # Pad all curves in all_prts_dict so all same length as longest_prt_
            all_prts_dict_for_df = {}
            for k__ in list(all_prts_dict.keys()):
                # padded_prts_ls = []
                for v, label_ in zip(all_prts_dict[k__], ['precision', 'recall', 'thresholds']):
                    amt_to_pad_ = longest_prt_ - len(v)  # [[ ],[ ], [ ]]
                    v += [0] * amt_to_pad_
                    all_prts_dict_for_df[label_ + '_' + k__] = v
                    # padded_prts_ls.append(v)
                # all_prts_dict[k__] = padded_prts_ls
            all_prts_paramopt_df = pd.DataFrame(all_prts_dict_for_df)
            # all_prts_paramopt_df = pd.DataFrame(all_prts_dict)

            # ** SAVE DATA **
            fnm_ = (self.output_directory + 'data/' + 'paramopt_performance_metrics_p-r_' + str(self.num_rerurun_model_building) + '-rnds_po_' + e + '.csv')
            all_prts_paramopt_df.to_csv(fnm_)
            logging.info('\n\nParamopt. Precision-Recall-Threshold curve data saved to: '+fnm_.replace(self.output_directory, '~/'))


            col_ = 0
            row_ = 0
            for n in range(self.num_rerurun_model_building):  # different axes per round
                axs[row_, col_].set_title('Round: ' + str(n + 1) + '\nBest ' + self.parameter_to_optimize +' '+ str(best_aucprs_dict[n][0]),
                                          fontsize=12, fontweight= 'bold', color=param_col_ls_dict[best_aucprs_dict[n][0]])

                for p in self.param_values_to_loop_: # different colors
                    k__ =str(e) + '-' + self.parameter_to_optimize + '-' + str(p) + '_round_' + str(n)
                    if k__ in pr_curves_keys_:

                        # NOTE: no p-r curves for label-propagation/spreading with one-hot encoding
                        if not(((self.parameter_to_optimize == 'model') and ('label' in p) and (e == 'one-hot')) or (
                                ('label' in self.model_type_) and (e == 'one-hot'))):
                            prt_ls = self.paramop_performance_curves_encodings_dict[k__]['Precision_Recall_Curve']
                            if (self.parameter_to_optimize == 'kmer-size') and (e == 'one-hot'):
                                axs[row_, col_].plot(
                                    prt_ls[1],  # r_OH,# x
                                    prt_ls[0],  # p_OH,# y
                                    lw=1,
                                    color=greys_col_ls_dict[p],
                                    # color = embd_color_dict[embd_][key__],
                                )
                            else:
                                axs[row_,col_].plot(
                                    prt_ls[1],#r_OH,# x
                                    prt_ls[0],#p_OH,# y
                                    lw=1,
                                    color= param_col_ls_dict[p],
                                    #color = embd_color_dict[embd_][key__],
                                )

                            aucpr_ = metrics.auc(prt_ls[1], prt_ls[0])

                        #logging.info(row_,col_)
                if col_ == rows_cols_compiled_po_fig-1:
                    col_ = 0
                    row_+=1
                else:
                    col_+=1
            # Format Axes
            col_ = 0
            row_ = 0
            for i_ in range(self.num_rerurun_model_building):#len(p_r_t_po__ls_)):
                axs[row_,col_].set_xlim(0,1)
                axs[row_,col_].set_ylim(0,1.1)
                axs[row_,col_].tick_params(direction='in',which='both',length=3,width=1)

                if (row_ == rows_cols_compiled_po_fig-1) and (col_ == 0):
                    axs[row_,col_].set_xlabel('Recall')
                    axs[row_,col_].set_ylabel('Precision')
                    axs[row_,col_].set_xticks(ticks=np.arange(0,1.1,.5),labels=['',0.5,1.0])
                    axs[row_,col_].set_yticks(ticks=[0.0 , 0.5, 1.0 ], labels=[0.0, 0.5, 1.0])

                else: # Don't label the ticks because are a grid
                    # axs[row_,col_].set_xticks([])
                    # axs[row_,col_].set_yticks([])
                    axs[row_, col_].set_xticks(ticks=[0.0 , 0.5, 1.0 ], labels=['', '', ''])
                    axs[row_, col_].set_yticks(ticks=[0.0 , 0.5, 1.0 ], labels=['', '', ''])

                if col_ == rows_cols_compiled_po_fig-1:
                    col_ = 0
                    row_+=1
                else:
                    col_+=1

            # Add legend for parameter values

            legend_elements = []
            for val__,i in zip(param_sizes_to_loop_,range(len(param_sizes_to_loop_))):
                legend_elements.append(Line2D([0], [0],
                                              color= param_col_ls_dict[val__],#embd_color_dict[embd_][val__],
                                              lw=4, label=str(val__)))
            axs[0][rows_cols_compiled_po_fig].legend(handles=legend_elements, loc='upper left', frameon=False,bbox_to_anchor = (0,1),title=self.parameter_to_optimize,title_fontsize=12,fontsize=12)
            axs[0][rows_cols_compiled_po_fig].axis('off')
            axbig.legend(handles=legend_elements, loc='upper left', frameon=False,bbox_to_anchor = (0,1),title=self.parameter_to_optimize,title_fontsize=12,fontsize=12)
            axbig.axis('off')

            plt.suptitle(e.replace('_',' ').capitalize(), fontsize = 12, fontweight = 'bold')
            fig.tight_layout() # NOTE: h and w (above in fig.set_size... MUST be large enough to accomodate legends or will be cut off/squished in output)
            # ** SAVE FIGURE **
            plt.rcParams['svg.fonttype'] = 'none' # exports text as strings rather than vector paths (images)
            fnm_ =     (self.output_directory+ 'figures/'+           'p-r_'+str(self.num_rerurun_model_building)+'-rnds_po_'+e)
            fnm_svg_ = (self.output_directory+'figures/'+'svg_figs/'+'p-r_'+str(self.num_rerurun_model_building)+'-rnds_po_'+e)
            fig.savefig(fnm_svg_.split('.')[0]+'.svg',format='svg',transparent=True)
            fig.savefig(fnm_.split('.')[0]+'.png',format='png',dpi=300,transparent=False)
            plt.close(fig)
            logging.info('Figure saved to: ' +(fnm_ + '.png').replace(self.output_directory, '~/'))



    def plot_param_opt_model_box_plots(self):
        logging.info("\nPlotting box plots for parameter optimization...")
        ## Plot Compiled Multimetrics Model Performance - Parameter Optimization Models
        # Each column of paramop_detailed_metric_df contains a single round for a single embedding type
        paramop_detailed_metric_df = pd.DataFrame(self.paramop_performance_metrics_encodings_dict)
        fnm_ = (self.output_directory + 'data/' + 'paramopt_performance_metrics_' + str(self.num_rerurun_model_building) + '-rnds_paramopt.csv')
        paramop_detailed_metric_df.to_csv(fnm_,index=True)
        logging.info("Parameter Optimization Models Performance Metrics Dataframe saved to:\n\t"+fnm_)

        flierprops__ = dict(marker='.', markerfacecolor='none', markersize=4, linewidth=0.1, markeredgecolor='black')  # linestyle='none',
        boxprops__ = dict(facecolor='none', linestyle='none', linewidth=1, edgecolor='k', )
        medianprops__paramopt = dict(linewidth=2, color=paramopt_set_plot_color)


        # Loop for each encoding and plot as a grid with different encoding per line

        # one row per embedding
        # one axis per performance metric
        # one box per embedding per axis
        # fig, axs = plt.subplots(len(self.feature_encoding_ls), len(paramop_detailed_metric_df))
        # fig.set_size_inches(w=14, h=3 * len(self.feature_encoding_ls))
        # Update: Make plots in sets of num_rows  (so figures not too large)
        num_rows_ = 10 # number of rows per plot (i.e. number of rows per figure)
        num_plots_ = 1# int(int(len(self.feature_encoding_ls)/ num_rows_ )+(len(self.feature_encoding_ls)% num_rows_ ))
        paired_feature_encoding_ls = [self.feature_encoding_ls[i:i+num_rows_] for i in range(0, len(self.feature_encoding_ls), num_rows_)]



        for plotn_ in list(range(num_plots_)):
            num_rows_actually_plotted_ = 0
            # if plotn_ == num_plots_ - 1:  # TODO: last plot is just 1 row since is a combined plot with all embeddings
            #     fig, axs = plt.subplots(1, len(paramop_detailed_metric_df))
            #     fig.set_size_inches(w=14, h=3)
            # else:
            #     fig, axs = plt.subplots(len(paired_feature_encoding_ls[plotn_]), len(paramop_detailed_metric_df))
            #     fig.set_size_inches(w=14, h=3*len(paired_feature_encoding_ls[plotn_]))

            fig, axs = plt.subplots(num_rows_, len(paramop_detailed_metric_df))
            fig.set_size_inches(w=12, h= num_rows_ * 1.5)

            # Split Evaluation Metric Data per parameter value
            # Loop through each embedding type
            #for embedding_type_paramop_eval_, j in zip(self.feature_encoding_ls, list(range(len(self.feature_encoding_ls)))):
            for embedding_type_paramop_eval_, j in zip(paired_feature_encoding_ls[plotn_], list(range(len(paired_feature_encoding_ls[plotn_])))):
                num_rows_actually_plotted_ +=1

                # From paramop_detailed_metric_df get just columns for a single selected embedding type
                # Get just columns with selected embedding metric (embedding_type_paramop_eval_)
                cols_with_embd_ = [x for x in list(paramop_detailed_metric_df.columns) if embedding_type_paramop_eval_ in x]
                logging.info("Columns from paramop_detailed_metric_df with embedding = 'embedding_type_paramop_eval_' = " +
                      str(embedding_type_paramop_eval_) + " - "+str(len(cols_with_embd_))+' out of '+str(len(list(paramop_detailed_metric_df.columns))))

                # Get just columns with selected embedding metric (embedding_type_paramop_eval_)
                paramop_detailed_metric_one_embd_df = paramop_detailed_metric_df[cols_with_embd_]

                # Get parameter VALUES only for each selected embedding
                param_vals_one_embd_ = list(set([x.split(str(self.parameter_to_optimize) + '-')[-1].split('_round_')[0] for x in list(paramop_detailed_metric_one_embd_df.columns)]))

                # If Parameter values are integers, order them from smallest to larget
                try:
                    int_param_vals_one_embd_ = [int(x) for x in param_vals_one_embd_]
                    int_param_vals_one_embd_.sort()
                    param_vals_one_embd_ = [str(x) for x in int_param_vals_one_embd_]
                except:
                    pass

                # # update x-axis labels
                tick_lab_list__ = []
                for x in param_vals_one_embd_:
                    tick_lab_list__.append('')
                for x in param_vals_one_embd_:
                    tick_lab_list__.append(x)


                metrics_ls = list(paramop_detailed_metric_one_embd_df.index)

                for i in range(len(metrics_ls)):
                    metric_ = metrics_ls[i]
                    # Plot by parameter value
                    data_ = [list(paramop_detailed_metric_one_embd_df[
                                      [embedding_type_paramop_eval_ + '-' + str(self.parameter_to_optimize) + '-' + param_val_ + '_round_' + str(i) for i in
                                       list(range(self.num_rerurun_model_building))]].transpose()[metric_]) for param_val_ in param_vals_one_embd_]
                    # Multiple rows
                    # try:
                    bplot1 = axs[j,i].boxplot(
                        data_,
                        vert=True,  # vertical box alignment
                        patch_artist=True,  # fill with color
                        labels=param_vals_one_embd_,
                        flierprops=flierprops__, boxprops=boxprops__,medianprops=medianprops__paramopt,
                        capprops=dict(color='black'),
                        whiskerprops=dict(color='black'),

                    )  # will be used to label x-ticks
                    axs[j,i].set_title(metric_.replace('beta',str(self.f_beta_)), fontsize=8.5 )
                    if i == 3:
                        #if embedding_type_paramop_eval_ == self.feature_encoding_ls[0]:# for first row of plots in figure
                        if embedding_type_paramop_eval_ == paired_feature_encoding_ls[0]:  # for first row of plots in figure
                            axs[j,i].set_title('Plot '+str(plotn_+1)+' / '+str(num_plots_)+' Parameter Optimization Model Performances (' + str(self.num_rerurun_model_building) + ' Rounds)\n'+str(embedding_type_paramop_eval_)+'\n' + str(metric_), fontsize=8.5 )  # ,fontweight='bold')
                        else:
                            axs[j,i].set_title(str(embedding_type_paramop_eval_)+'\n' + str(metric_), fontsize=8.5 )  # ,fontweight='bold')

                    # update x-axis labels
                    #axs[j, i].set_xticklabels(tick_lab_list__, rotation=0, fontsize=8.5)
                    axs[j,i].set_xticklabels(param_vals_one_embd_, rotation=0, fontsize=8.5)
                    axs[j,i].set_xlabel(str(self.parameter_to_optimize), fontsize=8.5 )
                    if metric_ == 'MCC':
                        axs[j,i].set_ylim(-1, 1)
                        axs[j,i].set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
                        axs[j,i].tick_params(axis='y', labelsize=7)
                    else:
                        axs[j,i].set_ylim(0, 1)
                        axs[j,i].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                        axs[j,i].tick_params(axis='y', labelsize=7)

                    #*# self.autolabel_boxplot_below(bplot1['caps'][::2],  bplot1['medians'], axs[j,i])
                    # # One row
                    # except:
                    #     bplot1 = axs[i].boxplot(
                    #         data_,
                    #         vert=True,  # vertical box alignment
                    #         patch_artist=True,  # fill with color
                    #         labels=param_vals_one_embd_,
                    #         flierprops=flierprops__, boxprops=boxprops__,medianprops=medianprops__paramopt,
                    #         capprops=dict(color='black'),
                    #         whiskerprops=dict(color='black'),
                    #
                    #     )  # will be used to label x-ticks
                    #     axs[i].set_title(metric_.replace('beta',str(self.f_beta_)), fontsize=8.5 )
                    #     if i == 3:
                    #         # if embedding_type_paramop_eval_ == self.feature_encoding_ls[0]:# for first row of plots in figure
                    #         if embedding_type_paramop_eval_ == paired_feature_encoding_ls[0]:  # for first row of plots in figure
                    #             axs[i].set_title('Plot ' + str(plotn_ + 1) + ' / ' + str(num_plots_) + ' Parameter Optimization Model Performances (' + str(self.num_rerurun_model_building) + ' Rounds)\n' + str(embedding_type_paramop_eval_) + '\n' + str(metric_), fontsize=8.5 )  # ,fontweight='bold')
                    #         else:
                    #             axs[i].set_title(str(embedding_type_paramop_eval_) + '\n' + str(metric_), fontsize=8.5 )  # ,fontweight='bold')
                    #
                    #     # update x-axis labels
                    #     #axs[i].set_xticklabels(tick_lab_list__, rotation=0, fontsize=8.5)
                    #     # axs[i].set_xticklabels(param_vals_one_embd_, rotation=0, fontsize=8.5)
                    #     axs[i].set_xlabel(str(self.parameter_to_optimize), fontsize=8.5 )
                    #     if metric_ == 'MCC':
                    #         axs[i].set_ylim(-1, 1)
                    #         axs[i].set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
                    #         axs[i].tick_params(axis='y', labelsize=7)
                    #     else:
                    #         axs[i].set_ylim(0, 1)
                    #         axs[i].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                    #         axs[i].tick_params(axis='y', labelsize=7)
                    #     #*# self.autolabel_boxplot_below(bplot1['caps'][::2], bplot1['medians'], axs[i])


            # hide axes if nothing plotted
            num_rows_to_erase_ = num_rows_ - num_rows_actually_plotted_
            for j in range(num_rows_to_erase_):
                for i in range(len(paramop_detailed_metric_df)):
                    axs[::-1][j, i].set_visible(False) # reverse because works from bottom up (backwards)

            # fig.suptitle(str(plotn_+1)+' Compiled Multiple Metrics Parameter Optimization Models - Per Parameter Value ' + str(self.num_rerurun_model_building) +
            #              ' rounds' + '\n' + self.output_run_file_info_string_.replace('_', ' ').replace(self.region_.replace('_', '-'), self.region_.replace('_', '-') + '\n'), fontsize=7)
            fig.tight_layout()

            # ** SAVE FIGURE **
            plt.rcParams['svg.fonttype'] = 'none'  # exports text as strings rather than vector paths (images)
            fnm_ = (self.output_directory + 'figures/' + str(plotn_+1)+'_po_bxp_per-param-val_' + str(self.num_rerurun_model_building) + '-rnds')
            fnm_svg_ = (self.output_directory + 'figures/' + 'svg_figs/' +str(plotn_+1)+'_po_bxp_per-param-val_' + str(self.num_rerurun_model_building) + '-rnds')
            fig.savefig(fnm_svg_.split('.')[0] + '.svg', format='svg', transparent=True)
            fig.savefig(fnm_.split('.')[0] + '.png', format='png', dpi=300, transparent=False)
            logging.info('Figure saved to:'+str(fnm_) + '.png'.replace(self.output_directory, '~/'))

        return



    def plot_final_model_precision_recall_curves(self):
        logging.info("\nPlotting precision-recall curves for final models...")
        ## Plot Final Model Precision-Recall curves as a single figure
        sup_title_id_info = ('' +  # str(num_rerurun_model_building*run_round_num)+' rounds'+'\n'+
                             self.output_run_file_info_string_.replace('_', ' ').replace(self.region_.replace('_', '-'), self.region_.replace('_', '-') + '\n'))

        pr_curves_dict_ = self.final_performance_curves_encodings_dict
        pr_curves_keys_ = self.final_key_ls


        # Create dictionary of parameter colors based on parameter
        param_col_ls = list(sns.color_palette("hls", len(self.param_values_to_loop_)).as_hex())
        greys_col_ls = list(sns.color_palette("Greys", len(self.param_values_to_loop_)).as_hex())

        param_col_ls_dict = {}
        for i in range(len(self.param_values_to_loop_)):
            param_col_ls_dict[self.param_values_to_loop_[i]] = param_col_ls[i]

        greys_col_ls_dict = {}
        for i in range(len(self.param_values_to_loop_)):
            greys_col_ls_dict[self.param_values_to_loop_[i]] = greys_col_ls[i]

        embd_color_dict = {}
        for e in self.feature_encoding_ls:
            embd_color_dict[e] = param_col_ls_dict



        # Plot a single PLOT for each embedding type
        fig, axs = plt.subplots(1,len(self.feature_encoding_ls )+ 1)
        fig.set_size_inches(w=3*(len(self.feature_encoding_ls )+ 1), h=4.5, )  # NOTE: h and w must be large enough to accomodate any legends

        all_prts_dict = {}  # For exporting Precision-Recall curve numeric data as a .csv file
        longest_prt_ = -1  # For exporting Precision-Recall curve numeric data as a .csv file

        for e, col_ in zip(self.feature_encoding_ls,list(range(len(self.feature_encoding_ls)))):  # different plots per embedding
            axs[col_].set_title(str(e))
            for n in range(self.num_rerurun_model_building):  # loop through rounds

                # Get p-r curves per round for given embedding and round
                key__ = str(e) + '_' + str(n)
                p = self.final_model_params_ls[n]

                try: # get color for parameter optimization looping
                    color_ = param_col_ls_dict[p]
                except: # if not looping through parameters select color from list
                    color_ = testing_set_plot_color #


                # NOTE: no p-r curves for label-propagation/spreading with one-hot encoding
                if not (((self.parameter_to_optimize == 'model') and ('label' in p) and (e == 'one-hot')) or (
                        ('label' in self.model_type_) and (e == 'one-hot'))):

                    pcurve__ = self.final_performance_curves_encodings_dict[key__]['Precision_Recall_Curve'][0]
                    rcurve__ = self.final_performance_curves_encodings_dict[key__]['Precision_Recall_Curve'][1]
                    thresholds__ = self.final_performance_curves_encodings_dict[key__]['Precision_Recall_Curve'][2]
                    unach_pcurve__ = self.final_performance_curves_encodings_dict[key__]['Unacheivable_Region_Curve'][0]
                    unach_rcurve__ = self.final_performance_curves_encodings_dict[key__]['Unacheivable_Region_Curve'][1]

                    # For exporting Precision-Recall curve numeric data as a .csv file
                    prt_ls = [list(pcurve__),list(rcurve__),list(thresholds__)]
                    all_prts_dict[str(n) + '_' + key__] = prt_ls
                    for v in prt_ls:
                        if longest_prt_ < len(v):  # keep track of longest Precision, Recall, Threshold curve list to pad later
                            longest_prt_ = len(v)

                    axs[col_].plot(
                        rcurve__,  # r_OH,# x
                        pcurve__,  # p_OH,# y
                        lw=1,
                        color= color_,
                    )

                    # axs[col_].plot(
                    #     rcurve__,  # r_OH,# x
                    #     unach_rcurve__,  # p_OH,# y
                    #     lw=1,
                    #     color=color_,
                    #     linestyle='dashed',
                    # )

            # Format Axes
            axs[col_].set_xlim(0, 1)
            axs[col_].set_ylim(0, 1.1)
            axs[col_].set_xticks(ticks=np.arange(0, 1.1, .25), minor=True)
            axs[col_].tick_params(direction='in', which='both', length=3, width=1)

            if col_ == 0:
                axs[col_].set_xlabel('Recall')
                axs[col_].set_ylabel('Precision')
                axs[col_].set_xticks(ticks=[0.5, 1.0 ], labels=[0.5, 1.0])
                axs[col_].set_yticks(ticks=[0.0 , 0.5, 1.0 ], labels=[0.0, 0.5, 1.0])

            else:
                axs[col_].set_xticks(ticks=[0.5, 1.0 ], labels=[0.5, 1.0])
                axs[col_].set_yticks(ticks=[0.0 , 0.5, 1.0 ], labels=[0.0, 0.5, 1.0])
                # axs[col_].set_xticks([])
                # axs[col_].set_yticks([])


        # Add legend for parameter values
        legend_elements = []
        for val__, i in zip(list(set(self.final_model_params_ls)), range(len(list(set(self.final_model_params_ls))))):
            try:  # get color for parameter optimization looping
                color_ = param_col_ls_dict[val__]
            except:  # if not looping through parameters select color from list
                color_ = testing_set_plot_color
        #for val__, i in zip(self.param_values_to_loop_, range(len(self.param_values_to_loop_))):
            legend_elements.append(Line2D([0], [0],
                                          color=color_,
                                          lw=4, label=str(val__)))
        axs[-1].legend(handles=legend_elements, loc='upper left', frameon=False, bbox_to_anchor=(0, 1), title=self.parameter_to_optimize, title_fontsize=12, fontsize=12)
        axs[-1].axis('off')

        fig.suptitle('Compiled Precision-Recall Curves Final Models - Per Embedding ' + str(self.num_rerurun_model_building) +
                     ' rounds' + '\n' + self.output_run_file_info_string_.replace('_', ' ').replace(self.region_.replace('_', '-'), self.region_.replace('_', '-') + '\n'), fontsize=9)
        fig.tight_layout()  # NOTE: h and w (above in fig.set_size... MUST be large enough to accomodate legends or will be cut off/squished in output)


        # Export Precision-Recall curve numeric data as a .csv file
        # Pad all curves in all_prts_dict so all same length as longest_prt_
        all_prts_dict_for_df = {}
        for k__ in list(all_prts_dict.keys()):
            #padded_prts_ls = []
            for v, label_ in zip(all_prts_dict[k__],['precision','recall','thresholds']):
                amt_to_pad_ = longest_prt_ - len(v)  # [[ ],[ ], [ ]]
                v += [0] * amt_to_pad_
                all_prts_dict_for_df[label_ +'_'+k__] = v
                # padded_prts_ls.append(v)
            #all_prts_dict[k__] = padded_prts_ls
        all_prts_final_df = pd.DataFrame(all_prts_dict_for_df)

        # ** SAVE DATA **
        fnm_ = (self.output_directory + 'data/' + 'final_performance_metrics_p-r_' + str(self.num_rerurun_model_building) + '-rnds.csv')
        all_prts_final_df.to_csv(fnm_)
        logging.info('\n\nFinal Precision-Recall-Threshold curve data saved to: '+fnm_.replace(self.output_directory, '~/'))


        # ** SAVE FIGURE **
        plt.rcParams['svg.fonttype'] = 'none'  # exports text as strings rather than vector paths (images)
        fnm_ = (self.output_directory + 'figures/' + 'p-r_' + str(self.num_rerurun_model_building) + '-rnds_final')
        fnm_svg_ = (self.output_directory + 'figures/' + 'svg_figs/' + 'p-r_' + str(self.num_rerurun_model_building) + '-rnds_final')
        fig.savefig(fnm_svg_.split('.')[0] + '.svg', format='svg', transparent=True)
        fig.savefig(fnm_.split('.')[0] + '.png', format='png', dpi=300, transparent=False)
        logging.info('Figure saved to:'+str(fnm_) + '.png'.replace(self.output_directory, '~/'))
        return

    def plot_final_model_top_precision_recall_curves(self):
        logging.info("\nPlotting top precision recall curves from final models...")
        ## Plot Top 5 Final Model Precision-Recall curves as a single figure
        sup_title_id_info = ('' +  # str(num_rerurun_model_building*run_round_num)+' rounds'+'\n'+
                             self.output_run_file_info_string_.replace('_', ' ').replace(self.region_.replace('_', '-'), self.region_.replace('_', '-') + '\n'))

        pr_curves_dict_ = self.final_performance_curves_encodings_dict
        pr_curves_keys_ = self.final_key_ls

        param_col_ls = list(sns.color_palette("hls", len(self.param_values_to_loop_)).as_hex())
        greys_col_ls = list(sns.color_palette("Greys", len(self.param_values_to_loop_)).as_hex())

        # Create dictionary of parameter colors based on parameter
        param_col_ls_dict = {}
        for i in range(len(self.param_values_to_loop_)):
            param_col_ls_dict[self.param_values_to_loop_[i]] = param_col_ls[i]

        greys_col_ls_dict = {}
        for i in range(len(self.param_values_to_loop_)):
            greys_col_ls_dict[self.param_values_to_loop_[i]] = greys_col_ls[i]

        embd_color_dict = {}
        for e in self.feature_encoding_ls:
            embd_color_dict[e] = param_col_ls_dict


        # Plot a single PLOT for each embedding type
        fig, axs = plt.subplots(1,len(self.feature_encoding_ls )+ 1)
        fig.set_size_inches(w=3*(len(self.feature_encoding_ls )+ 1), h=4.5, )  # NOTE: h and w must be large enough to accomodate any legends


        for e, col_ in zip(self.feature_encoding_ls,list(range(len(self.feature_encoding_ls)))):  # different plots per embedding
            axs[col_].set_title(str(e))
            aucpr_by_round_dict = {}
            # First Loop through rounds to get top 5 curves by AUCPR (for a given embedding
            for n in range(self.num_rerurun_model_building):  # loop through rounds

                # Get p-r curves per round for given embedding and round
                key__ = str(e) + '_' + str(n)
                p = self.final_model_params_ls[n]
                # NOTE: no p-r curves for label-propagation/spreading with one-hot encoding
                if not (((self.parameter_to_optimize == 'model') and ('label' in p) and (e == 'one-hot')) or (
                        ('label' in self.model_type_) and (e == 'one-hot'))):

                    pcurve__ = self.final_performance_curves_encodings_dict[key__]['Precision_Recall_Curve'][0]
                    rcurve__ = self.final_performance_curves_encodings_dict[key__]['Precision_Recall_Curve'][1]
                    unach_pcurve__ = self.final_performance_curves_encodings_dict[key__]['Unacheivable_Region_Curve'][0]
                    unach_rcurve__ = self.final_performance_curves_encodings_dict[key__]['Unacheivable_Region_Curve'][1]
                    aucpr_ = metrics.auc(rcurve__, pcurve__)
                    aucpr_by_round_dict[n] = aucpr_

            # Sort aucpr_by_round_dict by values from largest aucpr to smallest
            aucpr_by_round_dict = {k: v for k, v in sorted(aucpr_by_round_dict.items(), key=lambda item: item[1])[::-1]}
            # get rounds (dictkeys) of top 5 performing models
            top_5_rnds_for_emb_ = list( aucpr_by_round_dict.keys())[:5]

            # Then plot just the top 5 curves
            for n in top_5_rnds_for_emb_:  # loop through rounds
                # Get p-r curves per round for given embedding and round
                key__ = str(e) + '_' + str(n)
                p = self.final_model_params_ls[n]
                try:  # get color for parameter optimization looping
                    color_ = param_col_ls_dict[p]
                except:  # if not looping through parameters select color from list
                    color_ = testing_set_plot_color
                # NOTE: no p-r curves for label-propagation/spreading with one-hot encoding
                if not (((self.parameter_to_optimize == 'model') and ('label' in p) and (e == 'one-hot')) or (
                        ('label' in self.model_type_) and (e == 'one-hot'))):
                    pcurve__ = self.final_performance_curves_encodings_dict[key__]['Precision_Recall_Curve'][0]
                    rcurve__ = self.final_performance_curves_encodings_dict[key__]['Precision_Recall_Curve'][1]
                    unach_pcurve__ = self.final_performance_curves_encodings_dict[key__]['Unacheivable_Region_Curve'][0]
                    unach_rcurve__ = self.final_performance_curves_encodings_dict[key__]['Unacheivable_Region_Curve'][1]
                    axs[col_].plot(
                        rcurve__,  # r_OH,# x
                        pcurve__,  # p_OH,# y
                        lw=1,
                        color=color_,
                    )

                    # axs[col_].plot(
                    #     rcurve__,  # r_OH,# x
                    #     unach_rcurve__,  # p_OH,# y
                    #     lw=1,
                    #     color=color_,
                    #     linestyle='dashed',
                    # )

            # Format Axes
            axs[col_].set_xlim(0, 1)
            axs[col_].set_ylim(0, 1.1)
            # axs[col_].set_xticks(ticks=np.arange(0, 1.1, .25), minor=True)
            axs[col_].tick_params(direction='in', which='both', length=3, width=1)

            if col_ == 0:
                axs[col_].set_xlabel('Recall')
                axs[col_].set_ylabel('Precision')
                axs[col_].set_xticks(ticks=[0.5, 1.0 ], labels=[0.5, 1.0])
                axs[col_].set_yticks(ticks=[0.0 , 0.5, 1.0 ], labels=[0.0, 0.5, 1.0])

            else:
                axs[col_].set_xticks(ticks=[0.5, 1.0 ], labels=[0.5, 1.0])
                axs[col_].set_yticks(ticks=[0.0 , 0.5, 1.0 ], labels=[0.0, 0.5, 1.0])
                # axs[col_].set_xticks([])
                # axs[col_].set_yticks([])

        # Add legend for parameter values
        legend_elements = []
        for val__, i in zip(list(set(self.final_model_params_ls)), range(len(list(set(self.final_model_params_ls))))):
        #for val__, i in zip(self.param_values_to_loop_, range(len(self.param_values_to_loop_))):
            try:  # get color for parameter optimization looping
                color_ = param_col_ls_dict[val__]
            except:  # if not looping through parameters select color from list
                color_ = testing_set_plot_color
            legend_elements.append(Line2D([0], [0],
                                          color=color_,  # embd_color_dict[embd_][val__],
                                          lw=4, label=str(val__)))
        axs[-1].legend(handles=legend_elements, loc='upper left', frameon=False, bbox_to_anchor=(0, 1), title=self.parameter_to_optimize, title_fontsize=12, fontsize=12)
        axs[-1].axis('off')

        fig.suptitle('Top 5 Precision-Recall Curves Final Models - Per Embedding ' + str(self.num_rerurun_model_building) +
                     ' rounds' + '\n' + self.output_run_file_info_string_.replace('_', ' ').replace(self.region_.replace('_', '-'), self.region_.replace('_', '-') + '\n'), fontsize=9)
        fig.tight_layout()  # NOTE: h and w (above in fig.set_size... MUST be large enough to accomodate legends or will be cut off/squished in output)
        # ** SAVE FIGURE **
        plt.rcParams['svg.fonttype'] = 'none'  # exports text as strings rather than vector paths (images)
        fnm_ = (self.output_directory + 'figures/' + 'p-r_' + str(self.num_rerurun_model_building) + '-rnds_top5_final')
        fnm_svg_ = (self.output_directory + 'figures/' + 'svg_figs/' + 'p-r_' + str(self.num_rerurun_model_building) + '-rnds_top5_final')
        fig.savefig(fnm_svg_.split('.')[0] + '.svg', format='svg', transparent=True)
        fig.savefig(fnm_.split('.')[0] + '.png', format='png', dpi=300, transparent=False)
        logging.info('Figure saved to:'+str(fnm_) + '.png'.replace(self.output_directory, '~/'))
        # return

    def autolabel_boxplot(self, medians, ax_, label_color = 'black'):
        '''to run: autolabel_boxplot(bplot1['medians']) '''
        for m_ in medians:
            height = m_.get_ydata()[0]
            right_coord = m_.get_xdata()[-1]
            ax_.annotate('{}'.format(np.round(height, 3)),
                        xy=(right_coord, height),
                        xytext=(3, 0), textcoords="offset points",  # 3 points horizontal offset
                        ha='left', va='center',
                        fontsize=8.5, color=label_color,
                        )

    def autolabel_boxplot_below(self, caps, medians, ax_, label_color = 'black'):
        '''to run: autolabel_boxplot_below(bplot1['caps'][::2], bplot1['medians'], ax) '''
        for c_,m_ in zip(caps,medians):
            height = c_.get_ydata()[0]
            height_label = m_.get_ydata()[0]
            right_coord = c_.get_xdata()[-1]
            left_coord = c_.get_xdata()[0]
            ax_.annotate(str(np.round(height_label, 2))[1:],
                         # xy = (left_coord+((left_coord-right_coord)/2) , height),
                         xy=(left_coord, height),
                         xytext=(-3, -3), textcoords="offset points",  # 3 points vertical and horizontal offset
                         ha='left', va='top',
                         fontsize=8.5, color=label_color,
                         )

    def plot_final_model_box_plots_per_param_val(self):
        logging.info("\nPlotting box plots for final models per parameter value...")
        if self.param_values_to_loop_ == []:
            logging.info("No parameter values to plot by")
            return
        ## Plot Compiled Multimetrics Model Performance per Param Val - Final Models

        # Each column of final_detailed_metric_df contains a single round for a single embedding type
        final_detailed_metric_df = pd.DataFrame(self.final_detailed_performance_metrics_encodings_dict)
        fnm_ = (self.output_directory + 'data/' + 'final_performance_metrics_' + str(self.num_rerurun_model_building) + '-rnds_final.csv')
        final_detailed_metric_df.to_csv(fnm_,index=True)
        logging.info("Final Models Performance Metrics Dataframe saved to:\n\t"+fnm_)

        flierprops__ = dict(marker='.', markerfacecolor='none', markersize=4, linewidth=0.1, markeredgecolor='black')  # linestyle='none',
        boxprops__ = dict(facecolor='none', linestyle='none', linewidth=1, edgecolor='k', )

        # one axis per performance metric
        # one box per embedding per axis
        # fig, axs = plt.subplots( len(self.feature_encoding_ls), len(final_detailed_metric_df))
        # fig.set_size_inches(w=14, h=3*len(self.feature_encoding_ls))



        # Update: Make plots in sets of two or fewer (so not too large)
        # TODO: Add +1 to num_plots_ for an extra plot with all embeddings overlayed (in different colors)
        num_rows_ = 10
        num_plots_ =  1#int(int(len(self.feature_encoding_ls)/num_rows_)+(len(self.feature_encoding_ls)%num_rows_))# + 1
        paired_feature_encoding_ls = [self.feature_encoding_ls[i:i + num_rows_] for i in range(0, len(self.feature_encoding_ls), num_rows_)]

        for plotn_ in list(range(num_plots_)):
            num_rows_actually_plotted_ = 0

            # if plotn_ == num_plots_ - 1:  # TODO: last plot is just 1 row since is a combined plot with all embeddings
            #     fig, axs = plt.subplots(1, len(final_detailed_metric_df))
            #     fig.set_size_inches(w=14, h=3 )
            # else:
            #     fig, axs = plt.subplots(len(paired_feature_encoding_ls[plotn_]), len(final_detailed_metric_df))
            #     fig.set_size_inches(w=14, h=3*len(paired_feature_encoding_ls[plotn_]))
            fig, axs = plt.subplots(num_rows_, len(final_detailed_metric_df))
            fig.set_size_inches(w=12, h=num_rows_ * 1.5)

            # Split Evaluation Metric Data per parameter value
            # Loop through all embeddings used
            #for embedding_type_final_eval_, j in zip(self.feature_encoding_ls,list(range(len(self.feature_encoding_ls)))):
            for embedding_type_final_eval_, j in zip(paired_feature_encoding_ls[plotn_], list(range(len(paired_feature_encoding_ls[plotn_])))):
                num_rows_actually_plotted_+=1

                # From final_detailed_metric_df get just columns for a single selected embedding type
                # Get just columns with selected embedding metric (embedding_type_final_eval_)
                cols_with_embd_ = [x for x in list(final_detailed_metric_df.columns) if embedding_type_final_eval_ in x]
                logging.info("Columns from final_detailed_metric_df with embedding = 'embedding_type_final_eval_' = "+
                      str(embedding_type_final_eval_)+" - "+str(len(cols_with_embd_))+' out of '+str(len(list(final_detailed_metric_df.columns))))

                # Get just columns with selected embedding metric (embedding_type_final_eval_)
                final_detailed_metric_one_embd_df = final_detailed_metric_df[cols_with_embd_]

                # Get parameter VALUES only for each selected embedding
                param_vals_one_embd_ = list(set([x.split(str(self.parameter_to_optimize)+'-')[-1].split('_round_')[0] for x in list(final_detailed_metric_one_embd_df.columns)]))

                # Get counts for each


                param_vals_one_embd_mult_ = list([x.split(str(self.parameter_to_optimize)+'-')[-1].split('_round_')[0] for x in list(final_detailed_metric_df.columns)])

                ct_param_vals_one_embd_mult_ = Counter(param_vals_one_embd_mult_)
                x_labs_with_counts_ = [str(x) + '\n(' + str(ct_param_vals_one_embd_mult_[x]) + ')' for x in param_vals_one_embd_mult_]
                x_labs_with_counts_ = list(set(x_labs_with_counts_))
                # If Parameter values are integers, order them from smallest to larget
                try:
                    int_param_vals_one_embd_ = [int(x) for x in param_vals_one_embd_]
                    int_param_vals_one_embd_.sort()
                    param_vals_one_embd_ = [str(x) for x in int_param_vals_one_embd_]
                except:
                    pass

                metrics_ls = list(final_detailed_metric_one_embd_df.index)

                final_model_ct_per_param_val_dict = {} # holds counts of each parameter value used to build a final model
                for v__ in self.param_values_to_loop_:
                    ct_v__ = len([x for x in self.final_model_params_ls if str(x) == str(v__)])
                    final_model_ct_per_param_val_dict[str(v__)] = ct_v__

                for i in range(len(metrics_ls)):
                    metric_ = metrics_ls[i]
                    # Plot by parameter value
                    # data_ = [list(final_detailed_metric_one_embd_df[
                    #                   [embedding_type_final_eval_+'-'+str(self.parameter_to_optimize)+'-'+param_val_ + '_round_' + str(i) for i in
                    #                    list(range(final_model_ct_per_param_val_dict[str(param_val_)]))]].transpose()[metric_]) for param_val_ in param_vals_one_embd_]
                    # get columns for single parameter value and metric

                    cols_to_select_ = []
                    for param_val_ in param_vals_one_embd_:
                        # Get rounds for given parameter value and embedding
                        rounds_per_one_param_val_ = [int(x.split('_round_')[-1]) for x in list(final_detailed_metric_one_embd_df.columns) if embedding_type_final_eval_ + '-' + str(self.parameter_to_optimize) + '-' + param_val_ + '_round_' in x]
                        for rnd__ in rounds_per_one_param_val_:
                            cols_to_select_.append(embedding_type_final_eval_+'-'+str(self.parameter_to_optimize)+'-'+param_val_ +'_round_'+str(rnd__))

                    data_ = [list(final_detailed_metric_one_embd_df[cols_to_select_].transpose()[metric_]) for param_val_ in param_vals_one_embd_]
                    #logging.info(data_)
                    # Multiple Rows
                    # try:
                    bplot1 = axs[j,i].boxplot(
                        data_,
                        vert=True,  # vertical box alignment
                        patch_artist=True,  # fill with color
                        labels=param_vals_one_embd_,
                        flierprops=flierprops__, boxprops=boxprops__,
                        capprops=dict(color='black'),
                        whiskerprops=dict(color='black'),

                    )  # will be used to label x-ticks
                    axs[j,i].set_title(metric_.replace('beta',str(self.f_beta_)), fontsize=8.5 )
                    if i == 3:
                        #if embedding_type_final_eval_ == self.feature_encoding_ls[0]:# for first row of plots in figure
                        if embedding_type_final_eval_ == paired_feature_encoding_ls[0]:  # for first row of plots in figure
                            axs[j,i].set_title('Plot '+str(plotn_+1)+' / '+str(num_plots_)+' Final Model Performances (' + str(self.num_rerurun_model_building) + ' Rounds)\n' + str(metric_), fontsize=8.5 )  # ,fontweight='bold')
                        else:
                            axs[j,i].set_title(str(embedding_type_final_eval_)+'\n' + str(metric_), fontsize=8.5 )  # ,fontweight='bold')

                    # update x-axis labels
                    #axs[j,i].set_xticklabels(param_vals_one_embd_, rotation=0, fontsize=8.5)
                    # axs[j, i].set_xticklabels(x_labs_with_counts_, rotation=0, fontsize=8.5)
                    axs[j,i].set_xlabel(str(self.parameter_to_optimize),fontsize=8.5 )


                    if metric_ == 'MCC':
                        axs[j,i].set_ylim(-1, 1)
                        axs[j,i].set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
                        axs[j,i].tick_params(axis='y', labelsize=7)
                    else:
                        axs[j,i].set_ylim(0, 1)
                        axs[j,i].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                        axs[j,i].tick_params(axis='y', labelsize=7)

                    # # Single Row
                    # except:
                    #     bplot1 = axs[i].boxplot(
                    #         data_,
                    #         vert=True,  # vertical box alignment
                    #         patch_artist=True,  # fill with color
                    #         labels=param_vals_one_embd_,
                    #         flierprops=flierprops__, boxprops=boxprops__,
                    #         capprops=dict(color='black'),
                    #         whiskerprops=dict(color='black'),
                    #
                    #     )  # will be used to label x-ticks
                    #     axs[i].set_title(metric_.replace('beta',str(self.f_beta_)), fontsize=8.5 )
                    #     if i == 3:
                    #         # if embedding_type_final_eval_ == self.feature_encoding_ls[0]:# for first row of plots in figure
                    #         if embedding_type_final_eval_ == paired_feature_encoding_ls[0]:  # for first row of plots in figure
                    #             axs[i].set_title('Plot ' + str(plotn_ + 1) + ' / ' + str(num_plots_) + ' Final Model Performances (' + str(self.num_rerurun_model_building) + ' Rounds)\n' + str(metric_) , fontsize=8.5 )  # ,fontweight='bold')
                    #         else:
                    #             axs[i].set_title(str(embedding_type_final_eval_) + '\n' + str(metric_), fontsize=8.5 )  # ,fontweight='bold')
                    #
                    #     # update x-axis labels
                    #     # axs[i].set_xticklabels(param_vals_one_embd_, rotation=0, fontsize=8.5)
                    #     #axs[i].set_xticklabels(x_labs_with_counts_, rotation=0, fontsize=8.5)
                    #     axs[i].set_xlabel(str(self.parameter_to_optimize), fontsize=8.5 )
                    #
                    #     # label metric score values next to each box
                    #     #self.autolabel_boxplot(bplot1['medians'], axs[i], label_color='black')
                    #     #*# self.autolabel_boxplot_below(bplot1['caps'][::2], bplot1['medians'], axs[i])
                    #
                    #     if metric_ == 'MCC':
                    #         axs[i].set_ylim(-1, 1)
                    #         axs[i].set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
                    #         axs[i].tick_params(axis='y', labelsize=7)
                    #     else:
                    #         axs[i].set_ylim(0, 1)
                    #         axs[i].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                    #         axs[i].tick_params(axis='y', labelsize=7)
                    # # try:
                    #     # label metric score values next to each box
                    #     #self.autolabel_boxplot(bplot1['medians'], axs[j,i], label_color = 'black')
                    #     #*# self.autolabel_boxplot_below(bplot1['caps'][::2], bplot1['medians'], axs[j, i])
                    # except:
                    #     pass # already labeled above

            #if plotn_ == num_plots_ - 1:  # TODO: last plot needs a legend (to identify each embedding as a different color)

            # hide axes if nothing plotted
            num_rows_to_erase_ = num_rows_ - num_rows_actually_plotted_
            for j in range(num_rows_to_erase_):
                for i in range(len(final_detailed_metric_df)):
                    axs[::-1][j, i].set_visible(False)  # reverse because works from bottom up (backwards)

            fig.suptitle(str(plotn_+1)+' Compiled Multiple Metrics Final Models - Per Parameter Value ' + str(self.num_rerurun_model_building) +
                         ' rounds' + '\n' + self.output_run_file_info_string_.replace('_', ' ').replace(self.region_.replace('_', '-'), self.region_.replace('_', '-') + '\n'), fontsize=9)
            fig.tight_layout()

            # ** SAVE FIGURE **
            plt.rcParams['svg.fonttype'] = 'none'  # exports text as strings rather than vector paths (images)
            fnm_ = (self.output_directory + 'figures/' + str(plotn_+1)+'_final_bxp_per-param-val_' + str(self.num_rerurun_model_building) + '-rnds')
            fnm_svg_ = (self.output_directory + 'figures/' + 'svg_figs/' + str(plotn_+1)+'_final_bxp_per-param-val_' + str(self.num_rerurun_model_building) + '-rnds')
            fig.savefig(fnm_svg_.split('.')[0] + '.svg', format='svg', transparent=True)
            fig.savefig(fnm_.split('.')[0] + '.png', format='png', dpi=300, transparent=False)
            logging.info('Figure saved to:'+str(fnm_) + '.png'.replace(self.output_directory, '~/'))
        return


    def plot_final_model_box_plots_per_metric(self):
        logging.info("\nPlotting box plots -- per performance metric -- for final models...")
        ## Plot Compiled Multimetrics Model Performance - Final Models per metric
        final_metric_df = pd.DataFrame(self.final_performance_metrics_encodings_dict)

        metrics_ls = list(final_metric_df.index)
        enc_ls_ = self.feature_encoding_ls

        flierprops__ = dict(marker='.', markerfacecolor='none', markersize=4, linewidth=0.1, markeredgecolor='black')  # linestyle='none',
        boxprops__ = dict(facecolor='none', linestyle='none', linewidth=1, edgecolor='k', )
        medianprops__ = dict(linewidth=2, color=testing_set_plot_color)

        # one axis per performance metric
        # one box per embedding per axis
        fig, axs = plt.subplots(1, len(final_metric_df))
        fig.set_size_inches(w=12, h=3)
        fntsz_ = 7
        for i in range(len(metrics_ls)):
            metric_ = metrics_ls[i]

            data_ = [list(final_metric_df[[enc_ + '_' + str(i) for i in list(range(self.num_rerurun_model_building)) ]].transpose()[metric_]) for enc_ in enc_ls_]
            # data_ = [list(final_metric_df[[enc_ + '_' + str(i) for i in [0, 1]                                       ]].transpose()[metric_]) for enc_ in enc_ls_]

            bplot1 = axs[i].boxplot(
                data_,
                vert=True,  # vertical box alignment
                patch_artist=True,  # fill with color
                labels=enc_ls_,
                flierprops=flierprops__, boxprops=boxprops__, medianprops=medianprops__,
                capprops=dict(color='black'),
                whiskerprops=dict(color='black'),

            )  # will be used to label x-ticks
            axs[i].set_title(metric_.replace('beta',str(self.f_beta_)),fontsize = fntsz_)
            if i ==2:
                axs[i].set_title('Final Model Performances (' + str(self.num_rerurun_model_building) + ' Rounds)\n' + str(metric_), fontsize= fntsz_)  # ,fontweight='bold')

            # update x-axis labels
            axs[i].set_xticklabels([feature_encodings_dict[x] for x in self.feature_encoding_ls], fontsize = fntsz_, rotation=90)

            if metric_ == 'MCC':
                axs[i].set_ylim(-1, 1)
                axs[i].set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
                axs[i].tick_params(axis='y', labelsize=fntsz_)
            else:
                axs[i].set_ylim(0, 1)
                axs[i].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                axs[i].tick_params(axis='y', labelsize=fntsz_)


            #*# self.autolabel_boxplot_below(bplot1['caps'][::2],bplot1['medians'], axs[i])

        fig.suptitle('Compiled Multiple Metrics Final Models '+str(self.num_rerurun_model_building)+
                     ' rounds' +'\n'+self.output_run_file_info_string_.replace('_',' ').replace(self.region_.replace('_','-'),self.region_.replace('_','-')+'\n'),fontsize=9)

        fig.tight_layout()


        # ** SAVE FIGURE **
        plt.rcParams['svg.fonttype'] = 'none' # exports text as strings rather than vector paths (images)
        fnm_ =     (self.output_directory+ 'figures/'+           'bxp_'+str(self.num_rerurun_model_building)+'-rnds_final')
        fnm_svg_ = (self.output_directory+'figures/'+'svg_figs/'+'bxp_'+str(self.num_rerurun_model_building)+'-rnds_final')
        fig.savefig(fnm_svg_.split('.')[0]+'.svg',format='svg',transparent=True)
        fig.savefig(fnm_.split('.')[0]+'.png',format='png',dpi=300,transparent=False)
        logging.info('Figure saved to: '+ (fnm_+'.png').replace(self.output_directory,'~/'))
        return

    def plot_final_model_precision_recall_curves_on_ext_dataset(self):
        logging.info("\nPlotting precision-recall curves for final models evaluated on external dataset...")
        if not self.apply_final_models_to_external_dataset_:
            logging.info("apply_final_models_to_external_dataset_ is set to False so did not evaluate on an external dataset")
            return

        ## Plot Final Model Precision-Recall curves as a single figure
        sup_title_id_info = ('' +  # str(num_rerurun_model_building*run_round_num)+' rounds'+'\n'+
                             self.output_run_file_info_string_.replace('_', ' ').replace(self.region_.replace('_', '-'), self.region_.replace('_', '-') + '\n'))

        pr_curves_dict_ = self.ext_final_performance_curves_encodings_dict
        pr_curves_keys_ = self.final_key_ls


        # Create dictionary of parameter colors based on parameter
        param_col_ls = list(sns.color_palette("hls", len(self.param_values_to_loop_)).as_hex())
        greys_col_ls = list(sns.color_palette("Greys", len(self.param_values_to_loop_)).as_hex())

        param_col_ls_dict = {}
        for i in range(len(self.param_values_to_loop_)):
            param_col_ls_dict[self.param_values_to_loop_[i]] = param_col_ls[i]

        greys_col_ls_dict = {}
        for i in range(len(self.param_values_to_loop_)):
            greys_col_ls_dict[self.param_values_to_loop_[i]] = greys_col_ls[i]

        embd_color_dict = {}
        for e in self.feature_encoding_ls:
            embd_color_dict[e] = param_col_ls_dict


        # Plot a single PLOT for each embedding type
        fig, axs = plt.subplots(1,len(self.feature_encoding_ls )+ 1)
        fig.set_size_inches(w=3*(len(self.feature_encoding_ls )+ 1), h=4.5, )  # NOTE: h and w must be large enough to accomodate any legends

        all_prts_dict = {}
        longest_prt_ = -1
        for e, col_ in zip(self.feature_encoding_ls,list(range(len(self.feature_encoding_ls)))):  # different plots per embedding
            axs[col_].set_title(str(e))
            for n in range(self.num_rerurun_model_building):  # loop through rounds

                # Get p-r curves per round for given embedding and round
                key__ = str(e) + '_' + str(n)
                p = self.final_model_params_ls[n]

                try: # get color for parameter optimization looping
                    color_ = param_col_ls_dict[p]
                except: # if not looping through parameters select color from list
                    color_ = external_set_plot_color

                # NOTE: no p-r curves for label-propagation/spreading with one-hot encoding
                if not (((self.parameter_to_optimize == 'model') and ('label' in p) and (e == 'one-hot')) or (
                        ('label' in self.model_type_) and (e == 'one-hot'))):

                    pcurve__ = self.ext_final_performance_curves_encodings_dict[key__]['Precision_Recall_Curve'][0]
                    rcurve__ = self.ext_final_performance_curves_encodings_dict[key__]['Precision_Recall_Curve'][1]
                    thresholds__ = self.ext_final_performance_curves_encodings_dict[key__]['Precision_Recall_Curve'][1]
                    unach_pcurve__ = self.ext_final_performance_curves_encodings_dict[key__]['Unacheivable_Region_Curve'][0]
                    unach_rcurve__ = self.ext_final_performance_curves_encodings_dict[key__]['Unacheivable_Region_Curve'][1]

                    # For exporting Precision-Recall curve numeric data as a .csv file
                    prt_ls = [list(pcurve__), list(rcurve__), list(thresholds__)]
                    all_prts_dict[str(n) + '_' + key__] = prt_ls
                    for v in prt_ls:
                        if longest_prt_ < len(v):  # keep track of longest Precision, Recall, Threshold curve list to pad later
                            longest_prt_ = len(v)


                    axs[col_].plot(
                        rcurve__,  # r_OH,# x
                        pcurve__,  # p_OH,# y
                        lw=1,
                        color= color_,
                    )


                    # axs[col_].plot(
                    #     rcurve__,  # r_OH,# x
                    #     unach_rcurve__,  # p_OH,# y
                    #     lw=1,
                    #     color=color_,
                    #     linestyle='dashed',
                    # )

            # Format Axes
            axs[col_].set_xlim(0, 1)
            axs[col_].set_ylim(0, 1.1)
            # axs[col_].set_xticks(ticks=np.arange(0, 1.1, .25), minor=True)
            axs[col_].tick_params(direction='in', which='both', length=3, width=1)

            if col_ == 0:
                axs[col_].set_xlabel('Recall')
                axs[col_].set_ylabel('Precision')
                axs[col_].set_xticks(ticks=[0.5, 1.0 ], labels=[0.5, 1.0])
                axs[col_].set_yticks(ticks=[0.0 , 0.5, 1.0 ], labels=[0.0, 0.5, 1.0])

            else:
                # axs[col_].set_xticks([])
                # axs[col_].set_yticks([])
                axs[col_].set_xticks(ticks=[0.5, 1.0 ], labels=[0.5, 1.0])
                axs[col_].set_yticks(ticks=[0.0 , 0.5, 1.0 ], labels=[0.0, 0.5, 1.0])

            # Add colored border
            # for spine in axs[col_].spines.values():
            #     spine.set_edgecolor('red')
            #     spine.set_linewidth(3)


        # Export Precision-Recall curve numeric data as a .csv file
        # Pad all curves in all_prts_dict so all same length as longest_prt_
        all_prts_dict_for_df = {}
        for k__ in list(all_prts_dict.keys()):
            # padded_prts_ls = []
            for v, label_ in zip(all_prts_dict[k__], ['precision', 'recall', 'thresholds']):
                amt_to_pad_ = longest_prt_ - len(v)  # [[ ],[ ], [ ]]
                v += [0] * amt_to_pad_
                all_prts_dict_for_df[label_ + '_' + k__] = v
                # padded_prts_ls.append(v)
            # all_prts_dict[k__] = padded_prts_ls
        all_prts_final_df = pd.DataFrame(all_prts_dict_for_df)

        # ** SAVE DATA **
        fnm_ = (self.output_directory + 'data/' + 'external_eval_final_p-r-curve_' + str(self.num_rerurun_model_building) + '-rnds_ext-data-eval.csv')
        all_prts_final_df.to_csv(fnm_)
        logging.info('\n\nFinal model Applied to External Dataset Precision-Recall-Threshold curve data saved to: '+fnm_.replace(self.output_directory, '~/'))


        # Add legend for parameter values
        legend_elements = []
        for val__, i in zip(list(set(self.final_model_params_ls)), range(len(list(set(self.final_model_params_ls))))):
            try:  # get color for parameter optimization looping
                color_ = param_col_ls_dict[val__]
            except:  # if not looping through parameters select color from list
                color_ = external_set_plot_color
        #for val__, i in zip(self.param_values_to_loop_, range(len(self.param_values_to_loop_))):
            legend_elements.append(Line2D([0], [0],
                                          color=color_,  # embd_color_dict[embd_][val__],
                                          lw=4, label=str(val__)))
        axs[-1].legend(handles=legend_elements, loc='upper left', frameon=False, bbox_to_anchor=(0, 1), title=self.parameter_to_optimize, title_fontsize=12, fontsize=12)
        axs[-1].axis('off')



        random_flag_ = ''
        if self.randomize_ext_data_:
            random_flag_ = '(Randomized) '
        fig.suptitle('Compiled Precision-Recall Curves Final Models Evaluated on External Dataset '+random_flag_+'- Per Embedding ' + str(self.num_rerurun_model_building) +
                     ' rounds' + '\n' + self.output_run_file_info_string_.replace('_', ' ').replace(self.region_.replace('_', '-'), self.region_.replace('_', '-')) + '\n'+
                     'External Dataset: '+self.external_data_file_, fontsize=9)
        fig.tight_layout()  # NOTE: h and w (above in fig.set_size... MUST be large enough to accomodate legends or will be cut off/squished in output)
        # ** SAVE FIGURE **
        plt.rcParams['svg.fonttype'] = 'none'  # exports text as strings rather than vector paths (images)
        fnm_ = (self.output_directory + 'figures/ext_data/' + 'p-r_' + str(self.num_rerurun_model_building) + '-rnds_final_ext-data-eval')
        fnm_svg_ = (self.output_directory + 'figures/ext_data/' + 'svg_figs/' + 'p-r_' + str(self.num_rerurun_model_building) + '-rnds_final_ext-data-eval')
        fig.savefig(fnm_svg_.split('.')[0] + '.svg', format='svg', transparent=True)
        fig.savefig(fnm_.split('.')[0] + '.png', format='png', dpi=300, transparent=False)
        logging.info('Figure saved to:'+str(fnm_) + '.png'.replace(self.output_directory, '~/'))
        return

    def plot_final_model_top_precision_recall_curves_on_ext_dataset(self):
        logging.info("\nPlotting top precision recall curves from final models evaluated on external dataset...")
        if not self.apply_final_models_to_external_dataset_:
            logging.info("apply_final_models_to_external_dataset_ is set to False so did not evaluate on an external dataset")
            return

        ## Plot Top 5 Final Model Precision-Recall curves as a single figure
        sup_title_id_info = ('' +  # str(num_rerurun_model_building*run_round_num)+' rounds'+'\n'+
                             self.output_run_file_info_string_.replace('_', ' ').replace(self.region_.replace('_', '-'), self.region_.replace('_', '-') + '\n'))

        pr_curves_dict_ = self.ext_final_performance_curves_encodings_dict
        pr_curves_keys_ = self.final_key_ls

        param_col_ls = list(sns.color_palette("hls", len(self.param_values_to_loop_)).as_hex())
        greys_col_ls = list(sns.color_palette("Greys", len(self.param_values_to_loop_)).as_hex())

        # Create dictionary of parameter colors based on parameter
        param_col_ls_dict = {}
        for i in range(len(self.param_values_to_loop_)):
            param_col_ls_dict[self.param_values_to_loop_[i]] = param_col_ls[i]

        greys_col_ls_dict = {}
        for i in range(len(self.param_values_to_loop_)):
            greys_col_ls_dict[self.param_values_to_loop_[i]] = greys_col_ls[i]

        embd_color_dict = {}
        for e in self.feature_encoding_ls:
            embd_color_dict[e] = param_col_ls_dict


        # Plot a single PLOT for each embedding type
        fig, axs = plt.subplots(1,len(self.feature_encoding_ls )+ 1)
        fig.set_size_inches(w=3*(len(self.feature_encoding_ls )+ 1), h=4.5, )  # NOTE: h and w must be large enough to accomodate any legends


        for e, col_ in zip(self.feature_encoding_ls,list(range(len(self.feature_encoding_ls)))):  # different plots per embedding
            axs[col_].set_title(str(e))
            aucpr_by_round_dict = {}
            # First Loop through rounds to get top 5 curves by AUCPR (for a given embedding
            for n in range(self.num_rerurun_model_building):  # loop through rounds

                # Get p-r curves per round for given embedding and round
                key__ = str(e) + '_' + str(n)
                p = self.final_model_params_ls[n]
                # NOTE: no p-r curves for label-propagation/spreading with one-hot encoding
                if not (((self.parameter_to_optimize == 'model') and ('label' in p) and (e == 'one-hot')) or (
                        ('label' in self.model_type_) and (e == 'one-hot'))):

                    pcurve__ = self.ext_final_performance_curves_encodings_dict[key__]['Precision_Recall_Curve'][0]
                    rcurve__ = self.ext_final_performance_curves_encodings_dict[key__]['Precision_Recall_Curve'][1]
                    unach_pcurve__ = self.ext_final_performance_curves_encodings_dict[key__]['Unacheivable_Region_Curve'][0]
                    unach_rcurve__ = self.ext_final_performance_curves_encodings_dict[key__]['Unacheivable_Region_Curve'][1]
                    aucpr_ = metrics.auc(rcurve__, pcurve__)
                    aucpr_by_round_dict[n] = aucpr_

            # Sort aucpr_by_round_dict by values from largest aucpr to smallest
            aucpr_by_round_dict = {k: v for k, v in sorted(aucpr_by_round_dict.items(), key=lambda item: item[1])[::-1]}
            # get rounds (dictkeys) of top 5 performing models
            top_5_rnds_for_emb_ = list( aucpr_by_round_dict.keys())[:5]

            # Then plot just the top 5 curves
            for n in top_5_rnds_for_emb_:  # loop through rounds
                # Get p-r curves per round for given embedding and round
                key__ = str(e) + '_' + str(n)
                p = self.final_model_params_ls[n]
                try:  # get color for parameter optimization looping
                    color_ = param_col_ls_dict[p]
                except:  # if not looping through parameters select color from list
                    color_ = external_set_plot_color
                # NOTE: no p-r curves for label-propagation/spreading with one-hot encoding
                if not (((self.parameter_to_optimize == 'model') and ('label' in p) and (e == 'one-hot')) or (
                        ('label' in self.model_type_) and (e == 'one-hot'))):
                    pcurve__ = self.ext_final_performance_curves_encodings_dict[key__]['Precision_Recall_Curve'][0]
                    rcurve__ = self.ext_final_performance_curves_encodings_dict[key__]['Precision_Recall_Curve'][1]
                    unach_pcurve__ = self.ext_final_performance_curves_encodings_dict[key__]['Unacheivable_Region_Curve'][0]
                    unach_rcurve__ = self.ext_final_performance_curves_encodings_dict[key__]['Unacheivable_Region_Curve'][1]
                    axs[col_].plot(
                        rcurve__,  # r_OH,# x
                        pcurve__,  # p_OH,# y
                        lw=1,
                        color=color_,
                    )

                    # axs[col_].plot(
                    #     rcurve__,  # r_OH,# x
                    #     unach_rcurve__,  # p_OH,# y
                    #     lw=1,
                    #     color=color_,
                    #     linestyle='dashed',
                    # )

            # Format Axes
            axs[col_].set_xlim(0, 1)
            axs[col_].set_ylim(0, 1.1)
            # axs[col_].set_xticks(ticks=np.arange(0, 1.1, .25), minor=True)
            axs[col_].tick_params(direction='in', which='both', length=3, width=1)

            if col_ == 0:
                axs[col_].set_xlabel('Recall')
                axs[col_].set_ylabel('Precision')
                axs[col_].set_xticks(ticks=[0.5, 1.0 ], labels=[0.5, 1.0])
                axs[col_].set_yticks(ticks=[0.0 , 0.5, 1.0 ], labels=[0.0, 0.5, 1.0])

            else:
                # axs[col_].set_xticks([])
                # axs[col_].set_yticks([])
                axs[col_].set_xticks(ticks=[0.5, 1.0 ], labels=[0.5, 1.0])
                axs[col_].set_yticks(ticks=[0.0 , 0.5, 1.0 ], labels=[0.0, 0.5, 1.0])

            # Add colored border
            # for spine in axs[col_].spines.values():
            #     spine.set_edgecolor('red')
            #     spine.set_linewidth(3)

        # Add legend for parameter values
        legend_elements = []
        for val__, i in zip(list(set(self.final_model_params_ls)), range(len(list(set(self.final_model_params_ls))))):
        #for val__, i in zip(self.param_values_to_loop_, range(len(self.param_values_to_loop_))):
            try:  # get color for parameter optimization looping
                color_ = param_col_ls_dict[val__]
            except:  # if not looping through parameters select color from list
                color_ = external_set_plot_color
            legend_elements.append(Line2D([0], [0],
                                          color=color_,  # embd_color_dict[embd_][val__],
                                          lw=4, label=str(val__)))
        axs[-1].legend(handles=legend_elements, loc='upper left', frameon=False, bbox_to_anchor=(0, 1), title=self.parameter_to_optimize, title_fontsize=12, fontsize=12)
        axs[-1].axis('off')

        random_flag_ = ''
        if self.randomize_ext_data_:
            random_flag_ = '(Randomized) '

        fig.suptitle('Top 5 Precision-Recall Curves Final Models Evaluated on External Dataset '+random_flag_+'- Per Embedding ' + str(self.num_rerurun_model_building) +
                     ' rounds' + '\n' + self.output_run_file_info_string_.replace('_', ' ').replace(self.region_.replace('_', '-'), self.region_.replace('_', '-')) + '\n' + 'External Dataset: '+self.external_data_file_, fontsize=9)

        fig.tight_layout()  # NOTE: h and w (above in fig.set_size... MUST be large enough to accomodate legends or will be cut off/squished in output)
        # ** SAVE FIGURE **
        plt.rcParams['svg.fonttype'] = 'none'  # exports text as strings rather than vector paths (images)
        fnm_ = (self.output_directory + 'figures/ext_data/' + 'p-r_' + str(self.num_rerurun_model_building) + '-rnds_top5_final_ext-data-eval')
        fnm_svg_ = (self.output_directory + 'figures/ext_data/' + 'svg_figs/' + 'p-r_' + str(self.num_rerurun_model_building) + '-rnds_top5_final_ext-data-eval')
        fig.savefig(fnm_svg_.split('.')[0] + '.svg', format='svg', transparent=True)
        fig.savefig(fnm_.split('.')[0] + '.png', format='png', dpi=300, transparent=False)
        logging.info('Figure saved to:'+str(fnm_) + '.png'.replace(self.output_directory, '~/'))
        # return


    def plot_final_model_box_plots_per_param_val_on_ext_dataset(self):
        logging.info("\nPlotting box plots for final models evaluated on external dataset per parameter value...")
        if not self.apply_final_models_to_external_dataset_:
            logging.info("apply_final_models_to_external_dataset_ is set to False so did not evaluate on an external dataset")
            return

        ## Plot Compiled Multimetrics Model Performance per Param Val - Final Models

        # Each column of final_detailed_metric_df contains a single round for a single embedding type
        final_detailed_metric_df = pd.DataFrame(self.ext_final_detailed_performance_metrics_encodings_dict)
        fnm_ = (self.output_directory + 'data/' + 'performance_metrics_' + str(self.num_rerurun_model_building) + '-rnds_final_ext-data-eval.csv')
        final_detailed_metric_df.to_csv(fnm_,index=True)
        logging.info("Final Models Performance Metrics (when Evaluated on External Dataset) Dataframe saved to:\n\t"+fnm_)

        flierprops__ = dict(marker='.', markerfacecolor='none', markersize=4, linewidth=0.1, markeredgecolor='black')  # linestyle='none',
        boxprops__ = dict(facecolor='none', linestyle='none', linewidth=1, edgecolor='k')
        medianprops__ = dict(linewidth=2, color= external_set_plot_color)

        # one axis per performance metric
        # one box per embedding per axis
        # fig, axs = plt.subplots( len(self.feature_encoding_ls), len(final_detailed_metric_df))
        # fig.set_size_inches(w=14, h=3*len(self.feature_encoding_ls))
        # Update: Make plots in sets of two or fewer (so not too large)
        # num_plots_ =  int(int(len(self.feature_encoding_ls)/2)+(len(self.feature_encoding_ls)%2))
        # paired_feature_encoding_ls = [self.feature_encoding_ls[i:i + 2] for i in range(0, len(self.feature_encoding_ls), 2)]

        num_plots_ = int(int(len(self.feature_encoding_ls) / 3) + (len(self.feature_encoding_ls) % 3))
        paired_feature_encoding_ls = [self.feature_encoding_ls[i:i + 3] for i in range(0, len(self.feature_encoding_ls), 3)]

        for plotn_ in list(range(num_plots_)):

            fig, axs = plt.subplots(len(paired_feature_encoding_ls[plotn_]), len(final_detailed_metric_df))
            fig.set_size_inches(w=14, h=3 * len(paired_feature_encoding_ls[plotn_]))


            # Split Evaluation Metric Data per parameter value
            # Loop through all embeddings used
            #for embedding_type_final_eval_, j in zip(self.feature_encoding_ls,list(range(len(self.feature_encoding_ls)))):
            for embedding_type_final_eval_, j in zip(paired_feature_encoding_ls[plotn_], list(range(len(paired_feature_encoding_ls[plotn_])))):

                # From final_detailed_metric_df get just columns for a single selected embedding type
                # Get just columns with selected embedding metric (embedding_type_final_eval_)
                cols_with_embd_ = [x for x in list(final_detailed_metric_df.columns) if embedding_type_final_eval_ in x]
                logging.info("Columns from final_detailed_metric_df with embedding = 'embedding_type_final_eval_' = "+
                      str(embedding_type_final_eval_)+" - "+str(len(cols_with_embd_))+' out of '+str(len(list(final_detailed_metric_df.columns))))

                # Get just columns with selected embedding metric (embedding_type_final_eval_)
                final_detailed_metric_one_embd_df = final_detailed_metric_df[cols_with_embd_]

                # Get parameter VALUES only for each selected embedding
                param_vals_one_embd_ = list(set([x.split(str(self.parameter_to_optimize)+'-')[-1].split('_round_')[0] for x in list(final_detailed_metric_one_embd_df.columns)]))

                # If Parameter values are integers, order them from smallest to larget
                try:
                    int_param_vals_one_embd_ = [int(x) for x in param_vals_one_embd_]
                    int_param_vals_one_embd_.sort()
                    param_vals_one_embd_ = [str(x) for x in int_param_vals_one_embd_]
                except:
                    pass

                metrics_ls = list(final_detailed_metric_one_embd_df.index)

                final_model_ct_per_param_val_dict = {} # holds counts of each parameter value used to build a final model
                for v__ in self.param_values_to_loop_:
                    ct_v__ = len([x for x in self.final_model_params_ls if str(x) == str(v__)])
                    final_model_ct_per_param_val_dict[str(v__)] = ct_v__

                for i in range(len(metrics_ls)):
                    metric_ = metrics_ls[i]
                    # Plot by parameter value
                    # data_ = [list(final_detailed_metric_one_embd_df[
                    #                   [embedding_type_final_eval_+'-'+str(self.parameter_to_optimize)+'-'+param_val_ + '_round_' + str(i) for i in
                    #                    list(range(final_model_ct_per_param_val_dict[str(param_val_)]))]].transpose()[metric_]) for param_val_ in param_vals_one_embd_]
                    # get columns for single parameter value and metric

                    cols_to_select_ = []
                    for param_val_ in param_vals_one_embd_:
                        # Get rounds for given parameter value and embedding
                        rounds_per_one_param_val_ = [int(x.split('_round_')[-1]) for x in list(final_detailed_metric_one_embd_df.columns) if embedding_type_final_eval_ + '-' + str(self.parameter_to_optimize) + '-' + param_val_ + '_round_' in x]
                        for rnd__ in rounds_per_one_param_val_:
                            cols_to_select_.append(embedding_type_final_eval_+'-'+str(self.parameter_to_optimize)+'-'+param_val_ +'_round_'+str(rnd__))

                    data_ = [list(final_detailed_metric_one_embd_df[cols_to_select_].transpose()[metric_]) for param_val_ in param_vals_one_embd_]
                    #logging.info(data_)
                    # Multiple Rows
                    try:
                        bplot1 = axs[j,i].boxplot(
                            data_,
                            vert=True,  # vertical box alignment
                            patch_artist=True,  # fill with color
                            labels=param_vals_one_embd_,
                            flierprops=flierprops__, boxprops=boxprops__, medianprops=medianprops__,
                            capprops=dict(color='black'),
                            whiskerprops=dict(color='black'),

                        )  # will be used to label x-ticks
                        axs[j,i].set_title(metric_.replace('beta',str(self.f_beta_)), fontsize=8.5 )
                        if i == 3:
                            #if embedding_type_final_eval_ == self.feature_encoding_ls[0]:# for first row of plots in figure
                            if embedding_type_final_eval_ == paired_feature_encoding_ls[0]:  # for first row of plots in figure
                                axs[j,i].set_title('Plot '+str(plotn_+1)+' / '+str(num_plots_)+' Final Model Performances (' + str(self.num_rerurun_model_building) + ' Rounds)\n' + str(metric_), fontsize=8.5 )  # ,fontweight='bold')
                            else:
                                axs[j,i].set_title(str(embedding_type_final_eval_)+'\n' + str(metric_), fontsize=8.5 )  # ,fontweight='bold')

                        # update x-axis labels
                        axs[j,i].set_xticklabels(param_vals_one_embd_, rotation=0, fontsize=8.5)
                        axs[j,i].set_xlabel(str(self.parameter_to_optimize), fontsize=8.5 )

                        # # Add colored border
                        # for spine in axs[j,i].spines.values():
                        #     spine.set_edgecolor('red')
                        #     spine.set_linewidth(3)

                        if metric_ == 'MCC':
                            axs[j,i].set_ylim(-1, 1)
                            axs[j,i].set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
                            axs[j,i].tick_params(axis='y', labelsize=7)
                        else:
                            axs[j,i].set_ylim(0, 1)
                            axs[j,i].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                            axs[j,i].tick_params(axis='y', labelsize=7)


                    # Single Row
                    except:
                        bplot1 = axs[i].boxplot(
                            data_,
                            vert=True,  # vertical box alignment
                            patch_artist=True,  # fill with color
                            labels=param_vals_one_embd_,
                            flierprops=flierprops__, boxprops=boxprops__, medianprops=medianprops__,
                            capprops=dict(color='black'),
                            whiskerprops=dict(color='black'),

                        )  # will be used to label x-ticks
                        axs[i].set_title(metric_.replace('beta',str(self.f_beta_)), fontsize=8.5 )
                        if i == 3:
                            # if embedding_type_final_eval_ == self.feature_encoding_ls[0]:# for first row of plots in figure
                            if embedding_type_final_eval_ == paired_feature_encoding_ls[0]:  # for first row of plots in figure
                                axs[i].set_title('Plot ' + str(plotn_ + 1) + ' / ' + str(num_plots_) + ' Final Model Performances (' + str(self.num_rerurun_model_building) + ' Rounds)\n' + str(metric_),fontsize=8.5 )  # ,fontweight='bold')
                            else:
                                axs[i].set_title(str(embedding_type_final_eval_) + '\n' + str(metric_), fontsize=8.5 )  # ,fontweight='bold')

                        # update x-axis labels
                        axs[i].set_xticklabels(param_vals_one_embd_, rotation=0, fontsize=8.5)
                        axs[i].set_xlabel(str(self.parameter_to_optimize), fontsize=8.5 )

                        # label metric score values next to each box
                        #self.autolabel_boxplot(bplot1['medians'], axs[i], label_color='black')
                        #*# self.autolabel_boxplot_below(bplot1['caps'][::2], bplot1['medians'], axs[i])

                        # # Add colored border
                        # for spine in axs[i].spines.values():
                        #     spine.set_edgecolor('red')
                        #     spine.set_linewidth(3)

                        if metric_ == 'MCC':
                            axs[i].set_ylim(-1, 1)
                            axs[i].set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
                            axs[i].tick_params(axis='y', labelsize=7)
                        else:
                            axs[i].set_ylim(0, 1)
                            axs[i].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                            axs[i].tick_params(axis='y', labelsize=7)

                    # try:
                    #     # label metric score values next to each box
                    #     #self.autolabel_boxplot(bplot1['medians'], axs[j,i], label_color = 'black')
                    #     #*# self.autolabel_boxplot_below(bplot1['caps'][::2], bplot1['medians'], axs[j,i])
                    # except:
                    #     pass # already labeled above

            random_flag_ = ''
            if self.randomize_ext_data_:
                random_flag_ = '(Randomized) '

            fig.suptitle(str(plotn_+1)+' Compiled Multiple Metrics Final Models Evaluated on External Dataset '+random_flag_+'- Per Parameter Value ' + str(self.num_rerurun_model_building) +
                         ' rounds' + '\n' + self.output_run_file_info_string_.replace('_', ' ').replace(self.region_.replace('_', '-'), self.region_.replace('_', '-')) + '\n'+'External Dataset: '+self.external_data_file_, fontsize=9)
            # fig.patch.set_linewidth(10)
            # fig.patch.set_edgecolor('red')
            # fig.tight_layout()

            # ** SAVE FIGURE **
            plt.rcParams['svg.fonttype'] = 'none'  # exports text as strings rather than vector paths (images)
            fnm_ = (self.output_directory + 'figures/ext_data/' + str(plotn_+1)+'_ext-data-eval_final_bxp_per-param-val_' + str(self.num_rerurun_model_building) + '-rnds')
            fnm_svg_ = (self.output_directory + 'figures/ext_data/' + 'svg_figs/' + str(plotn_+1)+'_ext-data-eval_final_bxp_per-param-val_' + str(self.num_rerurun_model_building) + '-rnds')
            fig.savefig(fnm_svg_.split('.')[0] + '.svg', format='svg', transparent=True)
            fig.savefig(fnm_.split('.')[0] + '.png', format='png', dpi=300, transparent=False)
            logging.info('Figure saved to:'+str(fnm_) + '.png'.replace(self.output_directory, '~/'))
        return


    def plot_final_model_box_plots_per_param_val_on_final_and_external(self):
        logging.info("\nPlotting box plots for final models AND external dataset per parameter value...")
        if not self.apply_final_models_to_external_dataset_:
            logging.info("apply_final_models_to_external_dataset_ is set to False so did not evaluate on an external dataset")
            return

        ## Plot Compiled Multimetrics Model Performance per Param Val - Final Models

        # Each column of final_detailed_metric_df contains a single round for a single embedding type
        final_detailed_metric_df_ext = pd.DataFrame(self.ext_final_detailed_performance_metrics_encodings_dict)




        ######################################
        if self.param_values_to_loop_ == []:
            logging.info("No parameter values to plot by")
            return
        ## Plot Compiled Multimetrics Model Performance per Param Val - Final Models

        # Each column of final_detailed_metric_df contains a single round for a single embedding type
        final_detailed_metric_df = pd.DataFrame(self.final_detailed_performance_metrics_encodings_dict)

        flierprops__ = dict(marker='.', markerfacecolor='none', markersize=4, linewidth=0.1, markeredgecolor='black')  # linestyle='none',
        boxprops__ = dict(facecolor='none', linestyle='none', linewidth=1, edgecolor='k', )
        medianprops__ = dict(linewidth=2, color=testing_set_plot_color)
        medianprops__ext = dict(linewidth=2, color=external_set_plot_color)

        # one axis per performance metric
        # one box per embedding per axis
        # fig, axs = plt.subplots( len(self.feature_encoding_ls), len(final_detailed_metric_df))
        # fig.set_size_inches(w=14, h=3*len(self.feature_encoding_ls))
        # Update: Make plots in sets of two or fewer (so not too large)
        # TODO: Add +1 to num_plots_ for an extra plot with all embeddings overlayed (in different colors)
        num_plots_ =  int(int(len(self.feature_encoding_ls)/3)+(len(self.feature_encoding_ls)%3))# + 1
        paired_feature_encoding_ls = [self.feature_encoding_ls[i:i + 3] for i in range(0, len(self.feature_encoding_ls), 3)]

        for plotn_ in list(range(num_plots_)):
            if plotn_ == num_plots_ - 1:  # TODO: last plot is just 1 row since is a combined plot with all embeddings
                fig, axs = plt.subplots(1, len(final_detailed_metric_df))
                fig.set_size_inches(w=14, h=3 )
            else:
                fig, axs = plt.subplots(len(paired_feature_encoding_ls[plotn_]), len(final_detailed_metric_df))
                fig.set_size_inches(w=14, h=3*len(paired_feature_encoding_ls[plotn_]))

            # Split Evaluation Metric Data per parameter value
            # Loop through all embeddings used
            #for embedding_type_final_eval_, j in zip(self.feature_encoding_ls,list(range(len(self.feature_encoding_ls)))):
            for embedding_type_final_eval_, j in zip(paired_feature_encoding_ls[plotn_], list(range(len(paired_feature_encoding_ls[plotn_])))):

                # From final_detailed_metric_df get just columns for a single selected embedding type
                # Get just columns with selected embedding metric (embedding_type_final_eval_)
                cols_with_embd_ = [x for x in list(final_detailed_metric_df.columns) if embedding_type_final_eval_ in x]
                logging.info("Columns from final_detailed_metric_df with embedding = 'embedding_type_final_eval_' = "+
                      str(embedding_type_final_eval_)+" - "+str(len(cols_with_embd_))+' out of '+str(len(list(final_detailed_metric_df.columns))))

                # Get just columns with selected embedding metric (embedding_type_final_eval_)
                final_detailed_metric_one_embd_df = final_detailed_metric_df[cols_with_embd_]

                # Get parameter VALUES only for each selected embedding
                param_vals_one_embd_ = list(set([x.split(str(self.parameter_to_optimize)+'-')[-1].split('_round_')[0] for x in list(final_detailed_metric_one_embd_df.columns)]))

                # Get counts for each

                param_vals_one_embd_mult_ = list([x.split(str(self.parameter_to_optimize)+'-')[-1].split('_round_')[0] for x in list(final_detailed_metric_one_embd_df.columns)])

                ct_param_vals_one_embd_mult_ = Counter(param_vals_one_embd_mult_)
                x_labs_with_counts_ = [str(x) + '\n(' + str(ct_param_vals_one_embd_mult_[x]) + ')' for x in param_vals_one_embd_]

                # If Parameter values are integers, order them from smallest to larget
                try:
                    int_param_vals_one_embd_ = [int(x) for x in param_vals_one_embd_]
                    int_param_vals_one_embd_.sort()
                    param_vals_one_embd_ = [str(x) for x in int_param_vals_one_embd_]
                except:
                    pass

                metrics_ls = list(final_detailed_metric_one_embd_df.index)

                final_model_ct_per_param_val_dict = {} # holds counts of each parameter value used to build a final model
                for v__ in self.param_values_to_loop_:
                    ct_v__ = len([x for x in self.final_model_params_ls if str(x) == str(v__)])
                    final_model_ct_per_param_val_dict[str(v__)] = ct_v__

                for i in range(len(metrics_ls)):
                    metric_ = metrics_ls[i]
                    # Plot by parameter value
                    # data_ = [list(final_detailed_metric_one_embd_df[
                    #                   [embedding_type_final_eval_+'-'+str(self.parameter_to_optimize)+'-'+param_val_ + '_round_' + str(i) for i in
                    #                    list(range(final_model_ct_per_param_val_dict[str(param_val_)]))]].transpose()[metric_]) for param_val_ in param_vals_one_embd_]
                    # get columns for single parameter value and metric

                    cols_to_select_ = []
                    for param_val_ in param_vals_one_embd_:
                        # Get rounds for given parameter value and embedding
                        rounds_per_one_param_val_ = [int(x.split('_round_')[-1]) for x in list(final_detailed_metric_one_embd_df.columns) if embedding_type_final_eval_ + '-' + str(self.parameter_to_optimize) + '-' + param_val_ + '_round_' in x]
                        for rnd__ in rounds_per_one_param_val_:
                            cols_to_select_.append(embedding_type_final_eval_+'-'+str(self.parameter_to_optimize)+'-'+param_val_ +'_round_'+str(rnd__))
                    cols_to_select_ext_ = []
                    for param_val_ in param_vals_one_embd_:
                        # Get rounds for given parameter value and embedding
                        rounds_per_one_param_val_ = [int(x.split('_round_')[-1]) for x in list(final_detailed_metric_one_embd_df_ext.columns) if embedding_type_final_eval_ + '-' + str(self.parameter_to_optimize) + '-' + param_val_ + '_round_' in x]
                        for rnd__ in rounds_per_one_param_val_:
                            cols_to_select_ext_.append(embedding_type_final_eval_ + '-' + str(self.parameter_to_optimize) + '-' + param_val_ + '_round_' + str(rnd__))

                    data_ = [list(final_detailed_metric_one_embd_df[cols_to_select_].transpose()[metric_]) for param_val_ in param_vals_one_embd_]
                    data_ext = [list(final_detailed_metric_one_embd_df_ext[cols_to_select_ext_].transpose()[metric_]) for param_val_ in param_vals_one_embd_]
                    #logging.info(data_)
                    # Multiple Rows
                    try:
                        bplot1 = axs[j,i].boxplot(
                            data_,
                            vert=True,  # vertical box alignment
                            patch_artist=True,  # fill with color
                            labels=param_vals_one_embd_,
                            flierprops=flierprops__, boxprops=boxprops__,medianprops=medianprops__,
                            capprops=dict(color='black'),
                            whiskerprops=dict(color='black'),

                        )  # will be used to label x-ticks
                        bplot2 = axs[j, i].boxplot(
                            data_ext,
                            vert=True,  # vertical box alignment
                            patch_artist=True,  # fill with color
                            labels=param_vals_one_embd_,
                            flierprops=flierprops__, boxprops=boxprops__,medianprops=medianprops__ext,
                            capprops=dict(color='black'),
                            whiskerprops=dict(color='black'),

                        )  # will be used to label x-ticks
                        axs[j,i].set_title(metric_.replace('beta',str(self.f_beta_)), fontsize=8.5 )
                        if i == 3:
                            #if embedding_type_final_eval_ == self.feature_encoding_ls[0]:# for first row of plots in figure
                            if embedding_type_final_eval_ == paired_feature_encoding_ls[0]:  # for first row of plots in figure
                                axs[j,i].set_title('Plot '+str(plotn_+1)+' / '+str(num_plots_)+' Final Model Performances (' + str(self.num_rerurun_model_building) + ' Rounds)\n' + str(metric_), fontsize=8.5 )  # ,fontweight='bold')
                            else:
                                axs[j,i].set_title(str(embedding_type_final_eval_)+'\n' + str(metric_), fontsize=8.5 )  # ,fontweight='bold')

                        # update x-axis labels
                        #axs[j,i].set_xticklabels(param_vals_one_embd_, rotation=0, fontsize=8.5)
                        axs[j, i].set_xticklabels(x_labs_with_counts_, rotation=0, fontsize=8.5)
                        axs[j,i].set_xlabel(str(self.parameter_to_optimize),fontsize=8.5 )


                        if metric_ == 'MCC':
                            axs[j,i].set_ylim(-1, 1)
                            axs[j,i].set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
                            axs[j,i].tick_params(axis='y', labelsize=7)
                        else:
                            axs[j,i].set_ylim(0, 1)
                            axs[j,i].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                            axs[j,i].tick_params(axis='y', labelsize=7)

                    # Single Row
                    except:
                        bplot1 = axs[i].boxplot(
                            data_,
                            vert=True,  # vertical box alignment
                            patch_artist=True,  # fill with color
                            labels=param_vals_one_embd_,
                            flierprops=flierprops__, boxprops=boxprops__,medianprops=medianprops__,
                            capprops=dict(color='black'),
                            whiskerprops=dict(color='black'),

                        )  # will be used to label x-ticks
                        bplot2 = axs[i].boxplot(
                            data_ext,
                            vert=True,  # vertical box alignment
                            patch_artist=True,  # fill with color
                            labels=param_vals_one_embd_,
                            flierprops=flierprops__, boxprops=boxprops__, medianprops=medianprops__ext,
                            capprops=dict(color='black'),
                            whiskerprops=dict(color='black'),

                        )  # will be used to label x-ticks
                        axs[i].set_title(metric_.replace('beta',str(self.f_beta_)), fontsize=8.5 )
                        if i == 3:
                            # if embedding_type_final_eval_ == self.feature_encoding_ls[0]:# for first row of plots in figure
                            if embedding_type_final_eval_ == paired_feature_encoding_ls[0]:  # for first row of plots in figure
                                axs[i].set_title('Plot ' + str(plotn_ + 1) + ' / ' + str(num_plots_) + ' Final Model Performances (' + str(self.num_rerurun_model_building) + ' Rounds)\n' + str(metric_) , fontsize=8.5 )  # ,fontweight='bold')
                            else:
                                axs[i].set_title(str(embedding_type_final_eval_) + '\n' + str(metric_), fontsize=8.5 )  # ,fontweight='bold')

                        # update x-axis labels
                        # axs[i].set_xticklabels(param_vals_one_embd_, rotation=0, fontsize=8.5)
                        axs[i].set_xticklabels(x_labs_with_counts_, rotation=0, fontsize=8.5)
                        axs[i].set_xlabel(str(self.parameter_to_optimize), fontsize=8.5 )

                        # label metric score values next to each box
                        #self.autolabel_boxplot(bplot1['medians'], axs[i], label_color='black')
                        #*# self.autolabel_boxplot_below(bplot1['caps'][::2], bplot1['medians'], axs[i])

                        if metric_ == 'MCC':
                            axs[i].set_ylim(-1, 1)
                            axs[i].set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
                            axs[i].tick_params(axis='y', labelsize=7)
                        else:
                            axs[i].set_ylim(0, 1)
                            axs[i].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                            axs[i].tick_params(axis='y', labelsize=7)
                    # try:
                    #     # label metric score values next to each box
                    #     #self.autolabel_boxplot(bplot1['medians'], axs[j,i], label_color = 'black')
                    #     #*# self.autolabel_boxplot_below(bplot1['caps'][::2], bplot1['medians'], axs[j, i])
                    # except:
                    #     pass # already labeled above

            #if plotn_ == num_plots_ - 1:  # TODO: last plot needs a legend (to identify each embedding as a different color)

            fig.suptitle(str(plotn_+1)+' Compiled Multiple Metrics Final Models and External - Per Parameter Value ' + str(self.num_rerurun_model_building) +
                         ' rounds' + '\n' + self.output_run_file_info_string_.replace('_', ' ').replace(self.region_.replace('_', '-'), self.region_.replace('_', '-') + '\n'), fontsize=9)
            fig.tight_layout()

            # ** SAVE FIGURE **
            plt.rcParams['svg.fonttype'] = 'none'  # exports text as strings rather than vector paths (images)
            fnm_ = (self.output_directory + 'figures/' + str(plotn_+1)+'_final_and_ext_bxp_per-param-val_' + str(self.num_rerurun_model_building) + '-rnds')
            fnm_svg_ = (self.output_directory + 'figures/' + 'svg_figs/' + str(plotn_+1)+'_final_and_ext_bxp_per-param-val_' + str(self.num_rerurun_model_building) + '-rnds')
            fig.savefig(fnm_svg_.split('.')[0] + '.svg', format='svg', transparent=True)
            fig.savefig(fnm_.split('.')[0] + '.png', format='png', dpi=300, transparent=False)
            logging.info('Figure saved to:'+str(fnm_) + '.png'.replace(self.output_directory, '~/'))
        return


    def plot_final_model_box_plots_per_metric_on_ext_dataset(self):
        logging.info("\nPlotting box plots -- per performance metric -- for final models evaluated on external dataset per parameter value...")
        if not self.apply_final_models_to_external_dataset_:
            logging.info("apply_final_models_to_external_dataset_ is set to False so did not evaluate on an external dataset")
            return
        ## Plot Compiled Multimetrics Model Performance - Final Models per metric
        final_metric_df = pd.DataFrame(self.ext_final_performance_metrics_encodings_dict)

        metrics_ls = list(final_metric_df.index)
        enc_ls_ = self.feature_encoding_ls

        flierprops__ = dict(marker='.', markerfacecolor='none', markersize=4, linewidth=0.1, markeredgecolor='black')  # linestyle='none',
        medianprops__ext = dict(linewidth=2, color=external_set_plot_color)
        boxprops__ = dict(facecolor='none', linestyle='none', linewidth=1, edgecolor='k')



        # one axis per performance metric
        # one box per embedding per axis
        fig, axs = plt.subplots(1, len(final_metric_df))
        fig.set_size_inches(w=12, h=3)
        fntsz_ = 7
        for i in range(len(metrics_ls)):
            metric_ = metrics_ls[i]
            data_ = [list(final_metric_df[[enc_ + '_' + str(i) for i in list(range(self.num_rerurun_model_building))]].transpose()[metric_]) for enc_ in enc_ls_]
            # data_ = [list(final_metric_df[[enc_+'_' + str(i) for i in [0, 1]]].transpose()[metric_]) for enc_ in enc_ls_]

            bplot1 = axs[i].boxplot(
                data_,
                vert=True,  # vertical box alignment
                patch_artist=True,  # fill with color
                labels=enc_ls_,
                flierprops=flierprops__, boxprops=boxprops__, medianprops=medianprops__ext,
                capprops=dict(color='black'),
                whiskerprops=dict(color='black'),

            )  # will be used to label x-ticks
            axs[i].set_title(metric_.replace('beta',str(self.f_beta_)), fontsize = fntsz_)
            if i ==2:
                axs[i].set_title('Final Model Performances (' + str(self.num_rerurun_model_building) + ' Rounds)\n' + str(metric_), fontsize = fntsz_)  # ,fontweight='bold')

            # update x-axis labels
            axs[i].set_xticklabels([feature_encodings_dict[x] for x in self.feature_encoding_ls], rotation=90, fontsize = fntsz_)

            # # Add colored border
            # for spine in axs[i].spines.values():
            #     spine.set_edgecolor('red')
            #     spine.set_linewidth(3)

            if metric_ == 'MCC':
                axs[i].set_ylim(-1, 1)
                axs[i].set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
                axs[i].tick_params(axis='y', labelsize=fntsz_)
            else:
                axs[i].set_ylim(0, 1)
                axs[i].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                axs[i].tick_params(axis='y', labelsize=fntsz_)

            #*# self.autolabel_boxplot_below(bplot1['caps'][::2], bplot1['medians'], axs[i])

        random_flag_ = ''
        if self.randomize_ext_data_:
            random_flag_ = '(Randomized) '

        fig.suptitle('Compiled Multiple Metrics Final Models Evaluated on External Dataset '+random_flag_+str(self.num_rerurun_model_building)+
                     ' rounds' +'\n'+self.output_run_file_info_string_.replace('_',' ').replace(self.region_.replace('_','-'),self.region_.replace('_','-'))+'\n'+'External Dataset: '+self.external_data_file_, fontsize=9)
        # fig.patch.set_linewidth(10)
        # fig.patch.set_edgecolor('red')
        # fig.tight_layout()


        # ** SAVE FIGURE **
        plt.rcParams['svg.fonttype'] = 'none' # exports text as strings rather than vector paths (images)
        fnm_ =     (self.output_directory+ 'figures/ext_data/'+           'bxp_'+str(self.num_rerurun_model_building)+'-rnds_final_ext-data-eval')
        fnm_svg_ = (self.output_directory+'figures/ext_data/'+'svg_figs/'+'bxp_'+str(self.num_rerurun_model_building)+'-rnds_final_ext-data-eval')
        fig.savefig(fnm_svg_.split('.')[0]+'.svg',format='svg',transparent=True)
        fig.savefig(fnm_.split('.')[0]+'.png',format='png',dpi=300,transparent=False)
        logging.info('Figure saved to: '+ (fnm_+'.png').replace(self.output_directory,'~/'))
        return

    def plot_final_model_and_external_data_box_plots_per_metric(self):
        logging.info("\nPlotting box plots -- per performance metric -- for final models...")
        ## Plot Compiled Multimetrics Model Performance - Final Models per metric
        final_metric_df = pd.DataFrame(self.final_performance_metrics_encodings_dict)
        final_metric_df_ext = pd.DataFrame(self.ext_final_performance_metrics_encodings_dict)
        if self.include_random_background_comparison_:  # randombackground
            final_metric_df_randombackground = pd.DataFrame(self.randombackground_final_performance_metrics_encodings_dict)

        # Each column of final_detailed_metric_df contains a single round for a single embedding type
        fnm_ = (self.output_directory + 'data/' + 'performance_metrics_' + str(self.num_rerurun_model_building) + '-rnds_final_on-test-dataset_per-metric.csv')
        final_metric_df.to_csv(fnm_, index=True)
        logging.info("Final Models Performance Metrics on Test Dataset Dataframe saved to:\n\t"+fnm_)

        fnm_ = (self.output_directory + 'data/' + 'performance_metrics_' + str(self.num_rerurun_model_building) + '-rnds_final_ext-data-eval_per-metric.csv')
        final_metric_df_ext.to_csv(fnm_, index=True)
        logging.info("Final Models Performance Metrics (when Evaluated on External Dataset) Dataframe saved to:\n\t"+fnm_)

        if self.include_random_background_comparison_:  # randombackground
            fnm_ = (self.output_directory + 'data/' + 'performance_metrics_' + str(self.num_rerurun_model_building) + '-rnds_final_randombackground-data-eval_per-metric.csv')
            final_metric_df_randombackground.to_csv(fnm_, index=True)
            logging.info("Final Models Performance Metrics (when Evaluated on Random Background Shuffled Dataset) Dataframe saved to:\n\t"+fnm_)

        metrics_ls = list(final_metric_df.index)
        enc_ls_ = self.feature_encoding_ls

        flierprops__ = dict(marker='.', markerfacecolor='none', markersize=4, linewidth=0.1, markeredgecolor='black')  # linestyle='none',
        boxprops__ = dict(facecolor='none', linestyle='none', linewidth=1, edgecolor='k', )
        medianprops__= dict(linewidth=2, color=testing_set_plot_color)
        medianprops__ext = dict(linewidth=2, color=external_set_plot_color)
        if self.include_random_background_comparison_:  # randombackground
            medianprops__randombackground = dict(linewidth=2, color='grey')

        # one axis per performance metric
        # one box per embedding per axis
        fig, axs = plt.subplots(1, len(final_metric_df)+1)
        fig.set_size_inches(w=12+1, h=3)
        fntsz_ = 7

        # For exporting data used to make boxplots
        bxplt_data_dict ={} # dict of lists by metric
        bxplt_data_ext_dict ={} # dict of lists by metric
        if self.include_random_background_comparison_:  # randombackground
            bxplt_data_randombackground_dict = {}  # dict of lists by metric

        for i in range(len(metrics_ls)):
            metric_ = metrics_ls[i]

            data_ = [list(final_metric_df[[enc_ + '_' + str(i) for i in list(range(self.num_rerurun_model_building)) ]].transpose()[metric_]) for enc_ in enc_ls_]
            data_ext = [list(final_metric_df_ext[[enc_ + '_' + str(i) for i in list(range(self.num_rerurun_model_building)) ]].transpose()[metric_]) for enc_ in enc_ls_]
            if self.include_random_background_comparison_:  # randombackground
                data_randombackground = [list(final_metric_df_randombackground[[enc_ + '_' + str(i) for i in list(range(self.num_rerurun_model_building)) ]].transpose()[metric_]) for enc_ in enc_ls_]

            # data_ = [list(final_metric_df[[enc_ + '_' + str(i) for i in [0, 1]]].transpose()[metric_]) for enc_ in enc_ls_]
            # data_ext = [list(final_metric_df_ext[[enc_ + '_' + str(i) for i in [0, 1]]].transpose()[metric_]) for enc_ in enc_ls_]

            # For exporting data used to make boxplots
            bxplt_data_dict[metric_] = data_
            bxplt_data_ext_dict[metric_] = data_ext
            if self.include_random_background_comparison_:  # randombackground
                bxplt_data_randombackground_dict[metric_] = data_randombackground

            bplot1 = axs[i].boxplot(
                data_,
                vert=True,  # vertical box alignment
                patch_artist=True,  # fill with color
                labels=enc_ls_,
                flierprops=flierprops__, boxprops=boxprops__,medianprops=medianprops__,
                capprops=dict(color='black'),
                whiskerprops=dict(color='black'),

            )  # will be used to label x-ticks
            bplot2 = axs[i].boxplot(
                data_ext,
                vert=True,  # vertical box alignment
                patch_artist=True,  # fill with color
                labels=enc_ls_,
                flierprops=flierprops__, boxprops=boxprops__,medianprops=medianprops__ext,
                capprops=dict(color='black'),
                whiskerprops=dict(color='black'),

            )

            if self.include_random_background_comparison_:  # randombackground
                bplot3 = axs[i].boxplot(
                    data_randombackground,
                    vert=True,  # vertical box alignment
                    patch_artist=True,  # fill with color
                    labels=enc_ls_,
                    flierprops=flierprops__, boxprops=boxprops__, medianprops=medianprops__randombackground,
                    capprops=dict(color='black'),
                    whiskerprops=dict(color='black'),

                )

            axs[i].set_title(metric_.replace('beta',str(self.f_beta_)), fontsize=fntsz_)
            if i == 3:
                axs[i].set_title('Final Model Performances (' + str(self.num_rerurun_model_building) + ' Rounds)\n' + str(metric_), fontsize=fntsz_)  # ,fontweight='bold')


            # # update x-axis labels
            tick_lab_list__ = []
            for x in self.feature_encoding_ls:
                tick_lab_list__.append('')
            if self.include_random_background_comparison_:
                for x in self.feature_encoding_ls:
                    tick_lab_list__.append('')
            for x in self.feature_encoding_ls:
                tick_lab_list__.append(feature_encodings_dict[x])

            # axs[i].set_xticklabels([feature_encodings_dict[x] for x in self.feature_encoding_ls], fontsize=fntsz_, rotation=90)
            axs[i].set_xticklabels(tick_lab_list__, fontsize=fntsz_, rotation=90)
            # axs[i].set_xticklabels(['','','','one-hot','gensim-weights','gensim-values'], fontsize=fntsz_, rotation=90)

            if metric_ == 'MCC':
                axs[i].set_ylim(-1, 1)
                axs[i].set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
                axs[i].tick_params(axis='y', labelsize=fntsz_)
            else:
                axs[i].set_ylim(0, 1)
                axs[i].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                axs[i].tick_params(axis='y', labelsize=fntsz_)

            #*# self.autolabel_boxplot_below(bplot1['caps'][::2], bplot1['medians'], axs[i], label_color = testing_set_plot_color)
            #*# self.autolabel_boxplot_below(bplot2['caps'][::2], bplot2['medians'], axs[i], label_color = external_set_plot_color)

        fig.suptitle('Compiled Multiple Metrics Final Models ' + str(self.num_rerurun_model_building) +
                     ' rounds' + '\n' + self.output_run_file_info_string_.replace('_', ' ').replace(self.region_.replace('_', '-'), self.region_.replace('_', '-') + '\n'), fontsize=9)

        # Each column of final_detailed_metric_df contains a single round for a single embedding type
        # For exporting data used to make boxplots
        bxplt_data_df = pd.DataFrame(bxplt_data_dict)
        fnm_ = (self.output_directory + 'data/' + 'boxplot_performance_metrics_' + str(self.num_rerurun_model_building) + '-rnds_final_per-metric.csv')
        bxplt_data_df.to_csv(fnm_, index=True)
        logging.info("Final Models Performance Metrics used to make boxplots Dataframe saved to:\n\t"+fnm_)

        bxplt_data_ext_df = pd.DataFrame(bxplt_data_ext_dict)
        fnm_ = (self.output_directory + 'data/' + 'boxplot_performance_metrics_' + str(self.num_rerurun_model_building) + '-rnds_final_ext-data-eval_per-metric.csv')
        bxplt_data_ext_df.to_csv(fnm_, index=True)
        logging.info("Final Models Performance Metrics used to make boxplots (when Evaluated on External Dataset) Dataframe saved to:\n\t"+fnm_)

        if self.include_random_background_comparison_:  # randombackground
            bxplt_data_randombackground_df = pd.DataFrame(bxplt_data_randombackground_dict)
            fnm_ = (self.output_directory + 'data/' + 'boxplot_performance_metrics_' + str(self.num_rerurun_model_building) + '-rnds_final_randombackground-data-eval_per-metric.csv')
            bxplt_data_randombackground_df.to_csv(fnm_, index=True)
            logging.info("Final Models Performance Metrics used to make boxplots (when Evaluated on Randomized Background Shuffled Dataset) Dataframe saved to:\n\t"+fnm_)


        # Add legend for parameter values

        legend_elements = [
            Line2D([0], [0],
                   color=testing_set_plot_color,  # embd_color_dict[embd_][val__],
                   lw=4, label='Test Set'),
            Line2D([0], [0],
                   color=external_set_plot_color,  # color_,  # embd_color_dict[embd_][val__],
                   lw=4, label='External Dataset')
        ]
        if self.include_random_background_comparison_:  # randombackground
            legend_elements.append(
                Line2D([0], [0],
                   color='grey',  # color_,  # embd_color_dict[embd_][val__],
                   lw=4, label='Random Background'))

        axs[-1].legend(
            handles=legend_elements, loc='upper left', frameon=False, bbox_to_anchor=(0, 1), fontsize=12
            #title=self.parameter_to_optimize, title_fontsize=12,
        )
        axs[-1].axis('off')


        fig.tight_layout()

        # ** SAVE FIGURE **
        plt.rcParams['svg.fonttype'] = 'none'  # exports text as strings rather than vector paths (images)
        fnm_ = (self.output_directory + 'figures/' + 'bxp_' + str(self.num_rerurun_model_building) + '-rnds_final_and_external')
        fnm_svg_ = (self.output_directory + 'figures/' + 'svg_figs/' + 'bxp_' + str(self.num_rerurun_model_building) + '-rnds_final_and_external')
        fig.savefig(fnm_svg_.split('.')[0] + '.svg', format='svg', transparent=True)
        fig.savefig(fnm_.split('.')[0] + '.png', format='png', dpi=300, transparent=False)
        logging.info('Figure saved to:'+str(fnm_) + '.png'.replace(self.output_directory, '~/'))
        return

    def plot_final_model_precision_recall_curves_on_ext_dataset_and_test_set(self):
        logging.info("\nPlotting precision-recall curves for final models on test set AND evaluated on external dataset...")
        if not self.apply_final_models_to_external_dataset_:
            logging.info("apply_final_models_to_external_dataset_ is set to False so did not evaluate on an external dataset")
            return


        ## Plot Final Model Precision-Recall curves as a single figure
        sup_title_id_info = ('' +  # str(num_rerurun_model_building*run_round_num)+' rounds'+'\n'+
                             self.output_run_file_info_string_.replace('_', ' ').replace(self.region_.replace('_', '-'), self.region_.replace('_', '-') + '\n'))

        pr_curves_dict = self.final_performance_curves_encodings_dict
        pr_curves_dict_ext = self.ext_final_performance_curves_encodings_dict
        if self.include_random_background_comparison_: # randombackground
            pr_curves_dict_randombackground = self.randombackground_final_performance_curves_encodings_dict
        pr_curves_keys_ = self.final_key_ls


        # Plot a single PLOT for each embedding type
        fig, axs = plt.subplots(1,len(self.feature_encoding_ls )+ 1)
        fig.set_size_inches(w=3*(len(self.feature_encoding_ls )+ 1), h=4.5, )  # NOTE: h and w must be large enough to accomodate any legends


        longest_prt_ = -1  # For exporting Precision-Recall curve numeric data as a .csv file
        for e, col_ in zip(self.feature_encoding_ls,list(range(len(self.feature_encoding_ls)))):  # different plots per embedding
            axs[col_].set_title(str(e))
            for n in range(self.num_rerurun_model_building):  # loop through rounds

                # Get p-r curves per round for given embedding and round
                key__ = str(e) + '_' + str(n)
                p = self.final_model_params_ls[n]

                # try: # get color for parameter optimization looping
                #     color_ = param_col_ls_dict[p]
                # except: # if not looping through parameters select color from list
                #     color_ = '#5784db'
                # NOTE: no p-r curves for label-propagation/spreading with one-hot encoding
                if not (((self.parameter_to_optimize == 'model') and ('label' in p) and (e == 'one-hot')) or (
                        ('label' in self.model_type_) and (e == 'one-hot'))):



                    pcurve__ = self.final_performance_curves_encodings_dict[key__]['Precision_Recall_Curve'][0]
                    rcurve__ = self.final_performance_curves_encodings_dict[key__]['Precision_Recall_Curve'][1]
                    thresholds__ = self.final_performance_curves_encodings_dict[key__]['Precision_Recall_Curve'][1]
                    # unach_pcurve__ = self.final_performance_curves_encodings_dict[key__]['Unacheivable_Region_Curve'][0]
                    # unach_rcurve__ = self.final_performance_curves_encodings_dict[key__]['Unacheivable_Region_Curve'][1]



                    axs[col_].plot(
                        rcurve__,  # r_OH,# x
                        pcurve__,  # p_OH,# y
                        lw=1,
                        color= testing_set_plot_color,
                    )

                    pcurve__ext__ = self.ext_final_performance_curves_encodings_dict[key__]['Precision_Recall_Curve'][0]
                    rcurve__ext__ = self.ext_final_performance_curves_encodings_dict[key__]['Precision_Recall_Curve'][1]

                    axs[col_].plot(
                        rcurve__ext__,  # r_OH,# x
                        pcurve__ext__,  # p_OH,# y
                        lw=1,
                        color=external_set_plot_color,
                    )
                    if self.include_random_background_comparison_:
                        pcurve__randombackground__ = self.randombackground_final_performance_curves_encodings_dict[key__]['Precision_Recall_Curve'][0]
                        rcurve__randombackground__ = self.randombackground_final_performance_curves_encodings_dict[key__]['Precision_Recall_Curve'][1]

                        axs[col_].plot(
                            rcurve__randombackground__,  # r_OH,# x
                            pcurve__randombackground__,  # p_OH,# y
                            lw=1,
                            color='grey',
                        )
                        # Get longest curve (for padding)
                        if longest_prt_ < max(len(pcurve__), len(pcurve__ext__), len(pcurve__randombackground__)):
                            longest_prt_ = max(len(pcurve__), len(pcurve__ext__), len(pcurve__randombackground__))
                    else:
                        # Get longest curve (for padding)
                        if longest_prt_ < max(len(pcurve__), len(pcurve__ext__)):
                            longest_prt_ = max(len(pcurve__), len(pcurve__ext__))



                    # axs[col_].plot(
                    #     rcurve__,  # r_OH,# x
                    #     unach_rcurve__,  # p_OH,# y
                    #     lw=1,
                    #     color=color_,
                    #     linestyle='dashed',
                    # )

            # Format Axes
            axs[col_].set_xlim(0, 1)
            axs[col_].set_ylim(0, 1.1)
            # axs[col_].set_xticks(ticks=np.arange(0, 1.1, .25), minor=True)
            axs[col_].tick_params(direction='in', which='both', length=3, width=1)

            if col_ == 0:
                axs[col_].set_xlabel('Recall')
                axs[col_].set_ylabel('Precision')
                axs[col_].set_xticks(ticks=[0.5, 1.0 ], labels=[0.5, 1.0])
                # axs[col_].set_xticks(ticks=[0.0, 0.5, 1.0], labels=['', 0.5, 1.0])
                axs[col_].set_yticks(ticks=[0.0 , 0.5, 1.0 ], labels=[0.0, 0.5, 1.0])

            else:
                axs[col_].set_xticks(ticks=[0.5, 1.0 ], labels=[0.5, 1.0])
                # axs[col_].set_xticks(ticks=[0.0, 0.5, 1.0], labels=['', 0.5, 1.0])
                axs[col_].set_yticks(ticks=[0.0 , 0.5, 1.0 ], labels=[0.0, 0.5, 1.0])
                #axs[col_].set_xticks([])
                #axs[col_].set_yticks([])

            # # Add colored border
            # for spine in axs[col_].spines.values():
            #     spine.set_edgecolor('red')
            #     spine.set_linewidth(3)




        # Add legend for parameter values

        legend_elements = [
                        Line2D([0], [0],
                   color=testing_set_plot_color, # embd_color_dict[embd_][val__],
                   lw=4, label='Test Set'),
            Line2D([0], [0],
                   color= external_set_plot_color,#color_,  # embd_color_dict[embd_][val__],
                   lw=4, label='External Dataset')
        ]
        if self.include_random_background_comparison_:
            legend_elements.append(
                Line2D([0], [0],
                       color='grey',  # color_,  # embd_color_dict[embd_][val__],
                       lw=4, label='Random Background')
            )
        axs[-1].legend(handles=legend_elements, loc='upper left', frameon=False, bbox_to_anchor=(0, 1), title=self.parameter_to_optimize, title_fontsize=12, fontsize=12)
        axs[-1].axis('off')



        random_flag_ = ''
        if self.randomize_ext_data_:
            random_flag_ = '(Randomized) '
        fig.suptitle('Compiled Precision-Recall Curves Final Models Evaluated on External Dataset '+random_flag_+'- Per Embedding ' + str(self.num_rerurun_model_building) +
                     ' rounds' + '\n' + self.output_run_file_info_string_.replace('_', ' ').replace(self.region_.replace('_', '-'), self.region_.replace('_', '-')) + '\n'+
                     'External Dataset: '+self.external_data_file_, fontsize=9)

        fig.tight_layout()  # NOTE: h and w (above in fig.set_size... MUST be large enough to accomodate legends or will be cut off/squished in output)


        ## For exporting data to .csv file
        export_pr_dict = {}
        for e, col_ in zip(self.feature_encoding_ls,list(range(len(self.feature_encoding_ls)))):  # different plots per embedding
            for n in range(self.num_rerurun_model_building):  # loop through rounds
                # Get p-r curves per round for given embedding and round
                key__ = str(e) + '_' + str(n)
                p = self.final_model_params_ls[n]

                pcurve__ = self.final_performance_curves_encodings_dict[key__]['Precision_Recall_Curve'][0]
                rcurve__ = self.final_performance_curves_encodings_dict[key__]['Precision_Recall_Curve'][1]
                thresholds__ = self.final_performance_curves_encodings_dict[key__]['Precision_Recall_Curve'][1]

                # unach_pcurve__ = self.final_performance_curves_encodings_dict[key__]['Unacheivable_Region_Curve'][0]
                # unach_rcurve__ = self.final_performance_curves_encodings_dict[key__]['Unacheivable_Region_Curve'][1]

                pcurve__ext__ = self.ext_final_performance_curves_encodings_dict[key__]['Precision_Recall_Curve'][0]
                rcurve__ext__ = self.ext_final_performance_curves_encodings_dict[key__]['Precision_Recall_Curve'][1]

                # fill in values so all same length
                diff__ = longest_prt_ - len(pcurve__)
                pcurve__ = list(pcurve__) + diff__ * [-1.0]
                rcurve__ = list(rcurve__) + diff__ * [-1.0]
                thresholds__ = list(thresholds__) + diff__ * [-1.0]

                diff__ext__ = longest_prt_ - len(pcurve__ext__)
                pcurve__ext__ = list(pcurve__ext__) + diff__ext__ * [-1.0]
                rcurve__ext__ = list(rcurve__ext__) + diff__ext__ * [-1.0]

                export_pr_dict['test_precision_'+  key__] = pcurve__
                export_pr_dict['test_recall_' +    key__] = rcurve__
                export_pr_dict['test_threshold_' + key__] = thresholds__

                export_pr_dict['ext_precision_' + key__] = pcurve__ext__
                export_pr_dict['ext_recall_'    + key__] = rcurve__ext__

                if self.include_random_background_comparison_:
                    column_title_rand__ = 'random-backgorund_'+key__
                    pcurve__randombackground__ = self.randombackground_final_performance_curves_encodings_dict[key__]['Precision_Recall_Curve'][0]
                    rcurve__randombackground__ = self.randombackground_final_performance_curves_encodings_dict[key__]['Precision_Recall_Curve'][1]

                    # fill in values so all same length
                    diff__rand__ = longest_prt_ - len(pcurve__randombackground__)
                    pcurve__randombackground__ = list(pcurve__randombackground__) + diff__rand__*[-1.0]
                    rcurve__randombackground__ = list(rcurve__randombackground__) + diff__rand__ * [-1.0]

                    export_pr_dict['randbkgd_precision_' + key__] = pcurve__randombackground__
                    export_pr_dict['randbkgd_recall_'    + key__] = rcurve__randombackground__






        # Export p-r curve data as .csv
        fnm_ = (self.output_directory + 'data/' + 'p-r_curve_data_' + str(self.num_rerurun_model_building) + '-rnds_final_ext-data-eval_and-test-set.csv')
        logging.info('P-R curve data saved to:' + str(fnm_))
        pd.DataFrame(export_pr_dict).to_csv(fnm_)

        # ** SAVE FIGURE **
        plt.rcParams['svg.fonttype'] = 'none'  # exports text as strings rather than vector paths (images)
        fnm_ = (self.output_directory + 'figures/' + 'p-r_' + str(self.num_rerurun_model_building) + '-rnds_final_ext-data-eval_and-test-set')
        fnm_svg_ = (self.output_directory + 'figures/' + 'svg_figs/' + 'p-r_' + str(self.num_rerurun_model_building) + '-rnds_final_ext-data-eval_and-test-set')
        fig.savefig(fnm_svg_.split('.')[0] + '.svg', format='svg', transparent=True)
        fig.savefig(fnm_.split('.')[0] + '.png', format='png', dpi=300, transparent=False)
        logging.info('Figure saved to:'+str(fnm_) + '.png'.replace(self.output_directory, '~/'))
        return


    def plot_final_model_and_external_data_box_plots_f_score_only(self):
        logging.info("\nPlotting box plots -- F-Score ONLY -- for final models and on external datset...")
        ## Plot Compiled Multimetrics Model Performance - Final Models per metric
        final_metric_df = pd.DataFrame(self.final_performance_metrics_encodings_dict)
        final_metric_df_ext = pd.DataFrame(self.ext_final_performance_metrics_encodings_dict)
        if self.include_random_background_comparison_:  # randombackground
            final_metric_df_randombackground = pd.DataFrame(self.randombackground_final_performance_metrics_encodings_dict)
        # fnm_ = (self.output_directory + 'data/' + 'performance_metrics_' + str(self.num_rerurun_model_building) + '-rnds_final_on-test-dataset_per-metric.csv')
        # final_metric_df.to_csv(fnm_, index=True)
        # logging.info("Final Models Performance Metrics on Test Dataset Dataframe saved to:\n\t"+fnm_)

        # Each column of final_detailed_metric_df contains a single round for a single embedding type
        # fnm_ = (self.output_directory + 'data/' + 'performance_metrics_' + str(self.num_rerurun_model_building) + '-rnds_final_ext-data-eval_per-metric.csv')
        # final_metric_df_ext.to_csv(fnm_, index=True)
        # logging.info("Final Models Performance Metrics (when Evaluated on External Dataset) Dataframe saved to:\n\t"+fnm_)


        metrics_ls = ['F-Score']#list(final_metric_df.index)
        enc_ls_ = self.feature_encoding_ls

        flierprops__ = dict(marker='.', markerfacecolor='none', markersize=4, linewidth=0.1, markeredgecolor='black')  # linestyle='none',
        boxprops__ = dict(facecolor='none', linestyle='none', linewidth=1, edgecolor='k', )
        medianprops__= dict(linewidth=2, color=testing_set_plot_color)
        medianprops__ext = dict(linewidth=2, color=external_set_plot_color)
        if self.include_random_background_comparison_:  # randombackground
            medianprops__randombackground = dict(linewidth=2, color='grey')


        # one axis per performance metric
        # one box per embedding per axis
        fig, axs = plt.subplots(1, 1+1)
        fig.set_size_inches(w=8, h=6)
        fntsz_ = 12

        # For exporting data used to make boxplots
        bxplt_data_dict ={} # dict of lists by metric
        bxplt_data_ext_dict ={} # dict of lists by metric
        if self.include_random_background_comparison_:  # randombackground
            bxplt_data_randombackground_dict = {}  # dict of lists by metric
        for i in range(len(metrics_ls)):
            metric_ = metrics_ls[i]

            data_ = [list(final_metric_df[[enc_ + '_' + str(i) for i in list(range(self.num_rerurun_model_building)) ]].transpose()[metric_]) for enc_ in enc_ls_]
            data_ext = [list(final_metric_df_ext[[enc_ + '_' + str(i) for i in list(range(self.num_rerurun_model_building)) ]].transpose()[metric_]) for enc_ in enc_ls_]
            if self.include_random_background_comparison_:  # randombackground
                data_randombackground = [list(final_metric_df_randombackground[[enc_ + '_' + str(i) for i in list(range(self.num_rerurun_model_building))]].transpose()[metric_]) for enc_ in enc_ls_]

            # data_ = [list(final_metric_df[[enc_ + '_' + str(i) for i in [0, 1]]].transpose()[metric_]) for enc_ in enc_ls_]
            # data_ext = [list(final_metric_df_ext[[enc_ + '_' + str(i) for i in [0, 1]]].transpose()[metric_]) for enc_ in enc_ls_]

            # For exporting data used to make boxplots
            bxplt_data_dict[metric_] = data_
            bxplt_data_ext_dict[metric_] = data_ext
            if self.include_random_background_comparison_:  # randombackground
                bxplt_data_randombackground_dict[metric_] = data_randombackground

            bplot1 = axs[i].boxplot(
                data_,
                vert=True,  # vertical box alignment
                patch_artist=True,  # fill with color
                labels=enc_ls_,
                flierprops=flierprops__, boxprops=boxprops__,medianprops=medianprops__,
                capprops=dict(color='black'),
                whiskerprops=dict(color='black'),

            )  # will be used to label x-ticks
            bplot2 = axs[i].boxplot(
                data_ext,
                vert=True,  # vertical box alignment
                patch_artist=True,  # fill with color
                labels=enc_ls_,
                flierprops=flierprops__, boxprops=boxprops__,medianprops=medianprops__ext,
                capprops=dict(color='black'),
                whiskerprops=dict(color='black'),

            )
            if self.include_random_background_comparison_:  # randombackground
                bplot3 = axs[i].boxplot(
                    data_randombackground,
                    vert=True,  # vertical box alignment
                    patch_artist=True,  # fill with color
                    labels=enc_ls_,
                    flierprops=flierprops__, boxprops=boxprops__, medianprops=medianprops__randombackground,
                    capprops=dict(color='black'),
                    whiskerprops=dict(color='black'),

                )

            axs[i].set_title(metric_.replace('beta',str(self.f_beta_)), fontsize=fntsz_)
            if i == 3:
                axs[i].set_title('Final Model Performances (' + str(self.num_rerurun_model_building) + ' Rounds)\n' + str(metric_), fontsize=fntsz_)  # ,fontweight='bold')


            # # update x-axis labels
            tick_lab_list__ = []
            for x in self.feature_encoding_ls:
                tick_lab_list__.append('')
            if self.include_random_background_comparison_:
                for x in self.feature_encoding_ls:
                    tick_lab_list__.append('')
            for x in self.feature_encoding_ls:
                tick_lab_list__.append(feature_encodings_dict[x])

            # axs[i].set_xticklabels([feature_encodings_dict[x] for x in self.feature_encoding_ls], fontsize=fntsz_, rotation=90)
            axs[i].set_xticklabels(tick_lab_list__, fontsize=fntsz_, rotation=90)
            # axs[i].set_xticklabels(['','','','one-hot','gensim-weights','gensim-values'], fontsize=fntsz_, rotation=90)

            axs[i].set_ylim(0, 1)
            axs[i].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            axs[i].tick_params(axis='y', labelsize=fntsz_)

            # self.autolabel_boxplot_below(bplot1['caps'][::2], bplot1['medians'], axs[i], label_color = testing_set_plot_color)
            # self.autolabel_boxplot_below(bplot2['caps'][::2], bplot2['medians'], axs[i], label_color = external_set_plot_color)

        fig.suptitle('F-Score ONLY Final Models and applied to External Dataset ' + str(self.num_rerurun_model_building) +
                     ' rounds' + '\n' + self.output_run_file_info_string_.replace('_', ' ').replace(self.region_.replace('_', '-'), self.region_.replace('_', '-') + '\n'), fontsize=9)

        # # Each column of final_detailed_metric_df contains a single round for a single embedding type
        # # For exporting data used to make boxplots
        # bxplt_data_df = pd.DataFrame(bxplt_data_dict)
        # fnm_ = (self.output_directory + 'data/' + 'boxplot_performance_metrics_' + str(self.num_rerurun_model_building) + '-rnds_final_per-metric.csv')
        # bxplt_data_df.to_csv(fnm_, index=True)
        # logging.info("Final Models Performance Metrics used to make boxplots Dataframe saved to:\n\t"+fnm_)

        # bxplt_data_ext_df = pd.DataFrame(bxplt_data_ext_dict)
        # fnm_ = (self.output_directory + 'data/' + 'boxplot_performance_metrics_' + str(self.num_rerurun_model_building) + '-rnds_final_ext-data-eval_per-metric.csv')
        # bxplt_data_ext_df.to_csv(fnm_, index=True)
        # logging.info("Final Models Performance Metrics used to make boxplots (when Evaluated on External Dataset) Dataframe saved to:\n\t"+fnm_)

        # Add legend for parameter values

        legend_elements = [
            Line2D([0], [0],
                   color=testing_set_plot_color,  # embd_color_dict[embd_][val__],
                   lw=4, label='Test Set'),
            Line2D([0], [0],
                   color=external_set_plot_color,  # color_,  # embd_color_dict[embd_][val__],
                   lw=4, label='External Dataset')
        ]
        if self.include_random_background_comparison_:  # randombackground
            legend_elements.append(
                Line2D([0], [0],
                       color='grey',  # color_,  # embd_color_dict[embd_][val__],
                       lw=4, label='Random Background')
            )
        axs[-1].legend(
            handles=legend_elements, loc='upper left', frameon=False, bbox_to_anchor=(0, 1), fontsize=12
            #title=self.parameter_to_optimize, title_fontsize=12,
        )
        axs[-1].axis('off')


        fig.tight_layout()

        # ** SAVE FIGURE **
        plt.rcParams['svg.fonttype'] = 'none'  # exports text as strings rather than vector paths (images)
        fnm_ = (self.output_directory + 'figures/' + 'f-score_bxp_' + str(self.num_rerurun_model_building) + '-rnds_final_and_external')
        fnm_svg_ = (self.output_directory + 'figures/' + 'svg_figs/' + 'f-score_bxp_' + str(self.num_rerurun_model_building) + '-rnds_final_and_external')
        fig.savefig(fnm_svg_.split('.')[0] + '.svg', format='svg', transparent=True)
        fig.savefig(fnm_.split('.')[0] + '.png', format='png', dpi=300, transparent=False)
        logging.info('Figure saved to:'+str(fnm_) + '.png'.replace(self.output_directory, '~/'))
        return


# TODO: (maybe?) export metrics/scores from parameter optimization to a file?
# TODO: (maybe?) For Semi-supervised: Add back in the unlabelled data to the testing set?
# TODO: add back in undefined middle values and evaluate model (possibly using needle-in-haystack method)
# TODO: for final boxplots by parameter value include count of models with each parameter value (on x-axis)


# drb = DataRepresentationBuilder(model_type__ = 'random-forest', parameter_to_optimize__ = 'kmer-size', custom_parameter_values_to_loop__ = [3,8,9] ,num_rerurun_model_building__=5,flank_len__=10,
#                                 encoding_ls__ = ['one-hot', 'ann-word2vec-gensim'])#, 'bow-gensim', 'ann-keras', 'bow-countvect'])

# drb = DataRepresentationBuilder(model_type__ = 'semi-sup-label-spreading', parameter_to_optimize__ = 'kmer-size', custom_parameter_values_to_loop__ = [3,9] ,num_rerurun_model_building__=2,flank_len__=10,
#                                 encoding_ls__ = ['one-hot', 'ann-word2vec-gensim'])#, 'bow-gensim', 'ann-keras', 'bow-countvect'])

# drb = DataRepresentationBuilder(model_type__ = 'linear-classification', parameter_to_optimize__ = 'kmer-size', custom_parameter_values_to_loop__ = [3,9] ,num_rerurun_model_building__=2,flank_len__=10,
#                                 encoding_ls__ = ['one-hot', 'ann-word2vec-gensim'])#, 'bow-gensim', 'ann-keras', 'bow-countvect'])


