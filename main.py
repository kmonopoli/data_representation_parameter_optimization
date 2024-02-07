#!/opt/anaconda3/bin/python
#print("TEST - running Main 1: before import statements")
import pandas as pd
import numpy as np
import os
# warnings.filterwarnings('ignore')
from gensim import corpora
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt
from matplotlib import rcParams
from datetime import datetime
import calendar
import openpyxl

# from embedding_methods.embedding_methods import embed_sequences_with_ann
# from embedding_methods.embedding_methods import embed_sequences_with_gensim
# from embedding_methods.embedding_methods import one_hot_encode_sequences

from embedding_methods.sirna_model_building_helper_methods import classify
from embedding_methods.sirna_model_building_helper_methods import get_flanking_sequence


import matplotlib.pylab as pylab
#print("TEST - running Main 2: After import statements")

params = {'legend.fontsize': 12,
          'figure.figsize': (6, 4),
          'axes.labelsize': 12,
          'axes.titlesize': 12,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          #'font.family': 'Arial',
          }
pylab.rcParams.update(params)

#########################################################################################################################################
#####################################      Constants (Global Variables/Dictionaries)        #############################################
#########################################################################################################################################

# For organizing output files, everything will go into this folder
all_output_dir = 'output_model_fitting/'

# Directory path to and file holding siRNA data
input_data_dir = 'new_input_data/'
#input_data_dir = '/Users/kmonopoli/Dropbox (UMass Medical School)/compiling_sirna_screening_data/'
input_data_file = 'cleaned_no-bad-or-duplicate-screens_sirna_screen_data_4392sirnas-bdna-75-genes_JAN-29-2024.csv'
#input_data_file = 'cleaned_no-bad-or-duplicate-screens_sirna_screen_data_(4392sirnas-bdna|75-genes)_JAN-29-2024.csv'


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
    #'unlab-randomized': input_data_dir + '/semisupervised_for_cluster_oct-19-2023/' + 'unlabelled_randomized_sirna_data_37050-sirnas_hm_SEP-12-2023_pog.csv',
    'unlab-randomized': input_data_dir +'unlabelled_randomized_sirna_data_38024-sirnas_SEP-12-2023.csv',


    # Unlabelled siRNA Data Generated from Targeted Transcripts - evenly distributed throughout
    # 'sequences-from-targeted-transcripts':None, # TODO: generate sequences-from-targeted-transcripts unlabelled dataset

    # Unlabelled siRNA Data Generated from Targeted Transcripts - weighted by representation
    'weighted-sequences-from-targeted-transcripts': input_data_dir + 'unlabelled_weighted-sequences-from-targeted-transcripts_sirna_data_37928-sirnas_SEP-18-2023.csv',

    # Unlabelled siRNA Data Generated from Transcriptome (including untargeted transcripts)
    'sequences-from-all-transcripts': input_data_dir + 'unlabelled_sequences-from-species-transcriptomes_sirna_data_39000-sirnas_OCT-5-2023.csv',
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


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.linear_model import LogisticRegression
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
    'window_size':'w',
    'word_frequency_cutoff':'q',
    'ANN_output_dimmension':'d',
    'unlabeled_data_type':'u',
    'unlabeled_data_size':'z',
    'feature_encoding':'e',
    None:'n',
    # Note: if add more parameters to optimize, add them here
}
feature_encodings_dict = {
    'one-hot':'oh',
    'bow_countvect':'bowcv',
    'bow_gensim':'bowgen',
    'ann_keras':'annk',
    'ann_word2vec_gensim':'w2v'
}

feature_encodings_titles_dict = {
    'one-hot':'One-Hot',
    'bow_countvect':'BOW cv',
    'bow_gensim':'BOW g',
    'ann_keras':'ANN',
    'ann_word2vec_gensim':'Word2Vec',
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
    'window_size': [1, 2, 3, 4, 5, 7, 10, 15, 20],
    'word_frequency_cutoff': [1, 2, 3, 4, 5, 10],
    'ANN_output_dimmension': [4, 10, 20, 50, 100, 200, 500, 1000],
    'unlabeled_data_type': unlabelled_data_types,
    'unlabeled_data_size': [0.00, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00],  # decimal proportion of whole unlabelled dataset to take (0.50 = 50%, 1.00 = 100%)
    'feature_encoding':encodings_ls,
    None: [],
    # Note: if add more parameters to optimize, add them here
}


#print("TEST - running Main 3: After global variables")




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
                 encoding_ls__ = ['one-hot', 'ann_word2vec_gensim'],#, 'bow_gensim', 'ann_keras', 'bow_countvect'],
                 metric_used_to_id_best_po__='F-Score',
                 f_beta__ = 0.5,
                 ):
        '''
        #########################################################################################################################################
        #####################################        Dataset Parameters (Instance Variables)        #############################################
        #########################################################################################################################################
        '''
        pd.set_option('display.max_columns', None)
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
                    if (l_ > 100) or (l_ < 0) or type(l_ != int):
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
        self.metric_used_to_id_best_po = metric_used_to_id_best_po__   # TODO: change to different metric?
        self.f_beta_ = f_beta__

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
                print("NOTE: running without optimizing any parameters")
            else:
                self.param_opt_working_keys_ls = []


        if self.parameter_to_optimize == 'kmer-size':
            self.kmer_size_ = 'PARAMOPT' 
        elif self.parameter_to_optimize == 'flank-length':
            self.flank_len_ = 'PARAMOPT' 
        elif self.parameter_to_optimize == 'model':
            self.model_type_ = 'PARAMOPT' 
        elif self.parameter_to_optimize == 'window_size':
            self.window_size_ = 'PARAMOPT' 
        elif self.parameter_to_optimize == 'word_frequency_cutoff':
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
            self.parameter_to_optimize = None
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

        self.all_data_label_str_ = (  # general label describing siRNA data used for model building
                '-'.join(self.species_ls) +
                '_' + self.chemical_scaffold_lab +
                '_' + '-'.join(self.screen_type_ls) +

                '_' + param_norm_label_dict[self.normalized_] +
                '_effco-' + str(self.effco_) +
                '|ineffco-' + str(self.ineffco_) + '-' + remove_undefined_label_dict[self.remove_undefined_]
        ).replace(' ', '_')

        self.abbrev_all_data_label_str_ = (
                '-'.join([x[0] for x in self.species_ls]) +
                '_' + self.chemical_scaffold_lab +
                '_' + '-'.join(self.screen_type_ls) +
                '_' + '-'.join([feature_encodings_dict[e] for e in self.feature_encoding_ls]) +
                '_' + param_norm_label_dict[self.normalized_].replace('alized', '').replace('-', '') +
                '_' + str(self.effco_) +
                '-' + str(self.ineffco_) + '-' + remove_undefined_label_dict_abbrev[self.remove_undefined_]
        ).replace(' ', '_')
            
        print("Construction complete!")
        print("Creating processed datasets...")
        self.create_processed_datasets()
        print("Creating processed datasets complete!")
        print("Running model fittings...")
        pr_po, k_po, pr_f, m_f, k_f = self.run_model_fittings()
        print("Model Fittings complete!")
        print("Ploting precision-recall curves from Parameter Optimization...")
        self.plot_param_opt_precision_recall_curves()
        print("Ploting precision-recall curves from Final Model Building...")
        self.plot_final_model_precision_recall_curves() # TODO: finish writing method self.plot_param_opt_precision_recall_curves
        print("Curve plotting complete!")
        print("Plotting box plots from Parameter Optimization...")
        self.plot_param_opt_model_box_plots()
        print("Plotting box plots from Final Model Building...")
        self.plot_final_model_box_plots()
        self.plot_final_model_box_plots_per_param_val()
        print("Box plotting complete!")

        print("PROCESS FINISHED")
        #return ## End constructor

    def plot_thresholds(self, df_, figure_label_, output_dir__='', savefig=True):
        fig, ax = plt.subplots()
        fig.set_size_inches(w=5, h=4)

        colors_ls = [x.replace('inefficient', '#3AA6E2').replace('efficient', '#F7B531').replace('undefined', '#B6B6B7') for
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
        #caplines.set_marker(None)
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
            Patch(facecolor='#F7B531', edgecolor=None,
                  label=('< ' + str( self.effco_) + '% : Efficient (' + str(len(df_[df_['class'] == 'efficient'])) + ' siRNAs)')),
            Patch(facecolor='#3AA6E2', edgecolor=None, label=('≥ ' + str( self.ineffco_) + '% : Inefficient (' + str(
                len(df_[df_['class'] == 'inefficient'])) + ' siRNAs)')),
            Patch(facecolor='#B6B6B7', edgecolor=None,
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
            # print('Figure saved to:',fnm_svg_+'.svg')

            fig.savefig(fnm_.split('.')[0] + '.png', format='png', dpi=300, transparent=False)
            print('Figure saved to:', fnm_ + '.png')

    def plot_proportions_pie(self, figure_label_, output_dir__, df_, undefined_df_, train_split_df_, split_initial_df_,
                             df_train_kfld_, df_test_kfld_, df_paramopt_kfld_, round_ct_, savefig=True):
        # Checking porportions of partitioned data roughly match input parameters

        fig, ax = plt.subplots(1)  # 1,3)
        fig.set_size_inches(w=5, h=5)
        # For Plotting
        import matplotlib.pylab as pylab
        params = {'legend.fontsize': 12,
                  'font.size': 12,
                  #'font.family': 'Arial',
                  }
        pylab.rcParams.update(params)

        ax.set_title('Round ' + str(round_ct_ + 1) + ' Partition\nTotal siRNAs: ' + str(len(train_split_df_)))
        ax.pie(
            [len(df_train_kfld_), len(df_test_kfld_), len(df_paramopt_kfld_)],  # len(split_initial_df_) ],
            autopct='%1.f%%',
            startangle=90,
            colors=['#DB7AC8', '#FECC0A', '#28A18B'],  # '#28A18B'],
        )

        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor='#DB7AC8', edgecolor=None, label=(
                    str(100 - self.split_set_size_pcnt_) + '% Round ' + str(round_ct_ + 1) + ' Training Dataset (' + str(
                len(df_train_kfld_)) + ')')),
            Patch(facecolor='#FECC0A', edgecolor=None, label=(
                    str( self.test_set_size_pcnt_) + '% Round ' + str(round_ct_ + 1) + ' Testing Dataset (' + str(
                len(df_test_kfld_)) + ')')),
            Patch(facecolor='#28A18B', edgecolor=None, label=(str( self.paramopt_set_size_pcnt_) + '% Round ' + str(
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
            # print('Figure saved to:',fnm_svg_+'.svg')

            fig.savefig(fnm_.split('.')[0] + '.png', format='png', dpi=300, transparent=False)
            print('Figure saved to:', fnm_ + '.png')

    def create_processed_datasets(self):
        print("Creating processed datasets...")
        # Get date - for labeling and saving files
        self.date_ = calendar.month_abbr[datetime.now().month].upper() + '-' + str(datetime.now().day) + '-' + str(
            datetime.now().year)
        print("Date set successfully!",str(self.date_))
        # Read in Data
        try:
            print("Trying to read in xlsx data")
            self.df = pd.read_excel(input_data_dir + input_data_file)
            print("Successfully read in xlsx data")
        except:
            print("Trying to read in .csv data")
            self.df = pd.read_csv(input_data_dir + input_data_file)
            print("Successfully read in .csv data")

        self.df.drop(columns=['expression_replicate_1', 'expression_replicate_2', 'expression_replicate_3', 'ntc_replicate_1', 'ntc_replicate_2', 'ntc_replicate_3', 'untreated_cells_replicate_1', 'untreated_cells_replicate_2', 'untreated_cells_replicate_3'], inplace=True)

        print(self.df['chemical_scaffold'].value_counts())
        print()
        print(self.df['screen_type'].value_counts())
        print()
        print(self.df['species'].value_counts())

        print("Selecting data with screen type:\n", self.screen_type_ls)
        self.df = self.df[self.df['screen_type'].isin(self.screen_type_ls)]
        print("Selecting data with species:\n", self.species_ls)
        self.df = self.df[self.df['species'].isin(self.species_ls)]
        print("Selecting data with chemical scaffold:\n", self.chemical_scaffold_ls)
        self.df = self.df[self.df['chemical_scaffold'].isin(self.chemical_scaffold_ls)]

        ########################################################################
        ##                     ~*~ Select & Clean Data ~*~                    ##
        ########################################################################

        print('region_ =', self.region_)
        # Define key to identify column with sequence data used for model building
        self.flank_seq_working_key = 'seq'


        # If using flanking sequence Remove sequences missing flanking regions
        if 'flank' in self.region_:  # check if using flanking region
            if self.parameter_to_optimize != 'flank-length':
                self.flank_seq_working_key += '_flank-' + str(self.flank_len_) + 'nts'

            # 1) Drop sequences missing flanking sequences
            len_before = len(self.df)
            self.df = self.df[self.df['flanking_sequence_1'].notna()]
            self.df.reset_index(drop=True, inplace=True)
            print("Dropped", len_before - len(self.df), "siRNAs for missing flanking sequence data (", len(self.df), "Remaining)")

            # 2) Drop sequences with mismatch in 16mer (when finding flanks) # TODO: UPDATE THIS TO INCLUDE MISMATCH IN 16MERS?
            len_before = len(self.df)
            self.df = self.df[self.df['mismatch_16mer_for_flanks'] == '16mer perfect match to target']
            self.df.reset_index(inplace=True, drop=True)
            print("Dropped", len_before - len(self.df), 'siRNAs for having mismatch 16mer')

            # 3) Drop sequences where could not get longest flanking sequence
            if self.parameter_to_optimize == 'flank-length':
                self.longest_flank_len = max(self.param_values_to_loop_)
            else:
                self.longest_flank_len = self.flank_len_

            len_before = len(self.df)
            indxs_too_short_flanks_ = list(self.df[self.df.apply(lambda x: get_flanking_sequence(x['16mer_complementary_region'], x['flanking_sequence_1'], self.longest_flank_len, True), axis=1).isna()].index)
            self.df.drop(index=indxs_too_short_flanks_, inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            print("Dropped", len_before - len(self.df), "siRNAs because could not get longest flanking sequence (", len(self.df), "Remaining)")

            # 4) Take longest flank_len_ from flank_lens_to_loop_ and check there aren't any sequences that aren't long enough to extract longest flanking sequence
            if 'True' in [str(x) for x in list(set(self.df.apply(lambda x: get_flanking_sequence(x['16mer_complementary_region'], x['flanking_sequence_1'], self.longest_flank_len, True), axis=1).isna()))]:
                raise Exception("ERROR: some target sequences' surrounding regions are too short to extract longest flanking region")

            if self.parameter_to_optimize != 'flank-length':
                self.longest_flank_len = max(self.param_values_to_loop_)
                if 'target' in self.region_:
                    self.flank_seq_working_key += '_target'
                    print("***********")
                    print(self.df.apply(lambda x: get_flanking_sequence(x['16mer_complementary_region'], x['flanking_sequence_1'], self.flank_len_, True), axis=1))
                    self.df[self.flank_seq_working_key] = self.df.apply(lambda x: get_flanking_sequence(x['16mer_complementary_region'], x['flanking_sequence_1'], self.flank_len_, True), axis=1)
                else:
                    self.flank_seq_working_key += '_NO-target'
                    self.df[self.flank_seq_working_key] = self.df.apply(lambda x: get_flanking_sequence(x['16mer_complementary_region'], x['flanking_sequence_1'], self.flank_len_, False), axis=1)
        else:
            if self.parameter_to_optimize != 'flank-length':
                self.flank_seq_working_key = '20mer_targeting_region'


        # If optimizing Flank length parameter generate sequences with different flank lengths
        if self.parameter_to_optimize == 'flank-length':
            self.flank_seq_working_key = None
            for flank_len_ in self.param_values_to_loop_:  # if parameter_to_optimize == 'flank-len':
                if flank_len_ == 0:
                    flank_seq_working_key__ = '20mer_targeting_region'
                else:
                    flank_seq_working_key__ = 'seq'
                    flank_seq_working_key__ += '_flank-' + str(flank_len_) + 'nts'
                    print("Flanking sequence size (per side):", flank_len_, 'nts')
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
                self.param_opt_working_keys_ls.append(flank_seq_working_key__)
                #for f_ in self.param_opt_working_keys_ls:
                    #print('\n', f_, ' ', list(self.df[f_].apply(lambda x: len(x)).value_counts().index)[0])

        self.df.sort_values(by=[self.expr_key], inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print((self.df[['chemical_scaffold', 'screen_type', 'species']].value_counts()))
        self.df['class'] = self.df[self.expr_key].apply(lambda x: classify(x, self.effco_, self.ineffco_))
        print(self.df['class'].value_counts())
        # Convert classes into form that can be read by kfld splitter (0's and 1's and -1's for UNLABELLED)
        self.df['numeric_class'] = [int(x.replace('inefficient', '0').replace('efficient', '1').replace('undefined', '-1')) for x in list(self.df['class'])]


        self.load_in_unlab_data()
        self.perform_feature_embedding()
        self.split_train_test_paramopt()


    def load_in_unlab_data(self):
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
            print('region_ =', self.region_)
            # Define key to identify column with sequence data used for model building
            if 'flank' in self.region_:  # check if using flanking region
                # Remove sequences missing flanking regions (not necessary, undefined data all have flanks)
                # Take longest flank_len_ from flank_lens_to_loop_ and check there aren't any sequences that aren't long enough to extract longest flanking sequence
                print("Longest Flanking sequence size (per side):\n", self.longest_flank_len, 'nts')
                indxs_to_drop_ = list(self.df_unlab[(self.df_unlab.apply(lambda x: get_flanking_sequence(x['16mer_complementary_region'], x['flanking_sequence_1'], self.longest_flank_len, True), axis=1)).isna()].index)
                print("\n\n\nDropping", len(indxs_to_drop_), 'unlabelled siRNAs because could not determine flanking sequence')
                self.df_unlab.drop(index=indxs_to_drop_, inplace=True)
                self.df_unlab.reset_index(inplace=True, drop=True)
                print("Now have", len(self.df_unlab), 'siRNAs in unlabelled dataset')
            # If optimizing Flank length parameter generate sequences with different flank lengths
            if self.parameter_to_optimize == 'flank-length':
                print("Getting flanking sequences for unlabeled data...")
                for flank_len_, flank_seq_working_key in zip(self.param_values_to_loop_, self.param_opt_working_keys_ls):  # if parameter_to_optimize == 'flank-len':
                    print("Flanking sequence size (per side):", flank_len_, 'nts')
                    # Get flanking sequence of desired length along with target region (drop sequences that don't have long enough flanking sequences) and create new column to store sequences
                    if 'target' in self.region_:
                        self.df_unlab[flank_seq_working_key] = self.df_unlab.apply(lambda x: get_flanking_sequence(x['16mer_complementary_region'], x['flanking_sequence_1'], flank_len_, True), axis=1)
                    else:
                        self.df_unlab[flank_seq_working_key] = self.df_unlab.apply(lambda x: get_flanking_sequence(x['16mer_complementary_region'], x['flanking_sequence_1'], flank_len_, False), axis=1)
                # for f_ in self.param_opt_working_keys_ls:
                #     print('\n', f_, ' ', list(self.df_unlab[f_].apply(lambda x: len(x)).value_counts().index)[0])
                for flank_seq_working_key in self.param_opt_working_keys_ls:
                    ## Drop Unlabelled siRNAs where the working sequence could not be determined
                    len_before = len(self.df_unlab)
                    self.df_unlab.drop(index=self.df_unlab[self.df_unlab[flank_seq_working_key].isna()].index, inplace=True)
                    self.df_unlab.reset_index(inplace=True, drop=True)
                    print('Dropped', len_before - len(self.df_unlab), 'unlabelled siRNAs where the working (i.e. flanking) sequence could not be determined for flank_seq_working_key:', flank_seq_working_key)
            else:
                print("Getting flanking sequences for unlabeled data...")
                print("Flanking sequence size (per side):", self.flank_len_, 'nts')
                # Get flanking sequence of desired length along with target region (drop sequences that don't have long enough flanking sequences) and create new column to store sequences
                if 'target' in self.region_:
                    self.df_unlab[self.flank_seq_working_key] = self.df_unlab.apply(lambda x: get_flanking_sequence(x['16mer_complementary_region'], x['flanking_sequence_1'], self.flank_len_, True), axis=1)
                else:
                    self.df_unlab[self.flank_seq_working_key] = self.df_unlab.apply(lambda x: get_flanking_sequence(x['16mer_complementary_region'], x['flanking_sequence_1'], self.flank_len_, False), axis=1)

                
                ## Drop Unlabelled siRNAs where the working sequence could not be determined
                len_before = len(self.df_unlab)
                self.df_unlab.drop(index=self.df_unlab[self.df_unlab[self.flank_seq_working_key].isna()].index, inplace=True)
                self.df_unlab.reset_index(inplace=True, drop=True)
                print('Dropped', len_before - len(self.df_unlab), 'unlabelled siRNAs where the working (i.e. flanking) sequence could not be determined for flank_seq_working_key:', self.flank_seq_working_key)

            print("Selecting data with species:\n", self.species_ls)
            self.df_unlab = self.df_unlab[self.df_unlab['species'].isin(self.species_ls)]

            # TODO: for Parameter Optimization of unlabelled data set SIZE , add option to alter dataset size - BUT DO THIS CAREFULLY SO DON'T HAVE A LOT OF DATA IN RAM
            print('Number of unlabelled siRNAs loaded:', len(self.df_unlab))

            # Format Unlabelled to match Labelled Data
            # Add missing columns to self.df_unlab
            self.df_unlab['class'] = ['undefined'] * len(self.df_unlab)
            self.df_unlab['numeric_class'] = [-1] * len(self.df_unlab)
            for c in ['experiment_name', 'expression_percent_normalized_by_max_min', 'expression_percent_normalized_by_z_score', 'standard_deviation_normalized_by_subtracting_mean', 'cleaned_bdna_p2p3p5_human-mouse']:
                self.df_unlab[c] = np.nan
            self.df_unlab = self.df_unlab[list(self.df.columns)] # reorder columns to match labelled self.df
            # Combine unlabelled and labelled data into a single dataframe
            #self.df_before_adding_u = self.df.copy() # make a backup of self.df
            self.indxs_labeled_data = list(self.df.index)
            self.indxs_mid_undefined = list(self.df[self.df['numeric_class'] == -1].index)
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
            print(self.model_type_, "Does not use unlabeled data")
            self.indxs_mid_undefined = list(self.df[self.df['numeric_class'] == -1].index)
            self.indxs_labeled_data = list(self.df.index)



    def perform_feature_embedding(self):
        ###############################################################################################################
        ###############################          Perform Feature Embedding        #####################################
        ###############################################################################################################

        # TODO: update to work for semisupervised? ~ seem to be having a problem when running semi-sup encoding data


        from embedding_methods.embedding_methods import one_hot_encode_sequences
        from embedding_methods.embedding_methods import embed_sequences_with_bow_countvect
        from embedding_methods.embedding_methods import embed_sequences_with_gensim_doc2bow
        from embedding_methods.embedding_methods import embed_sequences_with_keras
        from embedding_methods.embedding_methods import embed_sequences_with_gensim_word2vec

        #['one-hot', 'bow_countvect', 'bow_gensim', 'ann_keras', 'ann_word2vec_gensim']

        if self.parameter_to_optimize == 'kmer-size':
            kmer_sizes_ls = self.param_values_to_loop_
        else:
            kmer_sizes_ls = [self.kmer_size_]

        if self.parameter_to_optimize == 'flank-length':
            flank_seq_working_key__ls = self.param_opt_working_keys_ls
        else:
            flank_seq_working_key__ls = [self.flank_seq_working_key]

        for kmer_ in kmer_sizes_ls:
            print('kmer size:', kmer_)
            for flank_seq_working_key__ in flank_seq_working_key__ls:
                print('flank_seq_working_key:',flank_seq_working_key__)
                for encoding_ in self.feature_encoding_ls:
                    print('encoding:', encoding_)
                    if encoding_ == 'one-hot': ### One-Hot Encoding ###
                        self.df['one-hot_encoded_' + flank_seq_working_key__ + '_kmer-' + str(kmer_)] = one_hot_encode_sequences(
                            list(self.df[flank_seq_working_key__]))
                    elif encoding_ == 'bow_countvect':### Bag-of-Words Embedding with Sklearn CountVectorizer###
                        self.df['bow_countvect_encoded_' + flank_seq_working_key__ + '_kmer-' + str(kmer_)] = embed_sequences_with_bow_countvect(
                            list(self.df[flank_seq_working_key__]), kmer_size_=kmer_, window_size_=self.window_size_,word_freq_cutoff_=self.word_freq_cutoff_)  # , output_directory = output_directory)
                    elif encoding_ == 'bow_gensim':### Bag-of-Words Embedding with Gensim Doc2bow ###
                        self.df['bow_gensim_encoded_' + flank_seq_working_key__ + '_kmer-' + str(kmer_)] = embed_sequences_with_gensim_doc2bow(
                            list(self.df[flank_seq_working_key__]), kmer_size_=kmer_, window_size_=self.window_size_,word_freq_cutoff_=self.word_freq_cutoff_)  # , output_directory = output_directory)
                    elif encoding_ == 'ann_keras': ### Deep Embedding with ANN - Keras ###
                        self.df['ann_keras_encoded_' + flank_seq_working_key__ + '_kmer-' + str(kmer_)] = embed_sequences_with_keras(
                            list(self.df[flank_seq_working_key__]), kmer_size_=kmer_, window_size_=self.window_size_,output_dimmension_=self.output_dimmension_)  # , output_directory = output_directory)
                    elif encoding_ == 'ann_word2vec_gensim':### Deep Embedding with Word2Vec - Gensim ###
                        self.df['ann_word2vec_gensim_encoded_' + flank_seq_working_key__ + '_kmer-' + str(kmer_)] = embed_sequences_with_gensim_word2vec(
                            list(self.df[flank_seq_working_key__]), kmer_size_=kmer_,window_size_=self.window_size_,word_freq_cutoff_=self.word_freq_cutoff_)  # , output_directory = output_directory)
                    else:
                        raise ValueError('ERROR: encoding '+str(encoding_)+' is not supported')

        ## Print out Encoded Vector Lengths
        print_vector_lens_start_ls = []
        print_vector_lens_end_ls = []
        for kmer_ in kmer_sizes_ls:
            for flank_seq_working_key__ in flank_seq_working_key__ls:
                for encoding_ in self.feature_encoding_ls:
                    v_len_start_ = len(self.df.iloc[0][encoding_ + '_encoded_' + flank_seq_working_key__ + '_kmer-' + str(kmer_)])
                    v_len_end_ = len(self.df.iloc[-1][encoding_+'_encoded_' + flank_seq_working_key__ + '_kmer-' + str(kmer_)])
                    if encoding_ == 'one-hot':
                        print_vector_lens_start_ls.append(str(v_len_start_) + ' \t')
                        print_vector_lens_end_ls.append(str(v_len_end_)+' \t')
                    else:
                        print_vector_lens_start_ls.append(str(v_len_start_) + ' \t\t')
                        print_vector_lens_end_ls.append(str(v_len_end_) + ' \t\t')
        print('\n\nEncoded vector lengths:\n\n')
        print('\t '.join(encodings_ls))
        print(' '.join(print_vector_lens_start_ls))
        print(' '.join(print_vector_lens_end_ls))
        # print(len(self.df.iloc[0]['one-hot_encoded_' + self.flank_seq_working_key + '_kmer-' + str(kmer_)]), '\t',
        #       len(self.df.iloc[0]['bow_countvect_encoded_' + self.flank_seq_working_key + '_kmer-' + str(kmer_)]), '\t\t',
        #       len(self.df.iloc[0]['bow_gensim_encoded_' + self.flank_seq_working_key + '_kmer-' + str(kmer_)]), '\t\t',
        #       len(self.df.iloc[0]['ann_keras_encoded_' + self.flank_seq_working_key + '_kmer-' + str(kmer_)]), '\t\t',
        #       len(self.df.iloc[0]['ann_word2vec_gensim_encoded_' + self.flank_seq_working_key + '_kmer-' + str(kmer_)]))
        #
        # print(len(self.df.iloc[-1]['one-hot_encoded_' + self.flank_seq_working_key + '_kmer-' + str(kmer_)]), '\t',
        #       len(self.df.iloc[-1]['bow_countvect_encoded_' + self.flank_seq_working_key + '_kmer-' + str(kmer_)]), '\t\t',
        #       len(self.df.iloc[-1]['bow_gensim_encoded_' + self.flank_seq_working_key + '_kmer-' + str(kmer_)]), '\t\t',
        #       len(self.df.iloc[-1]['ann_keras_encoded_' + self.flank_seq_working_key + '_kmer-' + str(kmer_)]), '\t\t',
        #       len(self.df.iloc[-1]['ann_word2vec_gensim_encoded_' + self.flank_seq_working_key + '_kmer-' + str(kmer_)]))



    def split_train_test_paramopt(self):
        ##############################################################################################################
        #################################    Split Dataset into Train:Paramopt:Test    ###############################
        ##############################################################################################################
        # Create a unique self.datasplit_id_ for data splitting so can re-run with same data
        from random import randint
        # self.datasplit_id_ = 'SUPk'+str(randint(100, 999) ) # ID used to find output data file
        self.datasplit_id_ = model_type_dict[self.model_type_] + self.parameter_to_optimize[0].upper() + str(randint(10000, 99999))  # ID used to find output data file
        # Check that datasplit_id_ doesn't exist already
        while self.datasplit_id_ in [x[0:7] for x in os.listdir(all_output_dir)]:
            # self.datasplit_id_ = 'SUPk'+str(randint(100, 999) ) # ID used to find output data file
            self.datasplit_id_ = model_type_dict[self.model_type_] + self.parameter_to_optimize[0].upper() + str(randint(10000, 99999))  # ID used to find output data file
            #print("self.datasplit_id_ already exists, generating new ID...")
            #print("NEW self.datasplit_id_ | Randomized " + str(len(self.datasplit_id_)) + "-digit ID for this Set of Rounds:\t " + self.datasplit_id_)
        print("self.datasplit_id_ | Randomized " + str(len(self.datasplit_id_)) + "-digit ID for this Set of Rounds:\t " + self.datasplit_id_)
        self.all_data_split_dir = 'data-' + self.datasplit_id_ + '_' + self.abbrev_all_data_label_str_.replace('|', '-') + '/'

        if not os.path.exists(all_output_dir + self.all_data_split_dir):
            os.makedirs(all_output_dir + self.all_data_split_dir)
            print("All datasets will be stored in:\n", os.getcwd() + all_output_dir + '\n\t' + self.all_data_split_dir)
            if not os.path.exists(all_output_dir + self.all_data_split_dir + 'figures/'):
                os.makedirs(all_output_dir + self.all_data_split_dir + 'figures/')
            if not os.path.exists(all_output_dir + self.all_data_split_dir + 'figures/svg_figs/'):
                os.makedirs(all_output_dir + self.all_data_split_dir + 'figures/svg_figs/')
            if not os.path.exists(all_output_dir + self.all_data_split_dir + 'datasets/'):
                os.makedirs(all_output_dir + self.all_data_split_dir + 'datasets/')

        else:
            raise Exception(
                'ERROR: directory with name ' + self.all_data_split_dir + ' exists. Check that self.datasplit_id_ is being randomized correctly')

        # Name and Plot Entire Dataset (excluding unlabelled data used for semi-supervised)
        all_data_label = "All siRNA Data"
        self.plot_thresholds(self.df.iloc[self.indxs_labeled_data], all_data_label, self.all_data_split_dir + 'figures/')

        # Undefined dataset holds all middle values (class = 'undefined' | numeric_class = -1 )
        self.mid_undef_df = self.df.iloc[self.indxs_mid_undefined].copy()

        # Remove undefined data
        self.df_noundef = self.df[self.df['numeric_class'] != -1].copy()
        self.df_noundef.reset_index(inplace=True, drop=False)

        # Name and Plot Undefined (excluded) Dataset
        undefined_label = "Undefined Data"
        self.plot_thresholds(self.mid_undef_df, undefined_label + "\nexcluded from training and evaluation - for now",self.all_data_split_dir + 'figures/')

        # Save Undefined dataset
        self.mid_undef_df_fnm = all_output_dir + self.all_data_split_dir + undefined_label.replace(' ', '_').replace('%','pcnt') + '_partition.csv'
        self.mid_undef_df.to_csv(self.mid_undef_df_fnm, index=False)
        print("Undefined Dataset saved to:\n\t", self.mid_undef_df_fnm)

        # Create 80:10:10 splits INDEPENDENTLY on the labelled (efficnet/inefficent) siRNA data
        print('Allowed percentage classification proportiond deviation: ' + str(self.allowed_classification_prop_deviation_pcnt_) + '% (' + str( int(np.round(len(self.df_noundef) * (self.allowed_classification_prop_deviation_pcnt_ / 100), 0))) + ' siRNAs)')
        for n_ in range(self.num_rerurun_model_building):  # [0:1]:
            need_to_resplit = True
            resplit_round_ct_ = 0
            while need_to_resplit:
                # print('Round:',n_+1,'/',num_rerurun_model_building)

                ## 1) Create Training set first --> split_initial and train_split_ dataframes for:
                #    80:10:10 --> train (80) : paramopt (10) : train (10)
                #    NOTE: split_initial will not contain unlabelled data
                from sklearn.model_selection import train_test_split

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
                    # print('ERROR: Need to resplit data due to Training set proportions (Round',resplit_round_ct_+1,')')
                    # print('** Dataset Excluding Undefined:',len(self.df_noundef),'siRNAs Total \t(',int(np.round(100*len(self.df_noundef)/len(self.df_noundef),0)),'% )','\n\t0:',self.df_noundef['numeric_class'].value_counts()[0],'siRNAs','\t\t(',int(starting_dataset_ineff_proportion_),'% )',' \n\t1:',self.df_noundef['numeric_class'].value_counts()[1],'siRNAs ','\t\t(',int(starting_dataset_eff_proportion_),'% )\n')
                    # print('** Training:',len(train_split_df),'siRNAs Total \t\t\t(',int(np.round(100*len(train_split_df)/len(self.df_noundef),0)),'% )','\n\t0:',train_split_df['numeric_class'].value_counts()[0],'siRNAs','\t\t(',int(train_dataset_ineff_proportion_),'% )',' \n\t1:',train_split_df['numeric_class'].value_counts()[1],'siRNAs ','\t\t(',int(train_dataset_eff_proportion_),'% )\n')
                    resplit_round_ct_ += 1

                elif (test_dataset_eff_proportion_ > (
                        starting_dataset_eff_proportion_ + self.allowed_classification_prop_deviation_pcnt_)) or (
                        test_dataset_eff_proportion_ < (
                        starting_dataset_eff_proportion_ - self.allowed_classification_prop_deviation_pcnt_)):
                    # print('ERROR: Need to resplit data due to Testing set proportions (Round',resplit_round_ct_+1,')')
                    # print('** Dataset Excluding Undefined:',len(self.df_noundef),'siRNAs Total \t(',int(np.round(100*len(self.df_noundef)/len(self.df_noundef),0)),'% )','\n\t0:',self.df_noundef['numeric_class'].value_counts()[0],'siRNAs','\t\t(',int(starting_dataset_ineff_proportion_),'% )',' \n\t1:',self.df_noundef['numeric_class'].value_counts()[1],'siRNAs ','\t\t(',int(starting_dataset_eff_proportion_),'% )\n')
                    # print('** Testing:',len(test_split_df),'siRNAs Total \t\t\t(',int(np.round(100*len(test_split_df)/len(self.df_noundef),0)),'% )','\n\t0:',test_split_df['numeric_class'].value_counts()[0],'siRNAs','\t\t(',int(test_dataset_ineff_proportion_),'% )',' \n\t1:',test_split_df['numeric_class'].value_counts()[1],'siRNAs ','\t\t(',int(test_dataset_eff_proportion_),'% )\n')
                    resplit_round_ct_ += 1

                elif (paramop_dataset_eff_proportion_ > (
                        starting_dataset_eff_proportion_ + self.allowed_classification_prop_deviation_pcnt_)) or (
                        paramop_dataset_eff_proportion_ < (
                        starting_dataset_eff_proportion_ - self.allowed_classification_prop_deviation_pcnt_)):
                    # print('ERROR: Need to resplit data due to Parameter Optimization set proportions (Round',resplit_round_ct_+1,')')
                    # print('** Dataset Excluding Undefined:',len(self.df_noundef),'siRNAs Total \t(',int(np.round(100*len(self.df_noundef)/len(self.df_noundef),0)),'% )','\n\t0:',self.df_noundef['numeric_class'].value_counts()[0],'siRNAs','\t\t(',int(starting_dataset_ineff_proportion_),'% )',' \n\t1:',self.df_noundef['numeric_class'].value_counts()[1],'siRNAs ','\t\t(',int(starting_dataset_eff_proportion_),'% )\n')
                    # print('** Parameter Optimization::',len(paramop_split_df),'siRNAs Total \t(',int(np.round(100*len(paramop_split_df)/len(self.df_noundef),0)),'% )','\n\t0:',paramop_split_df['numeric_class'].value_counts()[0],'siRNAs','\t\t(',int(paramop_dataset_ineff_proportion_),'% )',' \n\t1:',paramop_split_df['numeric_class'].value_counts()[1],'siRNAs ','\t\t(',int(paramop_dataset_eff_proportion_),'% )\n')
                    resplit_round_ct_ += 1

                else:
                    need_to_resplit = False

                    print('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Round ' + str(n_ + 1) + ' / ' + str(
                        self.num_rerurun_model_building) + ' Complete! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                    print('** Resplit', resplit_round_ct_, 'times')

                    print('Dataset Excluding Undefined:', len(self.df_noundef), 'siRNAs Total \t(',
                          int(np.round(100 * len(self.df_noundef) / len(self.df_noundef), 0)), '% )', '\n\t0:',
                          self.df_noundef['numeric_class'].value_counts()[0], 'siRNAs', '\t\t(',
                          int(starting_dataset_ineff_proportion_), '% )', ' \n\t1:',
                          self.df_noundef['numeric_class'].value_counts()[1], 'siRNAs ', '\t\t(',
                          int(starting_dataset_eff_proportion_), '% )\n')

                    print('Training:', len(train_split_df), 'siRNAs Total \t\t\t(',
                          int(np.round(100 * len(train_split_df) / len(self.df_noundef), 0)), '% )', '\n\t0:',
                          train_split_df['numeric_class'].value_counts()[0], 'siRNAs', '\t\t(',
                          int(train_dataset_ineff_proportion_), '% )', ' \n\t1:',
                          train_split_df['numeric_class'].value_counts()[1], 'siRNAs ', '\t\t(',
                          int(train_dataset_eff_proportion_), '% )\n')

                    print('Testing:', len(test_split_df), 'siRNAs Total \t\t\t(',
                          int(np.round(100 * len(test_split_df) / len(self.df_noundef), 0)), '% )', '\n\t0:',
                          test_split_df['numeric_class'].value_counts()[0], 'siRNAs', '\t\t(',
                          int(test_dataset_ineff_proportion_), '% )', ' \n\t1:',
                          test_split_df['numeric_class'].value_counts()[1], 'siRNAs ', '\t\t(',
                          int(test_dataset_eff_proportion_), '% )\n')

                    print('Parameter Optimization::', len(paramop_split_df), 'siRNAs Total \t(',
                          int(np.round(100 * len(paramop_split_df) / len(self.df_noundef), 0)), '% )', '\n\t0:',
                          paramop_split_df['numeric_class'].value_counts()[0], 'siRNAs', '\t\t(',
                          int(paramop_dataset_ineff_proportion_), '% )', ' \n\t1:',
                          paramop_split_df['numeric_class'].value_counts()[1], 'siRNAs ', '\t\t(',
                          int(paramop_dataset_eff_proportion_), '% )\n')

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
                        self.plot_thresholds(train_split_df, train_split__label + "\n of labeled (not undefined) data",
                                        self.all_data_split_dir + 'figures/')

                    # Name and Plot Testing Dataset
                    testing_data_label = 'ROUND-' + str(n_ + 1) + " Testing Data " + str(self.test_set_size_pcnt_) + "%"
                    if self.num_rerurun_model_building < 2:
                        self.plot_thresholds(test_split_df, testing_data_label + "\n of  evaluation (testing) dataset",
                                        self.all_data_split_dir + 'figures/')

                    # Name and Plot Parameter Optimization Dataset
                    paramopt_data_label = 'ROUND-' + str(n_ + 1) + " Parameter Optimization Data " + str(
                        self.paramopt_set_size_pcnt_) + "%"
                    if self.num_rerurun_model_building < 2:
                        self.plot_thresholds(paramop_split_df, paramopt_data_label + "\n of  evaluation (testing) dataset",
                                        self.all_data_split_dir + 'figures/')

                    ## Save Datasets

                    # Save  Training Dataset
                    train_split_df_fnm = all_output_dir + self.all_data_split_dir + 'datasets/' + train_split__label.replace(' ',
                                                                                                                             '_').replace(
                        '%', 'pcnt') + '_partition.csv'
                    train_split_df.to_csv(train_split_df_fnm, index=False)
                    print("split_initial Dataset saved to:\n\t", train_split_df_fnm)

                    # Save  Testing Dataset
                    testing_df_fnm = all_output_dir + self.all_data_split_dir + 'datasets/' + testing_data_label.replace(' ',
                                                                                                                         '_').replace(
                        '%', 'pcnt') + '_partition.csv'
                    test_split_df.to_csv(testing_df_fnm, index=False)
                    print("Testing Dataset saved to:\n\t", testing_df_fnm)

                    # Save   Parameter Optimization Dataset
                    paramopt_df_fnm = all_output_dir + self.all_data_split_dir + 'datasets/' + paramopt_data_label.replace(' ',
                                                                                                                           '_').replace(
                        '%', 'pcnt') + '_partition.csv'
                    paramop_split_df.to_csv(paramopt_df_fnm, index=False)
                    print("Parameter Optimization Dataset saved to:\n\t", paramopt_df_fnm)

                    # Name and Plot Partitions
                    partition_label = 'ROUND-' + str(n_ + 1) + '_pie'
                    if self.num_rerurun_model_building < 2:
                        self.plot_proportions_pie(partition_label, self.all_data_split_dir + 'figures/', self.df, self.mid_undef_df, train_split_df, split_initial_df, train_split_df, test_split_df, paramop_split_df, round_ct_=n_, savefig=True)

    #########################################################################################################################################
    ##################################################     Plot All Split Data (Optional)     ###############################################
    #########################################################################################################################################

    def plot_data_splits(self):
        if (self.plot_grid_splits_):
            # Plot Data Splitting for each round in a single figure
            if self.num_rerurun_model_building > 2:
                train_col = '#D46F37'
                test_col = '#4BB3B1'
                paramopt_col = '#6359A4'
                # sns.palplot([train_col,test_col,paramopt_col])

                # find nearest square for plotting
                import math

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
                            # print(ct_,':',[row_,col_])
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
                    from matplotlib.lines import Line2D
                    from matplotlib.patches import Patch

                    legend_elements = [
                        Patch(facecolor='#F7B531', edgecolor=None, label=('< ' + str(self.effco_) + '% : Efficient')),
                        # ('+str(len(self.df_[self.df_['class'] == 'efficient']))+' siRNAs)')),
                        Patch(facecolor='#3AA6E2', edgecolor=None, label=('≥ ' + str(
                            self.ineffco_) + '% : Inefficient')),
                        # ('+str(len(self.df_[self.df_['class'] == 'inefficient']))+' siRNAs)')),
                        Patch(facecolor='#B6B6B7', edgecolor=None, label=('Undefined')),
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
                        textprops=dict(color="black", fontsize=11),  # , fontweight='bold'),
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
                                 bbox_to_anchor=(0.2, 0.2), fontsize=11)

                    # Remove axes that are not needed (based on number of total parameter optmimization rounds)
                    ct_datasplit_rounds_to_plot_ = 1
                    for row_ in range(len(axs)):  # num rows
                        for col_ in range(len(axs[0]) - 1):  # num cols -1 (to skip last column)
                            # print(row_,col_,ct_datasplit_rounds_to_plot_)
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
                            x.replace('inefficient', '#3AA6E2').replace('efficient', '#F7B531').replace('undefined', '#B6B6B7') for
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
                        from matplotlib.lines import Line2D
                        from matplotlib.patches import Patch

                        legend_elements = [
                            Patch(facecolor='#F7B531', edgecolor=None, label=(str(len(self.df_[self.df_['class'] == 'efficient'])))),
                            # +' siRNAs')),
                            Patch(facecolor='#3AA6E2', edgecolor=None, label=(str(len(self.df_[self.df_['class'] == 'inefficient'])))),
                            # +' siRNAs')),
                            # Patch(facecolor='#B6B6B7', edgecolor=None, label=('Undefined ('+str(len(self.df_[self.df_['class'] == 'undefined']))+' siRNAs)')),
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
                    # print('Figure saved to:',fnm_svg_+'.svg')

                    fig.savefig(fnm_.split('.')[0] + '.png', format='png', dpi=300, transparent=False)
                    print('Figure saved to:', fnm_ + '.png')
            else:
                print("Not enough splits performed to plot as a grid \nwith 'num_rerurun_model_building' set to:",
                      self.num_rerurun_model_building)
        else:
            print("plot_grid_splits_ set to :" + str(self.plot_grid_splits_))

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

            train_col = '#D46F37'
            test_col = '#4BB3B1'
            paramopt_col = '#6359A4'
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

            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch

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
            print('Figure saved to:', fnm_ + '.png')
        else:
            print("plot_extra_visuals_ set to :" + str(self.plot_extra_visuals_))

    def plot_bar_data_splits(self):
        if self.plot_extra_visuals_:
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

                ineff_col = '#3AA6E2'
                eff_col = '#F7B531'
                undef_col = '#B6B6B7'
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

                ax_train_1.set_title('      ', rotation=90, fontweight='bold', fontsize=30, fontfamily='Times New Roman')
                ax_po_1.set_title('      ', rotation=90, fontweight='bold', fontsize=30, fontfamily='Times New Roman')
                ax_test_1.set_title('      ', rotation=90, fontweight='bold', fontsize=30, fontfamily='Times New Roman')

                ax_train_3.set_title('  ...', rotation=90, fontweight='bold', fontsize=25, fontfamily='Times New Roman')
                ax_po_3.set_title('  ...', rotation=90, fontweight='bold', fontsize=25, fontfamily='Times New Roman')
                ax_test_3.set_title('  ...', rotation=90, fontweight='bold', fontsize=25, fontfamily='Times New Roman')

                # ** SAVE FIGURE **
                plt.rcParams['svg.fonttype'] = 'none'  # exports text as strings rather than vector paths (images)
                output_dir__ = self.all_data_split_dir + 'figures/'
                fnm_ = (all_output_dir + output_dir__ + '_data_splitting_carton_for_slides')
                fnm_svg_ = fnm_

                fig.savefig(fnm_svg_.split('.')[0] + '.svg', format='svg', transparent=True)
                # print('Figure saved to:',fnm_svg_+'.svg')

                fig.savefig(fnm_.split('.')[0] + '.png', format='png', dpi=300, transparent=True)
                print('Figure saved to:', fnm_ + '.png')

            else:
                print("Not enough splits performed to construct a graphic \n\twith 'num_rerurun_model_building' set to:",
                      self.num_rerurun_model_building)

        else:
            print("plot_extra_visuals_ set to :" + str(self.plot_extra_visuals_))

        #########################################################################################################################################
        ###################################                    Run Model Fitting                     ############################################
        ###################################      (Parameter Optimization & Final Model Building)     ############################################
        #########################################################################################################################################



    def run_model_fittings(self):

        # Create a unique self.modeltrain_id_
        from random import randint
        # self.modeltrain_id_ = 'SUPk'+str(randint(10000, 99999) ) # ID used to find output data filie
        self.modeltrain_id_ = model_type_dict[self.model_type_] + param_id_dict[self.parameter_to_optimize].upper() + str(randint(10000, 99999))  # ID used to find output data file
        # be sure modeltrain_id_ doesn't exist already
        while self.modeltrain_id_ in [x[0:9] for x in os.listdir(all_output_dir)]:
            self.modeltrain_id_ = model_type_dict[self.model_type_] + param_id_dict[self.parameter_to_optimize].upper() + str(randint(10000, 99999))  # ID used to find output data file
            # print("self.modeltrain_id_ already exists, generating new ID...")
            # print("NEW self.modeltrain_id_ | Randomized " + str(len(self.modeltrain_id_)) + "-digit ID for this Set of Rounds:\t " + self.modeltrain_id_)
        print("self.modeltrain_id_ | Randomized " + str(len(self.modeltrain_id_)) + "-digit ID for this Set of Rounds:\t " + self.modeltrain_id_)
        self.all_output_dir_param_opt_round_ = 'popt-' + str(self.modeltrain_id_) + '_' + self.parameter_to_optimize + '_total-' + str(self.run_round_num) + '-rounds/'
        if not os.path.exists(all_output_dir + self.all_output_dir_param_opt_round_):
            os.makedirs(all_output_dir + self.all_output_dir_param_opt_round_)
            print("Output for all " + str(self.run_round_num) + " Parameter Optimization Rounds stored in:\n" + os.getcwd() + all_output_dir + self.all_output_dir_param_opt_round_)

        self.output_directory = 'output_' + model_type_dict[self.model_type_] + '_run_' + str(self.modeltrain_id_) + '_' + self.date_
        self.output_directory += '/'
        self.output_directory = all_output_dir + self.all_output_dir_param_opt_round_ + self.output_directory

        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
            print("Output data stored in:\n" + os.getcwd() + self.output_directory)
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
        else:
            raise Exception(
                'ERROR: folder with name "' + self.output_directory.replace(all_output_dir + self.all_output_dir_param_opt_round_,
                                                                            '') + '" already exists in ' + os.getcwd() + all_output_dir + self.all_output_dir_param_opt_round_ +
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
        # Perform Parameter Optimization first
        self.parameter_optimization()
        # Build Final Models
        self.build_final_models()

        return (self.paramop_performance_curves_encodings_dict, self.paramop_key_ls,
                self.final_performance_curves_encodings_dict, self.final_performance_metrics_encodings_dict, self.final_key_ls)




    def parameter_optimization(self):

        self.top_param_val_per_round_dict = {}
        self.paramop_performance_metrics_encodings_dict = {}
        self.paramop_performance_curves_encodings_dict = {}
        self.paramop_models_encodings_dict = {}
        self.paramop_key_ls = []
        for n_ in range(self.num_rerurun_model_building):
            print('Building Parameter Optimization ' + str(self.model_type_) + ' models for Round:', n_ + 1, '/',self.num_rerurun_model_building)

            train_data_fnm_ = all_output_dir + self.all_data_split_dir + 'datasets/' + 'ROUND-' + str(n_ + 1) + '_Training_Data_' + str(100 - (self.test_set_size_pcnt_ + self.paramopt_set_size_pcnt_)) + 'pcnt_partition.csv'
            paramop_data_fnm_ = all_output_dir + self.all_data_split_dir + 'datasets/' + 'ROUND-' + str(n_ + 1) + '_Parameter_Optimization_Data_' + str(self.paramopt_set_size_pcnt_) + 'pcnt_partition.csv'
            # Load in Training and Parameter Optimization Datasets
            print("Loading Training data...")
            print("1) all_output_dir:\n",all_output_dir,"\n")
            print("2) self.all_data_split_dir:\n",self.all_data_split_dir,"\n")
            print("3) train_data_fnm_:\n",train_data_fnm_,"\n")
            self.df_train = pd.read_csv(train_data_fnm_)
            print("Training data loaded successfully")
            print("Loading Parameter optimization data...")
            self.df_paramopt = pd.read_csv(paramop_data_fnm_)
            print("Parameter optimization data loaded successfully")

            ## Loop through Parameter Optimization
            # ['kmer-size', 'flank-length', 'model', 'window_size', 'word_frequency_cutoff', 'ANN_output_dimmension',
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

            param_ = 'X'
            for kmer_ in kmer_sizes_ls:
                for flank_seq_working_key__ in flank_seq_working_key__ls:
                    if flank_seq_working_key__ == '20mer_targeting_region':
                        flank_len__ = 0
                    else:
                        flank_len__ = flank_seq_working_key__.split('-')[-1].split('nts')[0]
                    for e in self.feature_encoding_ls: # ['one-hot', 'bow_countvect', 'bow_gensim', 'ann_keras', 'ann_word2vec_gensim']
                        for m_ in model_type_ls:
                            # Train Parameter Optimization Models
                            if self.parameter_to_optimize == 'kmer-size':
                                print('  ** Parameter Optimization -- '+str(self.parameter_to_optimize)+' :', kmer_, '**  ')
                                param_ = kmer_
                            if self.parameter_to_optimize == 'flank-length':
                                print('  ** Parameter Optimization -- ' + str(self.parameter_to_optimize) + ' :', flank_len__, '**  ')
                                param_ = flank_len__
                            if self.parameter_to_optimize == 'feature_encoding':
                                print('  ** Parameter Optimization -- ' + str(self.parameter_to_optimize) + ' :', e, '**  ')
                                param_ = e
                            if self.parameter_to_optimize == 'model':
                                print('  ** Parameter Optimization -- ' + str(self.parameter_to_optimize) + ' :', m_, '**  ')
                                param_ = m_


                            clf_po = model_dict[m_]

                            if e == 'one-hot' or e == 'bow_gensim':
                                X_train_ = [[float(y) for y in x.replace('[', '').replace(']', '').replace('\n', '').split(' ') if y != ''] for x in self.df_train[e + '_encoded_' + flank_seq_working_key__ + '_kmer-' + str(kmer_)]]
                            else:
                                X_train_ = [[float(y) for y in x.replace('[', '').replace(']', '').replace(' ', '').split(',')] for x in self.df_train[e + '_encoded_' + flank_seq_working_key__ + '_kmer-' + str(kmer_)]]
                            Y_train_ = list(self.df_train['numeric_class'])

                            if 'semi-sup' in self.model_type_:
                                # Add undefined data
                                X_train_u_ = [list(x) for x in list(self.df.loc[(self.indxs_labeled_data[-1]+1):][e + '_encoded_' + flank_seq_working_key__ + '_kmer-' + str(kmer_)])]
                                Y_train_u_ = list(self.df.loc[(self.indxs_labeled_data[-1]+1):]['numeric_class'])
                                X_train_ = X_train_u_ + X_train_
                                Y_train_ = np.array(Y_train_u_ + Y_train_)

                            if e == 'one-hot' or e == 'bow_gensim':
                                X_paramopt_ = [[float(y) for y in x.replace('[', '').replace(']', '').replace('\n', '').split(' ') if y != ''] for x in self.df_paramopt[e + '_encoded_' + flank_seq_working_key__ + '_kmer-' + str(kmer_)]]
                            else:
                                X_paramopt_ = [[float(y) for y in x.replace('[', '').replace(']', '').replace(' ', '').split(',')] for x in self.df_paramopt[e + '_encoded_' + flank_seq_working_key__ + '_kmer-' + str(kmer_)]]
                            
                            Y_paramopt_ = np.array(self.df_paramopt['numeric_class'])
                            print("Training paramopt model...")
                            clf_po.fit(X_train_, Y_train_)
                            print("Predicting with paramopt model...")
                            preds_po = clf_po.predict_proba(X_paramopt_)[:, 1]
                            preds_binary_po = clf_po.predict(X_paramopt_)

                            ## Evaluate Parameter Optimization Model Performance
                            print("Evaluating model performance of paramopt model...")
                            from sklearn.metrics import precision_recall_curve
                            print('\n\n\npreds_po :',set(preds_po),'\n\n\n')
                            p_po_, r_po_, ts_po_ = precision_recall_curve(Y_paramopt_, preds_po)
                            aucpr_po_ = metrics.auc(r_po_, p_po_)

                            from sklearn.metrics import f1_score
                            fscore_po_ = f1_score(Y_paramopt_, preds_binary_po)  # , average=None)

                            from sklearn.metrics import fbeta_score
                            fbetascore_po_ = fbeta_score(Y_paramopt_, preds_binary_po, beta=self.f_beta_)  # , average=None)
                            print("Computing fbeta_score with beta =",self.f_beta_)

                            from sklearn.metrics import accuracy_score
                            accuracy_po_ = accuracy_score(Y_paramopt_, preds_binary_po)

                            from sklearn.metrics import matthews_corrcoef
                            mcc_po_ = matthews_corrcoef(Y_paramopt_, preds_binary_po)

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
                                'Unacheivabe_Region_Curve': [unach_precs_po, unach_recalls_po],
                            }

                            paramop_performance_metrics_dict_ = {
                                'AUCPR': aucpr_po_,
                                'AUCPR-adj': aucpr_po_ - p_po_[0],
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
            print(
                "Identifying Best Parameter Value in Parameter Optimization:" + '\n\t Metric: ' + metric_used_to_id_best_po_ + '\n\t Score: ' +
                str(max_score_po_) + '\n\t Parameter ID: ' + best_po_id_ + '\n\t ' + str(self.parameter_to_optimize) + ': ' + str(
                    param_val_))

            self.output_run_file_info_string_.replace('PARAMOPT', str(param_val_))

            print('\n\n\n\n\n' +
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




    def build_final_models(self):
        self.final_models_encodings_dict = {}
        self.final_performance_metrics_encodings_dict = {}
        self.final_detailed_performance_metrics_encodings_dict = {}
        self.final_performance_curves_encodings_dict = {}
        self.final_key_ls = []
        self.final_model_params_ls = []


        # TODO: flank-seq-working key might not work for cases where don't have targeting region or just have target region alone
        for n_ in range(self.num_rerurun_model_building):
            param_val_ = self.top_param_val_per_round_dict[n_]
            if self.parameter_to_optimize == 'kmer-size':
                kmer_size___ = param_val_
            else:
                kmer_size___ = self.kmer_size_

            if self.parameter_to_optimize == 'flank-length':
                if param_val_ == 0:
                    flank_seq_working_key___ = '20mer_targeting_region'
                else:
                    flank_seq_working_key___ = 'seq_flank-'+str(param_val_)+'nts_target'
            else:
                flank_seq_working_key___ = self.flank_seq_working_key

            if self.parameter_to_optimize == 'model':
                model_type___ = param_val_
            else:
                model_type___ = self.model_type_


            print('Building Final ' + str(model_type___) + ' models for Round:', n_ + 1, '/', self.num_rerurun_model_building)
            train_data_fnm_ = all_output_dir + self.all_data_split_dir + 'datasets/' + 'ROUND-' + str(n_ + 1) + '_Training_Data_' + str(100 - (self.test_set_size_pcnt_ + self.paramopt_set_size_pcnt_)) + 'pcnt_partition.csv'
            test_data_fnm_ = all_output_dir + self.all_data_split_dir + 'datasets/' + 'ROUND-' + str(n_ + 1) + '_Testing_Data_' + str(self.test_set_size_pcnt_) + 'pcnt_partition.csv'

            # Load in Training and Testing Datasets
            self.df_train = pd.read_csv(train_data_fnm_)
            self.df_test = pd.read_csv(test_data_fnm_)
            for e in self.feature_encoding_ls:
                # Train Final  Models
                clf_final = model_dict[model_type___]

                if e == 'one-hot' or e == 'bow_gensim':
                    X_train_ = [[float(y) for y in x.replace('[', '').replace(']', '').replace('\n', '').split(' ') if y != ''] for x in self.df_train[e + '_encoded_' + flank_seq_working_key___ + '_kmer-' + str(kmer_size___ )]]
                else:
                    X_train_ = [[float(y) for y in x.replace('[', '').replace(']', '').replace(' ', '').split(',')] for x in self.df_train[e + '_encoded_' + flank_seq_working_key___ + '_kmer-' + str(kmer_size___ )]]
                Y_train_ = list(self.df_train['numeric_class'])

                if 'semi-sup' in model_type___:
                    # Add undefined data
                    X_train_u_ = [list(x) for x in list(self.df.loc[(self.indxs_labeled_data[-1]+1):][e + '_encoded_' + flank_seq_working_key___ + '_kmer-' + str(kmer_size___ )])]
                    Y_train_u_ = list(self.df.loc[(self.indxs_labeled_data[-1]+1):]['numeric_class'])
                    X_train_ = X_train_u_ + X_train_
                    Y_train_ = np.array(Y_train_u_ + Y_train_)

                if e == 'one-hot' or e == 'bow_gensim':
                    X_test_ = [[float(y) for y in x.replace('[', '').replace(']', '').replace('\n', '').split(' ') if y != ''] for x in self.df_test[e + '_encoded_' + flank_seq_working_key___ + '_kmer-' + str(kmer_size___ )]]
                else:
                    X_test_ = [[float(y) for y in x.replace('[', '').replace(']', '').replace(' ', '').split(',')] for x in self.df_test[e + '_encoded_' + flank_seq_working_key___ + '_kmer-' + str(kmer_size___ )]]

                Y_test_ = np.array(self.df_test['numeric_class'])
                clf_final.fit(X_train_, Y_train_)

                preds_final = clf_final.predict_proba(X_test_)[:, 1]
                preds_binary_final = clf_final.predict(X_test_)

                ## Evaluate Parameter Optimization Model Performance
                from sklearn.metrics import precision_recall_curve

                p_final_, r_final_, ts_final_ = precision_recall_curve(Y_test_, preds_final)
                aucpr_final_ = metrics.auc(r_final_, p_final_)
                from sklearn.metrics import f1_score

                fscore_final_ = f1_score(Y_test_, preds_binary_final)  # , average=None)
                from sklearn.metrics import accuracy_score

                from sklearn.metrics import fbeta_score
                fbetascore_final_ = fbeta_score(Y_test_, preds_binary_final, beta=self.f_beta_)  # , average=None)
                print("Computing Final fbeta_score with beta =", self.f_beta_)

                accuracy_final_ = accuracy_score(Y_test_, preds_binary_final)
                from sklearn.metrics import matthews_corrcoef

                mcc_final_ = matthews_corrcoef(Y_test_, preds_binary_final)

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
                    'Unacheivabe_Region_Curve': [unach_precs_final, unach_recalls_final],
                }

                final_performance_metrics_dict_ = {
                    'AUCPR': aucpr_final_,
                    'AUCPR-adj': aucpr_final_ - p_final_[0],
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





    def plot_param_opt_precision_recall_curves(self):
        ## Plot compiled Parameter Optimization Precision-Recall curves as a single figure
        sup_title_id_info = ('' +  # str(num_rerurun_model_building*run_round_num)+' rounds'+'\n'+
                             self.output_run_file_info_string_.replace('_', ' ').replace(self.region_.replace('_', '-'), self.region_.replace('_', '-') + '\n'))

        pr_curves_dict_ = self.paramop_performance_curves_encodings_dict
        pr_curves_keys_ = self.paramop_key_ls
        title_st_ = sup_title_id_info + 'Parameter Optimization ' + str(self.parameter_to_optimize) + '\n' + str(self.num_rerurun_model_building) + ' Rounds ' + str(len(self.param_values_to_loop_)) + ' Parameter Values'

        ## Find nearest square for plotting grid
        import math
        rows_cols_compiled_po_fig = math.ceil(math.sqrt(self.num_rerurun_model_building))

        param_sizes_to_loop_ = self.param_values_to_loop_

        import seaborn as sns
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
                    # print(row_,col_,ct_po_rounds_to_plot_)
                    if ct_po_rounds_to_plot_ > self.num_rerurun_model_building:#len(p_r_t_po__ls_):
                        # remove axis from figure
                        axs[row_][col_].remove()
                    ct_po_rounds_to_plot_ += 1
            # get best parameter value per round
            best_aucprs_dict = {}
            for n in range(self.num_rerurun_model_building): # loop through rounds
                best_round_aucpr_ = -9999
                best_round_param_ = ''
                for p in self.param_values_to_loop_:  # different colors
                    k__ =str(e) + '-' + self.parameter_to_optimize + '-' + str(p) + '_round_' + str(n)
                    if k__ in pr_curves_keys_:
                        prt_ls = self.paramop_performance_curves_encodings_dict[k__]['Precision_Recall_Curve']
                        aucpr_ = metrics.auc(prt_ls[1], prt_ls[0])
                        if aucpr_ > best_round_aucpr_:
                            best_round_aucpr_ = aucpr_
                            best_round_param_ = p
                best_aucprs_dict[n] = [best_round_param_,best_round_aucpr_,]

            col_ = 0
            row_ = 0
            for n in range(self.num_rerurun_model_building):  # different axes per round
                axs[row_, col_].set_title('Round: ' + str(n + 1) + '\nBest ' + self.parameter_to_optimize + str(best_aucprs_dict[n][0]),
                                          fontsize=8, color=param_col_ls_dict[best_aucprs_dict[n][0]])

                for p in self.param_values_to_loop_: # different colors
                    k__ =str(e) + '-' + self.parameter_to_optimize + '-' + str(p) + '_' + str(n)
                    if k__ in pr_curves_keys_:
                        prt_ls = self.paramop_performance_curves_encodings_dict[k__]['Precision_Recall_Curve']
                        axs[row_,col_].plot(
                            prt_ls[1],#r_OH,# x
                            prt_ls[0],#p_OH,# y
                            lw=1,
                            color= param_col_ls_dict[p],
                            #color = embd_color_dict[embd_][key__],
                        )

                        aucpr_ = metrics.auc(prt_ls[1], prt_ls[0])

                        #print(row_,col_)
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
                axs[row_,col_].set_xticks(ticks=np.arange(0,1.1,.25), minor=True)
                axs[row_,col_].tick_params(direction='in',which='both',length=3,width=1)

                if (row_ == rows_cols_compiled_po_fig-1) and (col_ == 0):
                    axs[row_,col_].set_xlabel('Recall')
                    axs[row_,col_].set_ylabel('Precision')
                    axs[row_,col_].set_xticks(ticks=np.arange(0,1.1,.5),labels=['',0.5,1.0])

                else:
                    axs[row_,col_].set_xticklabels([])
                    axs[row_,col_].set_yticklabels([])
                if col_ == rows_cols_compiled_po_fig-1:
                    col_ = 0
                    row_+=1
                else:
                    col_+=1

            # Add legend for parameter values
            from matplotlib.lines import Line2D
            legend_elements = []
            for val__,i in zip(param_sizes_to_loop_,range(len(param_sizes_to_loop_))):
                legend_elements.append(Line2D([0], [0],
                                              color= param_col_ls_dict[val__],#embd_color_dict[embd_][val__],
                                              lw=4, label=str(val__)))
            axs[0][rows_cols_compiled_po_fig].legend(handles=legend_elements, loc='upper left', frameon=False,bbox_to_anchor = (0,1),title=self.parameter_to_optimize,title_fontsize=10,fontsize=10)
            axs[0][rows_cols_compiled_po_fig].axis('off')
            axbig.legend(handles=legend_elements, loc='upper left', frameon=False,bbox_to_anchor = (0,1),title=self.parameter_to_optimize,title_fontsize=10,fontsize=10)
            axbig.axis('off')

            plt.suptitle(e.replace('_',' ').capitalize(), fontsize = 12, fontweight = 'bold')
            fig.tight_layout() # NOTE: h and w (above in fig.set_size... MUST be large enough to accomodate legends or will be cut off/squished in output)
            # ** SAVE FIGURE **
            plt.rcParams['svg.fonttype'] = 'none' # exports text as strings rather than vector paths (images)
            fnm_ =     (self.output_directory+ 'figures/'+           'p-r_'+str(self.num_rerurun_model_building)+'-rnds_comp_po_'+e)
            fnm_svg_ = (self.output_directory+'figures/'+'svg_figs/'+'p-r_'+str(self.num_rerurun_model_building)+'-rnds_comp_po_'+e)
            fig.savefig(fnm_svg_.split('.')[0]+'.svg',format='svg',transparent=True)
            fig.savefig(fnm_.split('.')[0]+'.png',format='png',dpi=300,transparent=False)



    def plot_param_opt_model_box_plots(self):
        ## Plot Compiled Multimetrics Model Performance - Parameter Optimization Models
        # Each column of paramop_detailed_metric_df contains a single round for a single embedding type
        paramop_detailed_metric_df = pd.DataFrame(self.paramop_performance_metrics_encodings_dict)

        flierprops__ = dict(marker='.', markerfacecolor='none', markersize=4, linewidth=0.1, markeredgecolor='black')  # linestyle='none',
        boxprops__ = dict(facecolor='none', linestyle='none', linewidth=1, edgecolor='k', )
        medianprops__1 = dict(linewidth=2, color='goldenrod')
        medianprops__2 = dict(linewidth=2, color='#2c8799')
        medianprops__3 = dict(linewidth=2, color='firebrick')

        # Loop for each encoding and plot as a grid with different encoding per line

        # one row per embedding
        # one axis per performance metric
        # one box per embedding per axis
        fig, axs = plt.subplots(len(self.feature_encoding_ls), len(paramop_detailed_metric_df))
        fig.set_size_inches(w=12, h=3*len(self.feature_encoding_ls))

        # Split Evaluation Metric Data per parameter value
        # Loop through each embedding type
        for embedding_type_paramop_eval_, j in zip(self.feature_encoding_ls, list(range(len(self.feature_encoding_ls)))):
            # From paramop_detailed_metric_df get just columns for a single selected embedding type
            # Get just columns with selected embedding metric (embedding_type_paramop_eval_)
            cols_with_embd_ = [x for x in list(paramop_detailed_metric_df.columns) if embedding_type_paramop_eval_ in x]
            print("Columns from paramop_detailed_metric_df with embedding = 'embedding_type_paramop_eval_' = " +
                  str(embedding_type_paramop_eval_) + " - ", len(cols_with_embd_), 'out of', len(list(paramop_detailed_metric_df.columns)))

            # Get just columns with selected embedding metric (embedding_type_paramop_eval_)
            paramop_detailed_metric_one_embd_df = paramop_detailed_metric_df[cols_with_embd_]

            # Get parameter VALUES only for each selected embedding
            param_vals_one_embd_ = list(set([x.split(str(self.parameter_to_optimize) + '-')[-1].split('_round_')[0] for x in list(paramop_detailed_metric_one_embd_df.columns)]))

            metrics_ls = list(paramop_detailed_metric_one_embd_df.index)



            for i in range(len(metrics_ls)):
                metric_ = metrics_ls[i]
                # Plot by parameter value
                data_ = [list(paramop_detailed_metric_one_embd_df[
                                  [embedding_type_paramop_eval_ + '-' + str(self.parameter_to_optimize) + '-' + param_val_ + '_round_' + str(i) for i in
                                   list(range(self.num_rerurun_model_building))]].transpose()[metric_]) for param_val_ in param_vals_one_embd_]

                bplot1 = axs[j,i].boxplot(
                    data_,
                    vert=True,  # vertical box alignment
                    patch_artist=True,  # fill with color
                    labels=param_vals_one_embd_,
                    flierprops=flierprops__, boxprops=boxprops__,
                    capprops=dict(color='black'),
                    whiskerprops=dict(color='black'),

                )  # will be used to label x-ticks
                axs[j,i].set_title(metric_)
                if i == 2:
                    if embedding_type_paramop_eval_ == self.feature_encoding_ls[0]:# for first row of plots in figure
                        axs[j,i].set_title('Parameter Optimization Model Performances (' + str(self.num_rerurun_model_building) + ' Rounds)\n'+str(embedding_type_paramop_eval_)+'\n' + str(metric_))  # ,fontweight='bold')
                    else:
                        axs[j,i].set_title(str(embedding_type_paramop_eval_)+'\n' + str(metric_))  # ,fontweight='bold')

                # update x-axis labels
                axs[j,i].set_xticklabels(param_vals_one_embd_, rotation=0)
                axs[j,i].set_xlabel(str(self.parameter_to_optimize))
                if metric_ == 'MCC':
                    axs[j,i].set_ylim(-1, 1)
                else:
                    axs[j,i].set_ylim(0,1)
        fig.suptitle('Compiled Multiple Metrics Parameter Optimization Models - Per Parameter Value ' + str(self.num_rerurun_model_building) +
                     ' rounds' + '\n' + self.output_run_file_info_string_.replace('_', ' ').replace(self.region_.replace('_', '-'), self.region_.replace('_', '-') + '\n'), fontsize=9)
        fig.tight_layout()

        # ** SAVE FIGURE **
        plt.rcParams['svg.fonttype'] = 'none'  # exports text as strings rather than vector paths (images)
        fnm_ = (self.output_directory + 'figures/' + 'bxp_multimetric_per-param-val_' + str(self.num_rerurun_model_building) + '-rnds_paramop')
        fnm_svg_ = (self.output_directory + 'figures/' + 'svg_figs/' + 'bxp_multimetric_per-param-val_' + str(self.num_rerurun_model_building) + '-rnds_paramop')
        fig.savefig(fnm_svg_.split('.')[0] + '.svg', format='svg', transparent=True)
        fig.savefig(fnm_.split('.')[0] + '.png', format='png', dpi=300, transparent=False)
        print('Figure saved to:', fnm_ + '.png'.replace(self.output_directory, '~/'))

        return





    def plot_final_model_precision_recall_curves(self):
        ## TODO: Plot Final Model Precision-Recall curves as a single figure
        sup_title_id_info = ('' +  # str(num_rerurun_model_building*run_round_num)+' rounds'+'\n'+
                             self.output_run_file_info_string_.replace('_', ' ').replace(self.region_.replace('_', '-'), self.region_.replace('_', '-') + '\n'))

        pr_curves_dict_ = self.final_performance_curves_encodings_dict
        pr_curves_keys_ = self.final_key_ls

        import seaborn as sns
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
        fig, axs = plt.subplots(self.feature_encoding_ls + 1)
        fig.set_size_inches(w=9, h=3, )  # NOTE: h and w must be large enough to accomodate any legends

        for e, col_ in zip(self.feature_encoding_ls,list(range(len(self.feature_encoding_ls)))):  # different plots per embedding
            axs[col_].set_title(str(e))
            for n in range(self.num_rerurun_model_building):  # loop through rounds

                # Get p-r curves per round for given embedding and round
                key__ = str(e) + '_' + str(n)
                pcurve__ = self.final_performance_curves_encodings_dict[key__]['Precision_Recall_Curve'][0]
                rcurve__ = self.final_performance_curves_encodings_dict[key__]['Precision_Recall_Curve'][1]
                unach_pcurve__ = self.final_performance_curves_encodings_dict[key__]['Unacheivabe_Region_Curve'][0]
                unach_rcurve__ = self.final_performance_curves_encodings_dict[key__]['Unacheivabe_Region_Curve'][1]
                p = self.final_model_params_ls[n]

                axs[col_].plot(
                    rcurve__,  # r_OH,# x
                    pcurve__,  # p_OH,# y
                    lw=1,
                    color=param_col_ls_dict[p],
                )

                axs[col_].plot(
                    rcurve__,  # r_OH,# x
                    unach_rcurve__,  # p_OH,# y
                    lw=1,
                    color=param_col_ls_dict[p],
                    linestyle='dashed',
                )

            # Format Axes
            axs[col_].set_xlim(0, 1)
            axs[col_].set_ylim(0, 1.1)
            axs[col_].set_xticks(ticks=np.arange(0, 1.1, .25), minor=True)
            axs[col_].tick_params(direction='in', which='both', length=3, width=1)

            if col_ == 0:
                axs[col_].set_xlabel('Recall')
                axs[col_].set_ylabel('Precision')
                axs[col_].set_xticks(ticks=np.arange(0, 1.1, .5), labels=['', 0.5, 1.0])

            else:
                axs[col_].set_xticklabels([])
                axs[col_].set_yticklabels([])

        # Add legend for parameter values
        from matplotlib.lines import Line2D
        legend_elements = []
        for val__, i in zip(list(set(self.final_model_params_ls)), range(len(list(set(self.final_model_params_ls))))):
        #for val__, i in zip(self.param_values_to_loop_, range(len(self.param_values_to_loop_))):
            legend_elements.append(Line2D([0], [0],
                                          color=param_col_ls_dict[val__],  # embd_color_dict[embd_][val__],
                                          lw=4, label=str(val__)))
        axs[-1].legend(handles=legend_elements, loc='upper left', frameon=False, bbox_to_anchor=(0, 1), title=self.parameter_to_optimize, title_fontsize=10, fontsize=10)
        axs[-1].axis('off')

        fig.suptitle('Compiled Precision-Recall Curves Final Models - Per Embedding ' + str(self.num_rerurun_model_building) +
                     ' rounds' + '\n' + self.output_run_file_info_string_.replace('_', ' ').replace(self.region_.replace('_', '-'), self.region_.replace('_', '-') + '\n'), fontsize=9)
        fig.tight_layout()  # NOTE: h and w (above in fig.set_size... MUST be large enough to accomodate legends or will be cut off/squished in output)
        # ** SAVE FIGURE **
        plt.rcParams['svg.fonttype'] = 'none'  # exports text as strings rather than vector paths (images)
        fnm_ = (self.output_directory + 'figures/' + 'p-r_' + str(self.num_rerurun_model_building) + '-rnds_final')
        fnm_svg_ = (self.output_directory + 'figures/' + 'svg_figs/' + 'p-r_' + str(self.num_rerurun_model_building) + '-rnds_final')
        fig.savefig(fnm_svg_.split('.')[0] + '.svg', format='svg', transparent=True)
        fig.savefig(fnm_.split('.')[0] + '.png', format='png', dpi=300, transparent=False)
        return



    def plot_final_model_box_plots_per_param_val(self):
        ## Plot Compiled Multimetrics Model Performance per Param Val - Final Models

        # Each column of final_detailed_metric_df contains a single round for a single embedding type
        final_detailed_metric_df = pd.DataFrame(self.final_detailed_performance_metrics_encodings_dict)

        flierprops__ = dict(marker='.', markerfacecolor='none', markersize=4, linewidth=0.1, markeredgecolor='black')  # linestyle='none',
        boxprops__ = dict(facecolor='none', linestyle='none', linewidth=1, edgecolor='k', )
        medianprops__1 = dict(linewidth=2, color='goldenrod')
        medianprops__2 = dict(linewidth=2, color='#2c8799')
        medianprops__3 = dict(linewidth=2, color='firebrick')

        # one axis per performance metric
        # one box per embedding per axis
        fig, axs = plt.subplots( len(self.feature_encoding_ls), len(final_detailed_metric_df))
        fig.set_size_inches(w=12, h=3*len(self.feature_encoding_ls))


        # Split Evaluation Metric Data per parameter value
        # Loop through all embeddings used
        for embedding_type_final_eval_, j in zip(self.feature_encoding_ls,list(range(len(self.feature_encoding_ls)))):
            # From final_detailed_metric_df get just columns for a single selected embedding type
            # Get just columns with selected embedding metric (embedding_type_final_eval_)
            cols_with_embd_ = [x for x in list(final_detailed_metric_df.columns) if embedding_type_final_eval_ in x]
            print("Columns from final_detailed_metric_df with embedding = 'embedding_type_final_eval_' = "+
                  str(embedding_type_final_eval_)+" - ",len(cols_with_embd_),'out of',len(list(final_detailed_metric_df.columns)))

            # Get just columns with selected embedding metric (embedding_type_final_eval_)
            final_detailed_metric_one_embd_df = final_detailed_metric_df[cols_with_embd_]

            # Get parameter VALUES only for each selected embedding
            param_vals_one_embd_ = list(set([x.split(str(self.parameter_to_optimize)+'-')[-1].split('_round_')[0] for x in list(final_detailed_metric_one_embd_df.columns)]))

            metrics_ls = list(final_detailed_metric_one_embd_df.index)

            for i in range(len(metrics_ls)):
                metric_ = metrics_ls[i]
                # Plot by parameter value
                data_ = [list(final_detailed_metric_one_embd_df[
                                  [embedding_type_final_eval_+'-'+str(self.parameter_to_optimize)+'-'+param_val_ + '_round_' + str(i) for i in
                                   list(range(self.num_rerurun_model_building))]].transpose()[metric_]) for param_val_ in param_vals_one_embd_]

                bplot1 = axs[j,i].boxplot(
                    data_,
                    vert=True,  # vertical box alignment
                    patch_artist=True,  # fill with color
                    labels=param_vals_one_embd_,
                    flierprops=flierprops__, boxprops=boxprops__,
                    capprops=dict(color='black'),
                    whiskerprops=dict(color='black'),

                )  # will be used to label x-ticks
                axs[j,i].set_title(metric_)
                if i == 2:
                    axs[j,i].set_title('Final Model Performances (' + str(self.num_rerurun_model_building) + ' Rounds)\n' + str(metric_))  # ,fontweight='bold')

                # update x-axis labels
                axs[j,i].set_xticklabels(param_vals_one_embd_, rotation=0)
                axs[j,i].set_xlabel(str(self.parameter_to_optimize))

                if metric_ == 'MCC':
                    axs[j,i].set_ylim(-1, 1)
                else:
                    axs[j,i].set_ylim(0, 1)

        fig.suptitle('Compiled Multiple Metrics Final Models - Per Parameter Value ' + str(self.num_rerurun_model_building) +
                     ' rounds' + '\n' + self.output_run_file_info_string_.replace('_', ' ').replace(self.region_.replace('_', '-'), self.region_.replace('_', '-') + '\n'), fontsize=9)
        fig.tight_layout()

        # ** SAVE FIGURE **
        plt.rcParams['svg.fonttype'] = 'none'  # exports text as strings rather than vector paths (images)
        fnm_ = (self.output_directory + 'figures/' + 'bxp_multimetric_per-param-val_' + str(self.num_rerurun_model_building) + '-rnds_final')
        fnm_svg_ = (self.output_directory + 'figures/' + 'svg_figs/' + 'bxp_multimetric_per-param-val_' + str(self.num_rerurun_model_building) + '-rnds_final')
        fig.savefig(fnm_svg_.split('.')[0] + '.svg', format='svg', transparent=True)
        fig.savefig(fnm_.split('.')[0] + '.png', format='png', dpi=300, transparent=False)
        print('Figure saved to:', fnm_ + '.png'.replace(self.output_directory, '~/'))
        return


    def plot_final_model_box_plots(self):
        ## Plot Compiled Multimetrics Model Performance - Final Models

        final_metric_df = pd.DataFrame(self.final_performance_metrics_encodings_dict)

        metrics_ls = list(final_metric_df.index)
        enc_ls_ = self.feature_encoding_ls

        flierprops__ = dict(marker='.', markerfacecolor='none', markersize=4, linewidth=0.1, markeredgecolor='black')  # linestyle='none',
        boxprops__ = dict(facecolor='none', linestyle='none', linewidth=1, edgecolor='k', )
        medianprops__1 = dict(linewidth=2, color='goldenrod')
        medianprops__2 = dict(linewidth=2, color='#2c8799')
        medianprops__3 = dict(linewidth=2, color='firebrick')

        # one axis per performance metric
        # one box per embedding per axis
        fig, axs = plt.subplots(1, len(final_metric_df))
        fig.set_size_inches(w=12, h=3)

        for i in range(len(metrics_ls)):
            metric_ = metrics_ls[i]

            data_ = [list(final_metric_df[[enc_+'_' + str(i) for i in [0, 1]]].transpose()[metric_]) for enc_ in
                     enc_ls_]
            bplot1 = axs[i].boxplot(
                data_,
                vert=True,  # vertical box alignment
                patch_artist=True,  # fill with color
                labels=enc_ls_,
                flierprops=flierprops__, boxprops=boxprops__,
                capprops=dict(color='black'),
                whiskerprops=dict(color='black'),

            )  # will be used to label x-ticks
            axs[i].set_title(metric_)
            if i ==2:
                axs[i].set_title('Final Model Performances (' + str(self.num_rerurun_model_building) + ' Rounds)\n' + str(metric_))  # ,fontweight='bold')

            # update x-axis labels
            axs[i].set_xticklabels([feature_encodings_dict[x] for x in self.feature_encoding_ls], rotation=90)

            if metric_ == 'MCC':
                axs[i].set_ylim(-1, 1)
            else:
                axs[i].set_ylim(0, 1)

        fig.suptitle('Compiled Multiple Metrics Final Models '+str(self.num_rerurun_model_building)+
                     ' rounds' +'\n'+self.output_run_file_info_string_.replace('_',' ').replace(self.region_.replace('_','-'),self.region_.replace('_','-')+'\n'),fontsize=9)
        fig.tight_layout()


        # ** SAVE FIGURE **
        plt.rcParams['svg.fonttype'] = 'none' # exports text as strings rather than vector paths (images)
        fnm_ =     (self.output_directory+ 'figures/'+           'bxp_multimetric_'+str(self.num_rerurun_model_building)+'-rnds_final')
        fnm_svg_ = (self.output_directory+'figures/'+'svg_figs/'+'bxp_multimetric_'+str(self.num_rerurun_model_building)+'-rnds_final')
        fig.savefig(fnm_svg_.split('.')[0]+'.svg',format='svg',transparent=True)
        fig.savefig(fnm_.split('.')[0]+'.png',format='png',dpi=300,transparent=False)
        print('Figure saved to:',fnm_+'.png'.replace(self.output_directory,'~/'))








#print("Running directly from main.py...")
#drb = DataRepresentationBuilder(parameter_to_optimize__ = 'kmer-size')
##drb = DataRepresentationBuilder(parameter_to_optimize__ = 'flank-length',model_type__ = 'semi-sup-random-forest')


#drb.create_processed_datasets()

#pr_po,kf,pr_f,m_f,k_f = drb.run_model_fittings()

#drb.plot_param_opt_precision_recall_curves()
#drb.plot_final_model_box_plots()


# TODO: pickle final models
# TODO: export metrics/scores to a file?
# TODO: plot final model p-r curves
# TODO: pickle final models
# TODO: For Semi-supervised: Add back in the unlabelled data to the testing set?
# TODO: add back in undefined middle values and evaluate model (possibly using needle-in-haystack method)

#print("Process Complete!")

#print(m_f)

# pr_po,k_po,pr_f,m_f,k_f


