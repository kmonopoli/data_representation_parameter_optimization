#
# # TODO: move relevant model evaluation (for parameter optimization, final model building, and external data evaluation) plotting methods here
#
#
# import pandas as pd
# import numpy as np
# from matplotlib import pyplot as plt
# from matplotlib import rcParams
# import matplotlib.pylab as pylab
# import seaborn as sns
# from matplotlib.lines import Line2D
# from matplotlib.patches import Patch
#
#
# params = {'legend.fontsize': 12,
#           'figure.figsize': (6, 4),
#           'axes.labelsize': 12,
#           'axes.titlesize': 12,
#           'xtick.labelsize': 12,
#           'ytick.labelsize': 12,
#           'font.family': 'Arial', # TODO: comment out if running on the cluster!
#           }
# pylab.rcParams.update(params)
#
#
#
#
#
# class DataRepresentationModelPerformance(DataRepresentationBuilder):
#     def __init__(self, parameter_to_optimize__,
#                  num_rerurun_model_building__=10,  # 25,
#                  custom_parameter_values_to_loop__=[],
#                  model_type__='random-forest',
#                  flank_len__=50, kmer_size__=9,
#                  normalized__=True, effco__=25, ineffco__=60, remove_undefined__=True,
#                  region__='flanking_and_target_regions',
#                  screen_type_ls__=['bDNA'], species_ls__=['human'],
#                  chemical_scaffold_ls__=['P3'],
#                  window_size__=1, word_freq_cutoff__=1, output_dimmension__=20,
#                  unlabelled_data__='sequences-from-all-transcripts',
#                  unlabeled_data_size__=1.00,
#                  plot_grid_splits__=False, plot_extra_visuals__=False,
#                  run_round_num__=1,
#                  encoding_ls__=['one-hot', 'ann-word2vec-gensim', 'bow-gensim', 'ann-keras', 'bow-countvect'],
#                  metric_used_to_id_best_po__='F-Score',
#                  f_beta__=0.5,
#                  run_param_optimization__=True,
#                  use_existing_processed_dataset__=False,
#                  apply_final_models_to_external_dataset__=False,  # whether or not to use external_data_file to evaluate final models
#                  ext_species_ls__=['human'], ext_chemical_scaffold_ls__=['P3'],
#                  # randomize_ext_data__ = False, # Not used if using external_data_file__ parameter
#                  external_data_file__='newly_added_sirna_screen_data_777-sirnas|-bdna_FEB-22-2024.csv',  # NO randomization of extenral data
#                  # external_data_file__ = 'randomized_sirna_screen_data_777-sirnas_p3-bdna_FEB-28-2024.csv',  # Randomization of extenral data
#                  # external_data_file__  = 'cleaned_no-bad-or-duplicate-screens_sirna_screen_data_4392sirnas-bdna-75-genes_JAN-29-2024.csv',  # NO randomization of extenral data
#                  input_data_dir__='new_input_data/',
#                  input_data_file__='training_sirna_screen_data_bdna-human-p3_1903-sirnas_MAR-21-2024.csv',
#                  ):
#                 super().__init__(num_rerurun_model_building__, custom_parameter_values_to_loop__, model_type__, flank_len__, kmer_size__, normalized__, effco__, ineffco__, remove_undefined__, region__, screen_type_ls__, species_ls__, chemical_scaffold_ls__, window_size__, output_dimmension__, unlabelled_data__, unlabeled_data_size__, plot_grid_splits__, run_round_num__, encoding_ls__, metric_used_to_id_best_po__, f_beta__, run_param_optimization__, use_existing_processed_dataset__, apply_final_models_to_external_dataset__, ext_species_ls__, external_data_file__, input_data_dir__, input_data_file__)
#
#     def plot_final_model_precision_recall_curves_on_ext_dataset_and_test_set(self):
#         print("\nPlotting precision-recall curves for final models on test set AND evaluated on external dataset...")
#         if not self.apply_final_models_to_external_dataset_:
#             print("apply_final_models_to_external_dataset_ is set to False so did not evaluate on an external dataset")
#             return
#
#         ## Plot Final Model Precision-Recall curves as a single figure
#         sup_title_id_info = ('' +  # str(num_rerurun_model_building*run_round_num)+' rounds'+'\n'+
#                              self.output_run_file_info_string_.replace('_', ' ').replace(self.region_.replace('_', '-'), self.region_.replace('_', '-') + '\n'))
#
#         pr_curves_dict = self.final_performance_curves_encodings_dict
#         pr_curves_dict_ext = self.ext_final_performance_curves_encodings_dict
#         pr_curves_keys_ = self.final_key_ls
#
#
#         # # Create dictionary of parameter colors based on parameter
#         # param_col_ls = list(sns.color_palette("hls", len(self.param_values_to_loop_)).as_hex())
#         # greys_col_ls = list(sns.color_palette("Greys", len(self.param_values_to_loop_)).as_hex())
#         #
#         # param_col_ls_dict = {}
#         # for i in range(len(self.param_values_to_loop_)):
#         #     param_col_ls_dict[self.param_values_to_loop_[i]] = param_col_ls[i]
#         #
#         # greys_col_ls_dict = {}
#         # for i in range(len(self.param_values_to_loop_)):
#         #     greys_col_ls_dict[self.param_values_to_loop_[i]] = greys_col_ls[i]
#         #
#         # embd_color_dict = {}
#         # for e in self.feature_encoding_ls:
#         #     embd_color_dict[e] = param_col_ls_dict
#
#
#         # Plot a single PLOT for each embedding type
#         fig, axs = plt.subplots(1,len(self.feature_encoding_ls )+ 1)
#         fig.set_size_inches(w=3*(len(self.feature_encoding_ls )+ 1), h=4.5, )  # NOTE: h and w must be large enough to accomodate any legends
#
#         all_prts_dict = {}
#         longest_prt_ = -1
#         for e, col_ in zip(self.feature_encoding_ls,list(range(len(self.feature_encoding_ls)))):  # different plots per embedding
#             axs[col_].set_title(str(e))
#             for n in range(self.num_rerurun_model_building):  # loop through rounds
#
#                 # Get p-r curves per round for given embedding and round
#                 key__ = str(e) + '_' + str(n)
#                 p = self.final_model_params_ls[n]
#
#                 # try: # get color for parameter optimization looping
#                 #     color_ = param_col_ls_dict[p]
#                 # except: # if not looping through parameters select color from list
#                 #     color_ = '#5784db'
#                 # NOTE: no p-r curves for label-propagation/spreading with one-hot encoding
#                 if not (((self.parameter_to_optimize == 'model') and ('label' in p) and (e == 'one-hot')) or (
#                         ('label' in self.model_type_) and (e == 'one-hot'))):
#
#                     pcurve__ = self.final_performance_curves_encodings_dict[key__]['Precision_Recall_Curve'][0]
#                     rcurve__ = self.final_performance_curves_encodings_dict[key__]['Precision_Recall_Curve'][1]
#                     thresholds__ = self.ext_final_performance_curves_encodings_dict[key__]['Precision_Recall_Curve'][1]
#                     unach_pcurve__ = self.ext_final_performance_curves_encodings_dict[key__]['Unacheivable_Region_Curve'][0]
#                     unach_rcurve__ = self.ext_final_performance_curves_encodings_dict[key__]['Unacheivable_Region_Curve'][1]
#
#
#
#                     axs[col_].plot(
#                         rcurve__,  # r_OH,# x
#                         pcurve__,  # p_OH,# y
#                         lw=1,
#                         color= '#1494DF',
#                     )
#
#                     pcurve__ext__ = self.ext_final_performance_curves_encodings_dict[key__]['Precision_Recall_Curve'][0]
#                     rcurve__ext__ = self.ext_final_performance_curves_encodings_dict[key__]['Precision_Recall_Curve'][1]
#
#                     axs[col_].plot(
#                         rcurve__ext__,  # r_OH,# x
#                         pcurve__ext__,  # p_OH,# y
#                         lw=1,
#                         color='#eb9834',
#                     )
#
#                     # axs[col_].plot(
#                     #     rcurve__,  # r_OH,# x
#                     #     unach_rcurve__,  # p_OH,# y
#                     #     lw=1,
#                     #     color=color_,
#                     #     linestyle='dashed',
#                     # )
#
#             # Format Axes
#             axs[col_].set_xlim(0, 1)
#             axs[col_].set_ylim(0, 1.1)
#             # axs[col_].set_xticks(ticks=np.arange(0, 1.1, .25), minor=True)
#             axs[col_].tick_params(direction='in', which='both', length=3, width=1)
#
#             if col_ == 0:
#                 axs[col_].set_xlabel('Recall')
#                 axs[col_].set_ylabel('Precision')
#                 axs[col_].set_xticks(ticks=[0.5, 1.0 ], labels=[0.5, 1.0])
#                 # axs[col_].set_xticks(ticks=[0.0, 0.5, 1.0], labels=['', 0.5, 1.0])
#                 axs[col_].set_yticks(ticks=[0.0 , 0.5, 1.0 ], labels=[0.0, 0.5, 1.0])
#
#             else:
#                 axs[col_].set_xticks(ticks=[0.5, 1.0 ], labels=[0.5, 1.0])
#                 # axs[col_].set_xticks(ticks=[0.0, 0.5, 1.0], labels=['', 0.5, 1.0])
#                 axs[col_].set_yticks(ticks=[0.0 , 0.5, 1.0 ], labels=[0.0, 0.5, 1.0])
#                 #axs[col_].set_xticks([])
#                 #axs[col_].set_yticks([])
#
#             # # Add colored border
#             # for spine in axs[col_].spines.values():
#             #     spine.set_edgecolor('red')
#             #     spine.set_linewidth(3)
#
#
#
#
#         # Add legend for parameter values
#
#         legend_elements = [
#                         Line2D([0], [0],
#                    color='#1494DF',  # embd_color_dict[embd_][val__],
#                    lw=4, label='Test Set'),
#             Line2D([0], [0],
#                    color= '#eb9834',#color_,  # embd_color_dict[embd_][val__],
#                    lw=4, label='External Dataset')
#         ]
#         axs[-1].legend(handles=legend_elements, loc='upper left', frameon=False, bbox_to_anchor=(0, 1), title=self.parameter_to_optimize, title_fontsize=12, fontsize=12)
#         axs[-1].axis('off')
#
#
#
#         random_flag_ = ''
#         if self.randomize_ext_data_:
#             random_flag_ = '(Randomized) '
#         fig.suptitle('Compiled Precision-Recall Curves Final Models Evaluated on External Dataset '+random_flag_+'- Per Embedding ' + str(self.num_rerurun_model_building) +
#                      ' rounds' + '\n' + self.output_run_file_info_string_.replace('_', ' ').replace(self.region_.replace('_', '-'), self.region_.replace('_', '-')) + '\n'+
#                      'External Dataset: '+self.external_data_file_, fontsize=9)
#
#         fig.tight_layout()  # NOTE: h and w (above in fig.set_size... MUST be large enough to accomodate legends or will be cut off/squished in output)
#
#
#         # ** SAVE FIGURE **
#         plt.rcParams['svg.fonttype'] = 'none'  # exports text as strings rather than vector paths (images)
#         fnm_ = (self.output_directory + 'figures/' + 'p-r_' + str(self.num_rerurun_model_building) + '-rnds_final_ext-data-eval_and-test-set')
#         fnm_svg_ = (self.output_directory + 'figures/' + 'svg_figs/' + 'p-r_' + str(self.num_rerurun_model_building) + '-rnds_final_ext-data-eval_and-test-set')
#         fig.savefig(fnm_svg_.split('.')[0] + '.svg', format='svg', transparent=True)
#         fig.savefig(fnm_.split('.')[0] + '.png', format='png', dpi=300, transparent=False)
#         print('Figure saved to:', fnm_ + '.png'.replace(self.output_directory, '~/'))
#         return
