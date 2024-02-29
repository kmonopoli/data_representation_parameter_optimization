
from main import DataRepresentationBuilder

# TODO: plot model performance
# TODO: store model performance metrics
# TODO: pickle final models
# TODO: test different models (semi-supervised)
# TODO: test different paramopts (flank lengths)
# TODO: export metrics/scores to a file
# TODO: plot final model performances
# TODO: run different model types

# TODO: put on the cluster
# pr_po,k_po,pr_f,m_f,k_f


#############################################################################



# parameter_to_optimize__,
# num_rerurun_model_building__=2,  # 25,
# custom_parameter_values_to_loop__=[],
# model_type__='random-forest',
# flank_len__=50, kmer_size__=9,
# normalized__=True, effco__=25, ineffco__=60, remove_undefined__=True,
# region__='flanking_and_target_regions',
# screen_type_ls__=['bDNA'], species_ls__=['human'],
# chemical_scaffold_ls__=['P3'],
# window_size__=1, word_freq_cutoff__=1, output_dimmension__=20,
# unlabelled_data__='sequences-from-all-transcripts',
# unlabeled_data_size__=1.00,
# plot_grid_splits__=False, plot_extra_visuals__=False,
# encoding_ls__ = ['one-hot', 'ann-word2vec-gensim', 'bow-gensim', 'ann-keras', 'bow-countvect'],
# metric_used_to_id_best_po__='F-Score',



# drb = DataRepresentationBuilder(parameter_to_optimize__ ='flank-length', model_type__ = 'random-forest',num_rerurun_model_building__=25,kmer_size__=9,encoding_ls__ = ['one-hot', 'ann-word2vec-gensim', 'bow-gensim', 'ann-keras', 'bow-countvect'])
# drb = DataRepresentationBuilder(parameter_to_optimize__ ='kmer-size', model_type__ = 'random-forest',num_rerurun_model_building__=25,flank_len__=50,encoding_ls__ = ['one-hot', 'ann-word2vec-gensim', 'bow-gensim', 'ann-keras', 'bow-countvect'])
# drb = DataRepresentationBuilder(parameter_to_optimize__ = 'window_size',model_type__ = 'random-forest',num_rerurun_model_building__=25,flank_len__=50, kmer_size__=9,encoding_ls__ = ['one-hot', 'ann-word2vec-gensim', 'bow-gensim', 'ann-keras', 'bow-countvect'])
# drb = DataRepresentationBuilder(parameter_to_optimize__ = 'word_frequency_cutoff',model_type__ = 'random-forest',num_rerurun_model_building__=25,flank_len__=50, kmer_size__=9,encoding_ls__ = ['one-hot', 'ann-word2vec-gensim', 'bow-gensim', 'ann-keras', 'bow-countvect'])
# drb = DataRepresentationBuilder(parameter_to_optimize__ = 'ANN_output_dimmension',model_type__ = 'random-forest',num_rerurun_model_building__=25,flank_len__=50, kmer_size__=9,encoding_ls__ = ['one-hot', 'ann-word2vec-gensim', 'bow-gensim', 'ann-keras', 'bow-countvect'])
# drb = DataRepresentationBuilder(parameter_to_optimize__ = 'feature_encoding',model_type__ = 'random-forest',num_rerurun_model_building__=25,flank_len__=50, kmer_size__=9)

# drb = DataRepresentationBuilder(parameter_to_optimize__ = 'model',custom_parameter_values_to_loop__ =['random-forest','sup-svm','linear-classification'],num_rerurun_model_building__=5,flank_len__=50, kmer_size__=9,encoding_ls__ = ['one-hot', 'ann-word2vec-gensim', 'bow-gensim', 'ann-keras', 'bow-countvect'])

# drb = DataRepresentationBuilder(parameter_to_optimize__ = 'flank-length',model_type__ = 'semi-sup-svm',num_rerurun_model_building__=5,kmer_size__=9,encoding_ls__ = ['one-hot', 'ann-word2vec-gensim', 'bow-gensim', 'ann-keras', 'bow-countvect'])
# drb = DataRepresentationBuilder(parameter_to_optimize__ = 'kmer-size',model_type__ = 'semi-sup-svm',num_rerurun_model_building__=5,flank_len__=50,encoding_ls__ = ['one-hot', 'ann-word2vec-gensim', 'bow-gensim', 'ann-keras', 'bow-countvect'])
#
# # NEED TO RUN NEXT
# drb = DataRepresentationBuilder(parameter_to_optimize__ = 'kmer-size', model_type__ = 'linear-classification',
#                                 custom_parameter_values_to_loop__ = [3,9] ,num_rerurun_model_building__=10,
#                                 flank_len__=50,encoding_ls__ = ['one-hot', 'ann-word2vec-gensim', 'bow-gensim'])

# drb = DataRepresentationBuilder(parameter_to_optimize__ = 'window_size',model_type__ = 'semi-sup-random-forest',num_rerurun_model_building__=25,flank_len__=50, kmer_size__=9,encoding_ls__ = ['one-hot', 'ann-word2vec-gensim', 'bow-gensim', 'ann-keras', 'bow-countvect'])
# drb = DataRepresentationBuilder(parameter_to_optimize__ = 'word_frequency_cutoff',model_type__ = 'semi-sup-random-forest',num_rerurun_model_building__=25,flank_len__=50, kmer_size__=9,encoding_ls__ = ['one-hot', 'ann-word2vec-gensim', 'bow-gensim', 'ann-keras', 'bow-countvect'])
# drb = DataRepresentationBuilder(parameter_to_optimize__ = 'ANN_output_dimmension',model_type__ = 'semi-sup-random-forest',num_rerurun_model_building__=25,flank_len__=50, kmer_size__=9,encoding_ls__ = ['one-hot', 'ann-word2vec-gensim', 'bow-gensim', 'ann-keras', 'bow-countvect'])
# drb = DataRepresentationBuilder(parameter_to_optimize__ = 'unlabeled_data_type',model_type__ = 'semi-sup-random-forest',num_rerurun_model_building__=25,flank_len__=50, kmer_size__=9,encoding_ls__ = ['one-hot', 'ann-word2vec-gensim', 'bow-gensim', 'ann-keras', 'bow-countvect'])
# drb = DataRepresentationBuilder(parameter_to_optimize__ = 'feature_encoding',model_type__ = 'semi-sup-random-forest',num_rerurun_model_building__=25,flank_len__=50, kmer_size__=9)
#
# drb = DataRepresentationBuilder(parameter_to_optimize__ = 'model',num_rerurun_model_building__=25,flank_len__=50, kmer_size__=9,encoding_ls__ = ['one-hot', 'ann-word2vec-gensim', 'bow-gensim', 'ann-keras', 'bow-countvect'])


# # # drb = DataRepresentationBuilder(model_type__ = 'random-forest', parameter_to_optimize__ = 'kmer-size', custom_parameter_values_to_loop__ = [3,8,9] ,num_rerurun_model_building__=5,flank_len__=10,
# # #                                 encoding_ls__ = ['one-hot', 'ann-word2vec-gensim'])#, 'bow-gensim', 'ann-keras', 'bow-countvect'])


# drb = DataRepresentationBuilder(model_type__ = 'semi-sup-random-forest', parameter_to_optimize__ = 'kmer-size', custom_parameter_values_to_loop__ = [9] ,num_rerurun_model_building__=2,flank_len__=50,
#                                 encoding_ls__ = ['one-hot', 'ann-word2vec-gensim', 'bow-gensim', 'ann-keras', 'bow-countvect'])
# drb = DataRepresentationBuilder(model_type__ = 'sup-svm', parameter_to_optimize__ = 'kmer-size', custom_parameter_values_to_loop__ = [9] ,num_rerurun_model_building__=2,flank_len__=50,
#                                 encoding_ls__ = ['one-hot', 'ann-word2vec-gensim', 'bow-gensim', 'ann-keras', 'bow-countvect'])
# drb = DataRepresentationBuilder(model_type__ = 'semi-sup-svm', parameter_to_optimize__ = 'kmer-size', custom_parameter_values_to_loop__ = [3,9] ,
#                                 num_rerurun_model_building__=2,flank_len__=10,
#                                 encoding_ls__ = ['one-hot', 'ann-word2vec-gensim'])#, 'bow-gensim', 'ann-keras', 'bow-countvect'])

# drb = DataRepresentationBuilder(parameter_to_optimize__ = 'kmer-size', model_type__ = 'random-forest', custom_parameter_values_to_loop__ = [9]
#                                 ,num_rerurun_model_building__=25,flank_len__=50,encoding_ls__ = ['one-hot', 'ann-word2vec-gensim', 'bow-gensim'])

# drb = DataRepresentationBuilder(model_type__ = 'linear-classification', parameter_to_optimize__ = 'kmer-size', custom_parameter_values_to_loop__ = [9] ,num_rerurun_model_building__=2,flank_len__=50,
#                                 encoding_ls__ = ['one-hot', 'ann-word2vec-gensim', 'bow-gensim', 'ann-keras', 'bow-countvect'])
# drb = DataRepresentationBuilder(model_type__ = 'semi-sup-label-propagation', parameter_to_optimize__ = 'kmer-size', custom_parameter_values_to_loop__ = [9] ,num_rerurun_model_building__=2,flank_len__=50,
#                                 encoding_ls__ = ['one-hot', 'ann-word2vec-gensim', 'bow-gensim', 'ann-keras', 'bow-countvect'])

# drb = DataRepresentationBuilder(model_type__ = 'semi-sup-label-spreading', parameter_to_optimize__ = 'kmer-size', custom_parameter_values_to_loop__ = [2,9] ,num_rerurun_model_building__=2,flank_len__=50,
#                                 encoding_ls__ = ['one-hot', 'ann-word2vec-gensim'])#, 'bow-gensim', 'ann-keras', 'bow-countvect'])


##########################################################################################################################################################


##########################################################################################################################################################

# drb = DataRepresentationBuilder(model_type__ = 'semi-sup-random-forest',
#                                 parameter_to_optimize__ = 'kmer-size',
#                                 custom_parameter_values_to_loop__ = [3, 5, 7, 9, 12, 15, 17, 20],
#                                 num_rerurun_model_building__=25,
#                                 flank_len__=10,
#                                 encoding_ls__ = ['one-hot', 'ann-word2vec-gensim', 'bow-gensim'],
#                                 )

# drb = DataRepresentationBuilder(model_type__ = 'semi-sup-random-forest',
#                                 parameter_to_optimize__ = 'kmer-size',
#                                 custom_parameter_values_to_loop__ = [3, 5, 7, 9, 12, 15, 17, 20],
#                                 num_rerurun_model_building__=25,
#                                 flank_len__=10,
#                                 encoding_ls__ = ['one-hot', 'ann-word2vec-gensim', 'bow-gensim'],
#                                 )

# drb = DataRepresentationBuilder(model_type__ = 'random-forest',
#                                 parameter_to_optimize__ = 'None',
#                                 custom_parameter_values_to_loop__ = [],
#                                 num_rerurun_model_building__=2,
#                                 kmer_size__=9,
#                                 flank_len__=20,
#                                 encoding_ls__ = ['one-hot','bow-gensim-weights-times-values','bow-gensim-weights','bow-gensim-values','ann-word2vec-gensim',],#'ann-word2vec-gensim'],#['bow-countvect', 'one-hot'],
#                                 run_param_optimization__ = False,
#                                 use_existing_processed_dataset__ = False,
#                                 apply_final_models_to_external_dataset__ = True,
#                                 #ext_species_ls__ = ['human'],
#                                 #ext_chemical_scaffold_ls__=['P5'],
#                                 randomize_ext_data__ = False,
#
#                                 )

# drb = DataRepresentationBuilder(model_type__ = 'random-forest',
#                                 parameter_to_optimize__ = 'flank-length',
#                                 custom_parameter_values_to_loop__ = [0,  20,  50,  100],
#                                 num_rerurun_model_building__=10,
#                                 flank_len__=10,
#                                 encoding_ls__ = ['ann-word2vec-gensim', 'bow-gensim'],
#                                 run_param_optimization__ = False,
#                                 )

# drb = DataRepresentationBuilder(model_type__ = 'semi-sup-random-forest',
#                                 parameter_to_optimize__ = 'flank-length',
#                                 custom_parameter_values_to_loop__ = [0, 5, 10, 20, 25, 50, 75, 100],
#                                 num_rerurun_model_building__=25,
#                                 flank_len__=10,
#                                 encoding_ls__ = ['one-hot', 'ann-word2vec-gensim', 'bow-gensim'],
#                                 )

# drb = DataRepresentationBuilder(
#     model_type__ = 'random-forest',
#     parameter_to_optimize__ = 'None',
#     custom_parameter_values_to_loop__ = [],
#     run_param_optimization__ = False,
#
#     num_rerurun_model_building__=2,
#     kmer_size__=9,
#     flank_len__=20,
#     #window_size__= 5,
#     #word_freq_cutoff__ = 5,
#
#     encoding_ls__ = ['one-hot',
#
#                      'bow-gensim-weights-times-values','bow-gensim-weights','bow-gensim-values',
#                      'bow-gensim-weights-times-values-adjusted',
#                      'bow-gensim-values-adjusted',
#
#                      'ann-word2vec-gensim','bow-countvect'
#                      ],
#
#     use_existing_processed_dataset__ = False,
#
#     apply_final_models_to_external_dataset__ = True,
#     randomize_ext_data__ = False,
#
#     #ext_species_ls__ = ['human'],
#     #ext_chemical_scaffold_ls__=['P5'],
# )

drb = DataRepresentationBuilder(
    model_type__ = 'linear-classification',
    run_param_optimization__ = True,
    num_rerurun_model_building__=2,

    parameter_to_optimize__ = 'kmer-size',#flank-length',
    custom_parameter_values_to_loop__ = [2,20],#[20,50],

    # kmer_size__=9,
    flank_len__=20,
    #window_size__= 5,
    #word_freq_cutoff__ = 5,
    encoding_ls__ = ['one-hot','bow-gensim-weights','bow-gensim-weights-times-values','bow-countvect','ann-word2vec-gensim', 'ann-keras'],
    use_existing_processed_dataset__ = False,
    apply_final_models_to_external_dataset__ = False,
)



