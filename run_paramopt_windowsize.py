from main import DataRepresentationBuilder


##### Window Size

drb = DataRepresentationBuilder(
    model_type__ = 'semi-sup-random-forest',
    run_param_optimization__ = True,
    num_rerurun_model_building__=25,

    parameter_to_optimize__ = 'window-size',
    custom_parameter_values_to_loop__ = [1, 2, 3,  5, 8 ],#, 10, 20], # window size cannot exceed kmer size(?)

    kmer_size__=9,
    flank_len__=20,
    #window_size__= 5,
    #word_freq_cutoff__ = 5,
    encoding_ls__ = ['one-hot','bow-gensim-weights','bow-gensim-weights-times-values','bow-countvect','ann-word2vec-gensim', 'ann-keras'],
    use_existing_processed_dataset__ = False,
    apply_final_models_to_external_dataset__ = False,
)

drb = DataRepresentationBuilder(
    model_type__ = 'semi-sup-svm',
    run_param_optimization__ = True,
    num_rerurun_model_building__=25,

    parameter_to_optimize__ = 'window-size',
    custom_parameter_values_to_loop__ = [1, 2, 3,  5, 8 ],#, 10, 20], # window size cannot exceed kmer size(?)

    kmer_size__=9,
    flank_len__=20,
    #window_size__= 5,
    #word_freq_cutoff__ = 5,
    encoding_ls__ = ['one-hot','bow-gensim-weights','bow-gensim-weights-times-values','bow-countvect','ann-word2vec-gensim', 'ann-keras'],
    use_existing_processed_dataset__ = False,
    apply_final_models_to_external_dataset__ = False,
)

# drb = DataRepresentationBuilder(
#     model_type__ = 'linear-classification',
#     run_param_optimization__ = True,
#     num_rerurun_model_building__=25,
#
#     parameter_to_optimize__ = 'window-size',
#     custom_parameter_values_to_loop__ = [1, 2, 3,  5, 8 ],#, 10, 20], # window size cannot exceed kmer size(?)
#
#     kmer_size__=9,
#     flank_len__=20,
#     #window_size__= 5,
#     #word_freq_cutoff__ = 5,
#     encoding_ls__ = ['one-hot','bow-gensim-weights','bow-gensim-weights-times-values','bow-countvect','ann-word2vec-gensim', 'ann-keras'],
#     use_existing_processed_dataset__ = False,
#     apply_final_models_to_external_dataset__ = False,
# )
