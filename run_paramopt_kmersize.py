from main import DataRepresentationBuilder

########### kmer size

drb = DataRepresentationBuilder(
    model_type__ = 'sup-svm',
    run_param_optimization__ = True,
    num_rerurun_model_building__=25,

    parameter_to_optimize__ = 'kmer-size',
    custom_parameter_values_to_loop__ = [2,3,5,7,9,12,15,17,20],

    flank_len__=20,
    #window_size__= 5,
    #word_freq_cutoff__ = 5,
    encoding_ls__ = ['one-hot','bow-gensim-weights','bow-gensim-weights-times-values','bow-countvect','ann-word2vec-gensim', 'ann-keras'],
    use_existing_processed_dataset__ = False,
    apply_final_models_to_external_dataset__ = False,
)

drb = DataRepresentationBuilder(
    model_type__ = 'random-forest',
    run_param_optimization__ = True,
    num_rerurun_model_building__=25,

    parameter_to_optimize__ = 'kmer-size',
    custom_parameter_values_to_loop__ = [2,3,5,7,9,12,15,17,20],

    flank_len__=20,
    #window_size__= 5,
    #word_freq_cutoff__ = 5,
    encoding_ls__ = ['one-hot','bow-gensim-weights','bow-gensim-weights-times-values','bow-countvect','ann-word2vec-gensim', 'ann-keras'],
    use_existing_processed_dataset__ = False,
    apply_final_models_to_external_dataset__ = False,
)



drb = DataRepresentationBuilder(
    model_type__ = 'linear-classification',
    run_param_optimization__ = True,
    num_rerurun_model_building__=25,

    parameter_to_optimize__ = 'kmer-size',
    custom_parameter_values_to_loop__ = [2,3,5,7,9,12,15,17,20],

    flank_len__=20,
    #window_size__= 5,
    #word_freq_cutoff__ = 5,
    encoding_ls__ = ['one-hot','bow-gensim-weights','bow-gensim-weights-times-values','bow-countvect','ann-word2vec-gensim', 'ann-keras'],
    use_existing_processed_dataset__ = False,
    apply_final_models_to_external_dataset__ = False,
)