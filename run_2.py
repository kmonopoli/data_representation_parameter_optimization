from main import DataRepresentationBuilder


drb = DataRepresentationBuilder(
    model_type__ = 'random-forest',
    parameter_to_optimize__ = 'None',
    custom_parameter_values_to_loop__ = [],
    run_param_optimization__ = False,

    num_rerurun_model_building__=10,
    kmer_size__=9,
    flank_len__=20,
    #window_size__= 5,
    #word_freq_cutoff__ = 5,

    encoding_ls__ = ['one-hot',

                     'bow-gensim-weights-times-values','bow-gensim-weights','bow-gensim-values',
                     'bow-gensim-weights-times-values-adjusted',
                     'bow-gensim-values-adjusted',

                     'ann-word2vec-gensim','bow-countvect'
                     ],

    use_existing_processed_dataset__ = False,

    apply_final_models_to_external_dataset__ = True,
    randomize_ext_data__ = True,

    #ext_species_ls__ = ['human'],
    #ext_chemical_scaffold_ls__=['P5'],
)