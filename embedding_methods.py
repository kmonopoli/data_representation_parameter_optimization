


# helper functions
def get_kmers_(seq__, kmer_size__, window_size__):
    """ Returns kmers for a SINGLE sequence """
    return [''.join(seq__[i:kmer_size__+i]) for i in range(len(seq__)-kmer_size__+window_size__)]
## Testing:
## get_kmers_('TTCAAAAATAGTGACTCAGAAAAGGACAATTCAAAAAGGACATTAC',kmer_size__ = 6,window_size__ = 1)

def pad_list_(list_, ref_len_):
    # Create an array of zeros with the reference shape
    to_pad_ = ref_len_-len(list_)
    return list_+[0]*to_pad_

def pad_corpus_(list_, ref_len_):
    # Create an array of zeros with the reference shape
    to_pad_ = ref_len_-len(list_)
    return list_+['']*to_pad_

def embed_sequences_with_bow_countvect(seq_ls, kmer_size_, window_size_, word_freq_cutoff_):
    '''
    ##                                                                                         
    ##     ~*~ Performs Sequence Data Embedding as Bag-of-Words Sklearn's CountVectorizer  ~*~
    ##
    ## Takes a list of sequences (seq_ls) as input and returns a list of encoded sequences
    ##    - Encodes sequence data as kmer integer words with sliding window
    ##
    ## word_freq_cutoff_ -->  Number of times a word must occur in the Bag-of-words Corpus 
    ##      * when word_freq_cutoff = 1 only include words that occur more than once # TODO: adjust this parameter
    ##                                                                                       
    #########################################################################################
    '''
    from sklearn.feature_extraction.text import CountVectorizer
    # count_vect = CountVectorizer(ngram_range=(kmer_size_,kmer_size_), analyzer = 'char_wb',min_df=word_freq_cutoff_)#,max_features=100)
    count_vect = CountVectorizer(ngram_range=(kmer_size_,kmer_size_), analyzer = 'char',min_df=word_freq_cutoff_)#,max_features=100)

    count_vect.fit(seq_ls)
    bow_model = count_vect.transform(seq_ls)
    #bow_model.shape
    # count_vect.get_feature_names_out()

    # output to list format
    out_ls_ = [list(bow_model.toarray()[i_]) for i_ in list(range(len(seq_ls)))]
    if len(str(out_ls_[0][0])) >= 32000:
        raise Exception('ERROR: exported vector for bow countvectorizer embeddings is longer than 32000, update embedding parameters to shorten vector sizes')

    #**#print('BOW embedding using CountVectorizer complete!')
    return out_ls_


def embed_sequences_with_gensim_doc2bow_tfidf(seq_ls, kmer_size_, window_size_, word_freq_cutoff_, vector_output_ = 'weights',
                                        # TODO: Below all for troubleshooting remove later
                                        data_expr_pcnts_ = [], data_numeric_classes_= [], data_classes_= [],
                                        data_source_ = [], data_oligo_names_ = [], data_expr_pcnts_norm_ = [],
                                        ):
    '''
    ##
    ##      ~*~ Performs Sequence Data Embedding as Bag-of-Words using Gensim  ~*~
    ##
    ## Takes a list of sequences (seq_ls) as an input and returns a list of encoded sequences
    ##    - Encodes sequence data as kmer integer words with sliding window
    ## word_freq_cutoff_ -->  Number of times a word must occur in the Bag-of-words Corpus 
    ##      * when word_freq_cutoff = 1 only include words that occur more than once
    ##
    ## https://radimrehurek.com/gensim/auto_examples/core/run_core_concepts.html#core-concepts-vector
    ##
    #########################################################################################
    '''
    import numpy as np
    import pandas as pd

    #**#print('\n\n\n\n*****************************************\n\n')
    #**#print("KMER SIZE:",kmer_size_)
    #**#print('WINDOW SIZE:',window_size_)
    #**#print('seq_ls[0]:',seq_ls[0][0:20])
    #**#print(set([type(x) for x in seq_ls]))
    texts_sirna = [get_kmers_(x,kmer_size_,window_size_) for x in seq_ls]

    


    # pos_ = -2
    # len_ = 14

    # Count word frequencies
    from collections import defaultdict
    kmer_freq = defaultdict(int)
    for text in texts_sirna:
        for token in text:
            kmer_freq[token] += 1



    max_freq_ = 0
    kmer_with_max_freq_ = ""
    cts_kmer_freq_0 = 0
    cts_kmer_freq_1 = 0
    cts_all_ls = []
    for k in kmer_freq:
        cts_all_ls.append(kmer_freq[k])
        if kmer_freq[k]>max_freq_:
            max_freq_ = kmer_freq[k]
            kmer_with_max_freq_ = k
        if kmer_freq[k] == 0:
            cts_kmer_freq_0+=1
        elif kmer_freq[k] == 1:
            cts_kmer_freq_1+=1
    import pandas as pd
    cts_all_ls = pd.Series(cts_all_ls)


    processed_corpus_sirna = [[token for token in text if kmer_freq[token] > word_freq_cutoff_] for text in texts_sirna]



    # Build Dictionary of K-mers
    from gensim import corpora

    dictionary_sirna = corpora.Dictionary(processed_corpus_sirna)

    ## Vectorize Corpus
    # Using Bag-of-words representation
    #**NOTE**: it is possible for two unique seequences to have **IDENTICAL vectorization**
    bow_corpus_sirna = [dictionary_sirna.doc2bow(seq) for seq in processed_corpus_sirna]


    # Get information about vectorized sequence lengths:
    bow_corpus_sirna_lens_ = pd.Series([len(x) for x in bow_corpus_sirna])
    bow_corpus_sirna_lens_df_ = pd.DataFrame(bow_corpus_sirna_lens_.value_counts())
    bow_corpus_sirna_lens_df_.index.name="Length of Vector"
    bow_corpus_sirna_lens_df_.columns = ['Number of Sequences']
    bow_corpus_sirna_lens_df_.reset_index(inplace=True)


    # TODO: for troubleshooting (delete/commment out below)
    import random
    rand_seed_enc_lab__ = str(random.randint(1000,9999))
    embedding_dict_dir = '/Users/kmonopoli/Dropbox (UMass Medical School)/data_representation-sequences/cleaned_parameter_optimization/data_representation_parameter_optimization/embedding_dictionaries/'
    # TODO: for troubleshooting (delete/commment out above)



    # TODO: for troubleshooting (delete/commment out below)
    # Export bag of words corpus lengths to .csv file
    bow_corpus_lens_fnm__ = rand_seed_enc_lab__+'_bow_corpus_lengths_dict_kmer-'+str(kmer_size_)+'_window-'+str( window_size_)+'_word-freq-co-'+str(word_freq_cutoff_)+'.csv'
    bow_corpus_sirna_lens_df_.to_csv(embedding_dict_dir+bow_corpus_lens_fnm__)
    #**#print('Bag-of-words corpus lengths saved to:\n\t','embedding_dictionaries/'+bow_corpus_lens_fnm__)
    # TODO: for troubleshooting (delete/commment out above)



    # TODO: for troubleshooting (delete/commment out below)
    # Export bow_corpus_sirna to a file
    ## B-O-W corpus:
    ## [
    ##  seq1 --> [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1)] ,
    ##  seq2 --> [(12, 1), (13, 1), (14, 1), (15, 1), (16, 1), (17, 1), (18, 1), (19, 1), (20, 1)] ,
    ##  seq3 --> [(21, 1), (22, 1), (23, 1), (24, 1), (25, 1), (26, 1), (27, 1), (28, 1), (29, 1), ... , (71, 1)] ,
    ##  ... ,
    ##  seqN --> [(654, 1), (1707, 1), (1732, 1), (2751, 1), (2774, 1), (4518, 1), (5542, 1), (5543, 1), ... , (16536, 1)] ,
    ## ]
    ##
    ##########################################################################
    ## seq1 --> [(kmer_1__key_num, kmer_1__count), (kmer_2__key_num, kmer_2__count), ..., (kmer_12__key_num, kmer_12__count)]
    ##
    bow_corpus_sirna_fnm__ = rand_seed_enc_lab__ + '_bow_corpus_sirna' + '_kmer-' + str(kmer_size_) + '_window-' + str(window_size_) + '_word-freq-co-' + str(word_freq_cutoff_) + '.txt'
    with open(embedding_dict_dir + bow_corpus_sirna_fnm__, 'w+') as f:
        for b in bow_corpus_sirna:
            f.write(str(b) + ' ,\n')
    f.close()
    #**#print('Bag-of-words corpus saved to:\n\t', 'embedding_dictionaries/' + bow_corpus_sirna_fnm__)
    # TODO: for troubleshooting (delete/commment out above)

    #########################################################################################
    ##     Transform Corpus Vectorization using tf-idf model                               ##
    ##        TODO: consider other transformations/models see:                             ##
    ##            `sphx_glr_auto_examples_core_run_topics_and_transformations.py`          ##
    ##                https://radimrehurek.com/gensim/auto_examples/core/run_topics_and_transformations.html
    #########################################################################################
    # # **tf-idf model**: transforms vectors from bag-of-words representation to a vector space where the frequency 
    # #                   counts are weighted according to the relative rarity of each word in the corpus

    from gensim import models
    # train the model on the corpus
    tfidf_sirna = models.TfidfModel(bow_corpus_sirna) 

    # Transform all sequences in Corpus
    # 1) get kmers
    kmers_all = [get_kmers_(x,kmer_size_,window_size_) for x in seq_ls]
    # 2) vectorize
    vects_all = [dictionary_sirna.doc2bow(x) for x in kmers_all]


    # # TODO: for troubleshooting (delete/commment out below)
    # # **SAME AS bag-of-words corpus** (exported to file above above)
    # # Export vectors to a file
    # vect_fnm__ = rand_seed_enc_lab__ + '_vects-all'  + '_kmer-' + str(kmer_size_) + '_window-' + str(window_size_) + '_word-freq-co-' + str(word_freq_cutoff_) + '.txt'
    # with open(embedding_dict_dir + vect_fnm__,'w+') as f:
    #     for v in vects_all:
    #         f.write(str(v)+' ,\n')
    # f.close()
    # #**#print('Vectors saved to:\n\t', 'embedding_dictionaries/' + vect_fnm__)
    # # TODO: for troubleshooting (delete/commment out above)


    # 3) Transform vectorization to tfidf vector space
    tfidf_vecs_all = [tfidf_sirna[x] for x in vects_all]

    # TODO: for troubleshooting (delete/commment out below)
    # Export Transformed TF-IDIF vectors to a file
    tfidf_vect_fnm__ = rand_seed_enc_lab__ + '_tfidf-vects'  + '_kmer-' + str(kmer_size_) + '_window-' + str(window_size_) + '_word-freq-co-' + str(word_freq_cutoff_) + '.txt'
    with open(embedding_dict_dir + tfidf_vect_fnm__, 'w+') as f:
        for t in tfidf_vecs_all:
            f.write(str(t) + ' ,\n')
    f.close()
    #**#print('Vectors transformed to TF-IDF vector space saved to:\n\t', 'embedding_dictionaries/' + tfidf_vect_fnm__)
    # TODO: for troubleshooting (delete/commment out above)



    # NOTE: vectorized sequences do not have uniform shapes (see below)
    # Get information about vectorized sequence lengths:
    tfidf_vecs_all_lens_ = pd.Series([len(x) for x in tfidf_vecs_all])
    tfidf_vecs_all_lens_df_ = pd.DataFrame(tfidf_vecs_all_lens_.value_counts())
    tfidf_vecs_all_lens_df_.index.name="Length of Vector"
    tfidf_vecs_all_lens_df_.columns = ['Number of Sequences']
    tfidf_vecs_all_lens_df_.reset_index(inplace=True)


    tfidf_vecs_all_lens_df_.sort_values(by=['Length of Vector'],ascending=False,inplace=True)
    tfidf_vecs_all_lens_df_.reset_index(drop=True,inplace=True)

    # # TODO: for troubleshooting (delete/commment out below)
    # # Export tfidf vector lengths(?) dataframe to .csv file
    # # **SAME AS bag of words corpus lengths (exported to file above)
    # tfidf_vector_lens_fnm__ = rand_seed_enc_lab__ + '_tfidf-vector-lens-df'    + '_kmer-' + str(kmer_size_) + '_window-' + str(window_size_) + '_word-freq-co-' + str(word_freq_cutoff_) + '.csv'
    # tfidf_vecs_all_lens_df_.to_csv(embedding_dict_dir + tfidf_vector_lens_fnm__)
    # #**#print('TF-IDF vector lengths(?) saved to:\n\t', 'embedding_dictionaries/' + tfidf_vector_lens_fnm__)
    # # TODO: for troubleshooting (delete/commment out above)

    ## Pad Vectors so all same length 
    max_vect_len = tfidf_vecs_all_lens_df_.iloc[0]['Length of Vector']+2

    padded_tfidf_vecs_all = []
    for v in tfidf_vecs_all:
        if len(v) < max_vect_len:
            num_to_pad_ = max_vect_len-len(v)
            # add padding
            padded_v_ = v+[(0,0)]*num_to_pad_
            padded_tfidf_vecs_all.append(padded_v_)
        else:
            padded_tfidf_vecs_all.append(v)


    ## Convert to form useable in training by sklearn models
    ##    Either:
    ##      1) Flattened arrays 
    ##      2) Second value (tf-idf weight) for each kmer <-- used this originally (padded_tfidf_weight_vecs_all)
    ##      3) Multiply tf-idf weight by value <--- NOW using this (padded_tfidf_mult_vecs_all)
    # padded_tfidf_weight_vecs_all =  [np.array([y[1] for y in x]) for x in padded_tfidf_vecs_all]
    # padded_tfidf_flat_arr_vecs_all =  [np.array([y[0] for y in x]) for x in padded_tfidf_vecs_all]
    # padded_tfidf_mult_vecs_all =  [np.array([y[0]*y[1] for y in x]) for x in padded_tfidf_vecs_all]

    # TODO: BELOW is for troubleshooting remove later


    # Build a dictionary of the weight information for each sequence (to later transform to dataframe to export to .csv file for troubleshooting)
    bow_temp_label_data_dict__ = {
        'count':[],
        'oligo_name':[],
        'data_source': [],  # i.e., labeled data, external dataset, etc.
        'class': [],
        'numeric_class': [],
        'expression_percent_normalized': [],
        'expression_percent':[],
        'weights_vector': [],
        'values_vector': [],
        'weights_times_values_vector': []
    }

    #**#print('\n\n\n\n\n*****\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/****\n')
    #**#print('\n\nLENGTHS:', len(padded_tfidf_vecs_all), len(data_classes_), '\n\n\n')

    for x, dat__, src__, nm__, expcnt_norm__, expcnt__, num_class__, ct in zip(
            padded_tfidf_vecs_all,
            data_classes_,
            data_source_,
            data_oligo_names_,
            data_expr_pcnts_norm_,
            data_expr_pcnts_,
            data_numeric_classes_,
            list(range(len(data_classes_)))):

        bow_temp_label_data_dict__['count'].append(ct+1)
        bow_temp_label_data_dict__['oligo_name'].append(nm__)
        bow_temp_label_data_dict__['data_source'].append(src__)
        bow_temp_label_data_dict__['class'].append(dat__)
        bow_temp_label_data_dict__['numeric_class'].append(num_class__)
        bow_temp_label_data_dict__['expression_percent_normalized'].append(expcnt_norm__)
        bow_temp_label_data_dict__['expression_percent'].append(expcnt__) # using expression_key (usually the same as expcnt_norm__ and expression_percent_normalized)
        bow_temp_label_data_dict__['weights_vector'].append([y[1] for y in x])
        bow_temp_label_data_dict__['values_vector'].append( [y[0] for y in x])
        bow_temp_label_data_dict__['weights_times_values_vector'] .append([y[1] * y[0] for y in x])

    #     #**#print('\n')
    #     #**#print(str(ct+1))
    #     #**#print('class:',dat__)
    #     #**#print('data source:',src__)
    #     #**#print('oligo name:', nm__)
    #     #**#print('expression percent:', expcnt_norm__)
    #     #**#print('weight:',[y[1] for y in x])
    #     #**#print('sum weight:',sum([y[1] for y in x]))
    #     #**#print('value:',[y[0] for y in x])
    #     #**#print('sum value:', sum([y[0] for y in x]))
    #     #**#print('weight*value:',[(y[0])*y[1] for y in x])
    #     #**#print('sum weight*value:', sum([(y[0])*y[1] for y in x]))
    #     #**#print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
    #
    # #**#print('\n\n\n\n\n*****/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\****\n')

    # Export bow_temp_label_data_df__ to .csv file
    bow_temp_label_data_df_fnm__ = rand_seed_enc_lab__ + '_bow-tfidf-dataframe' +  '_kmer-' + str(kmer_size_) + '_window-' + str(window_size_) + '_word-freq-co-' + str(word_freq_cutoff_) + '.csv'
    bow_temp_label_data_df__ = pd.DataFrame(bow_temp_label_data_dict__).transpose()
    bow_temp_label_data_df__.to_csv(embedding_dict_dir + bow_temp_label_data_df_fnm__)
    #**#print('Bag-of-words tf-idf dataframe saved to:\n\t', 'embedding_dictionaries/' + bow_temp_label_data_df_fnm__)

    # Export ow_temp_label_data_dict__ to .txt file
    bow_temp_label_data_dict_fnm__ = rand_seed_enc_lab__ + '_bow-tfidf-dict' + '_kmer-' + str(kmer_size_) + '_window-' + str(window_size_) + '_word-freq-co-' + str(word_freq_cutoff_) + '.txt'
    with open(embedding_dict_dir + bow_temp_label_data_dict_fnm__, 'w+') as f:
        for key in list(bow_temp_label_data_dict__.keys()):
            f.write(str(bow_temp_label_data_dict__[key]) + ' ,\n')
    f.close()
    #**#print('Bag-of-words tf-idf dictionary saved to:\n\t', 'embedding_dictionaries/' + bow_temp_label_data_dict_fnm__)
    # TODO: for troubleshooting (delete/commment out above)


    # TODO: ABOVE is for troubleshooting remove later


    if vector_output_ == 'weights':
        padded_tfidf_vecs_all =  [list([y[1] for y in x]) for x in padded_tfidf_vecs_all]

    elif vector_output_ == 'values':
        padded_tfidf_vecs_all =  [list([y[0] for y in x]) for x in padded_tfidf_vecs_all]

    elif vector_output_ == 'weights-times-values':
        padded_tfidf_vecs_all =  [list([(y[0])*y[1] for y in x]) for x in padded_tfidf_vecs_all]

    elif vector_output_ == 'weights-times-values-adjusted':
        padded_tfidf_vecs_all = [list([(y[0]) * (y[1]*1000) for y in x]) for x in padded_tfidf_vecs_all]

    elif vector_output_ == 'values-adjusted':
        padded_tfidf_vecs_all =  [list([(y[0]*1000) for y in x]) for x in padded_tfidf_vecs_all]


    if len(str(list(padded_tfidf_vecs_all)[0][0])) >= 32000:
        raise Exception('ERROR: exported vector for gensim bow embeddings is longer than 32000, update embedding parameters to shorten vector sizes')
    #**#print('BOW embedding using Gensim doc2bow complete!')
    return list(padded_tfidf_vecs_all)




    
    
    
    
def embed_sequences_with_keras(seq_ls, kmer_size_, window_size_, output_dimmension_ = 20):
    '''
    ## Takes a list of sequences (seq_ls) as input and returns a list of encoded sequences
    ##    - Encodes sequence data as kmer integer words with sliding window
    ##                                                                                    
    ##   ~*~ Performs Sequence Data Embedding using Keras ANN ~*~  
    ##                                                                                     
    #########################################################################################
    '''
    import numpy as np
    import pandas as pd
    import tensorflow as tf # NOTE: tensorflow can take a bit of time to load
    from gensim import corpora

    ## Change output to log file
    
    
    # integer encode the sequence
    iencode_dict = {
        'A':'1',
        'U':'2','T':'2',
        'C':'3',
        'G':'4',
        'X':'0',
    }

    i_encoded_seq_ls = [[iencode_dict[y] for y in x] for x in seq_ls]

    i_encoded_seq_ls = [[int(''.join(x[i:kmer_size_+i])) for i in range(len(x)-kmer_size_+window_size_)] for x in i_encoded_seq_ls]
    
    # Now each sequence consists of a list that is a sentence of kmer words (encoded as integers) representing the sequence seq
    

    # Now convert encoded sequence integers to a range of integers from 0 to vocabulary size:
    # if each word is a kmer then for k=6 the total vocabulary size is:
    #    4**6 = 4096
    # so we can encode these words using integers 0 to 4095
    all_enc_seqs_ls = list(set([item for sublist in i_encoded_seq_ls for item in sublist]))
    all_enc_seqs_ls.sort() # sort the list smallest to largest
    vocab_size = len(all_enc_seqs_ls)
    
    ## Create dataframe holding the integer conversions and then convert that to a dictionary for quick lookups
    enc_df = pd.DataFrame(list(zip(all_enc_seqs_ls,list(range(len(all_enc_seqs_ls))))), columns =['old_encoding','new_encoding'])
    enc_df.index=enc_df['old_encoding']
    enc_df.drop(columns=['old_encoding'],inplace=True)
    enc_dict = enc_df.to_dict()['new_encoding']
    #fnm_ = output_directory+'encoding-dict_cnn_embedding'+'_seqs-'+str(len(seq_ls))+'_kmer-'+str(kmer_size_)+'_window-'+str(window_size_)+'_'+str(date_)+'.csv'
    #enc_df.to_csv(fnm_,index=False)

    i_encoded_seq_ls = [np.array([enc_dict[y] for y in x]) for x in i_encoded_seq_ls]

    ## Build an Artificial Neural Network (ANN)
    ##   - will use ANN word embedding layer to transform kmer "sentence" into a dense feature vector matrix
    model = tf.keras.Sequential()
    
    ## Get parameters for Embedded layer (will be added next)
    input_dim_ = vocab_size # vocabulary size (determined from dataset)
    output_dim_ = output_dimmension_  # shape of output array's 3rd dimmension (length of single "feature" from a set of "features" for a single training set element)
    input_length_=len(i_encoded_seq_ls[0]) # flank_len_*2+20-(kmer_size_-1)

    ## Add Embedded layer to the model 
    ##   - Layer converts words to a vector space model 
    ##   - Converts words based on how often it appears close to other words
    ##   - Layer uses random weights to learn embedding for all of the terms in the training dataset
    model.add(tf.keras.layers.Embedding(
        input_dim = input_dim_,
        output_dim = output_dim_,
        input_length= input_length_,
    ))   

    from keras.layers import Dense
    from keras.layers import Flatten

    model.add(Flatten())
    model.add(Dense(output_dim_, activation='sigmoid')) # TODO: possibly remove/change activation function

    
    # Configure the model with losses and metrics for training
    #model.compile(optimizer='rmsprop', loss='mse') # TODO: consider other parameters for compile --> optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    ###**#print(model.summary())

    # Model will take as input a matrix of size (batch, input_length)
    input_array = np.array(i_encoded_seq_ls)
    
    ###**#print('\nCNN Input matrix shape:',input_array.shape)

    # Transform kmer "sentence" into a dense feature vector matrix using CNN word Embedding layer
    output_array = model.predict(input_array)
    ###**#print('\nCNN Output matrix shape:',output_array.shape)

    # convert array to form that can be easily added to dataframe
    output_array = [list(x) for x in list(output_array)]
    
    if len(str(output_array[0][0])) >= 32000:
        raise Exception('ERROR: exported vector for keras deep ann embeddings is longer than 32000, update embedding parameters to shorten vector sizes')
    #**#print('Keras Embedding Complete!')
    return output_array







def embed_sequences_with_gensim_word2vec_skipgram(seq_ls, kmer_size_, window_size_, word_freq_cutoff_):
    ''' NEW UPDATED WORD2VEC METHOD'''
    ## UPDATED to use Word2Vec correctly - Skip-gram model: Using the context to predict a target word
    ## https://michael-fuchs-python.netlify.app/2021/09/01/nlp-word-embedding-with-gensim-for-text-classification/#gensim---word2vec
    from gensim.models import Word2Vec

    vector_size_n_w2v = 120 # TODO: update to something higher(?) 100?

    w2v_model = Word2Vec(vector_size=vector_size_n_w2v,
                         window=5, # number of words to consider on each side for the context TODO: update to larger value?
                         min_count=word_freq_cutoff_,
                         sg=1)  # 0=CBOW, 1=Skip-gram

    texts_sirna = [get_kmers_(x, kmer_size_, window_size_) for x in seq_ls]

    w2v_model.build_vocab(texts_sirna) # Create the vocabulary, which is to be learned by Word2Vec
    ##**#print(w2v_model)

    w2v_model.train(texts_sirna, # Train Neural Network over 5 epochs (NOTE: this can take some time)
                    total_examples=w2v_model.corpus_count,
                    epochs=50) # TODO: update epochs?

    # Save to Word2Vec Model (and the vector size
    w2v_model.save("embedding_dictionaries/word2vec_model")
    import pickle as pk
    pk.dump(vector_size_n_w2v, open('embedding_dictionaries/vector_size_w2v_metric.pkl', 'wb'))

    # EXAMPLE: Output of the calculated vector for a given word (kmer) from the vocabulary:
    # single_example_kmer__ = texts_sirna[0][0]
    # w2v_model.wv[single_example_kmer__]
    # # EXAMPLE: Display the words that are most similar to a given word from the vocabulary:
    # w2v_model.wv.most_similar(single_example_kmer__)

    # Generate aggregate sentence vectors based on the kmer vectors for each kmer in the given sequence
    words = set(w2v_model.wv.index_to_key)
    import numpy as np
    text_vect_ls = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                                for ls in texts_sirna ])

    # NOTE: since specify vector initially no need to pad them (since all same length)
    # Generation of averaged Sentence Vectors
    text_vect_avg = []
    for v in text_vect_ls:
        if v.size:
            text_vect_avg.append(list(v.mean(axis=0)))
        else:
            text_vect_avg.append(list(np.zeros(vector_size_n_w2v, dtype=float))) # the same vector size must be used here as for model training

    # TODO: for troubleshotting (delete later)
    # #**#print("\n\n\n\n\n\n**********************************  WORD2VEC-NEW  ***************************************\n\n")
    # for t in text_vect_avg[0:10]:
    #     #**#print(t)
    # #**#print("\n\n**********************************  WORD2VEC-NEW  ***************************************\n\n\n\n\n")
    # NOTE: could move forward and train model based on siRNA efficacy, but not doing that here (just using word2vec to represent/embed data)

    return text_vect_avg


def embed_sequences_with_gensim_word2vec_cbow(seq_ls, kmer_size_, window_size_, word_freq_cutoff_):
    ''' NEW UPDATED WORD2VEC METHOD'''
    ## UPDATED to use Word2Vec correctly - using CBOW (Continuous Bag Of Words): Using the context to predict a target word
    ## https://michael-fuchs-python.netlify.app/2021/09/01/nlp-word-embedding-with-gensim-for-text-classification/#gensim---word2vec
    from gensim.models import Word2Vec

    vector_size_n_w2v = 120 # TODO: update to something higher(?) 100?

    w2v_model = Word2Vec(vector_size=vector_size_n_w2v,
                         window=5, # number of words to consider on each side for the context TODO: update to larger value?
                         min_count=word_freq_cutoff_,
                         sg=0 ) # 0=CBOW, 1=Skip-gram

    texts_sirna = [get_kmers_(x, kmer_size_, window_size_) for x in seq_ls]

    w2v_model.build_vocab(texts_sirna) # Create the vocabulary, which is to be learned by Word2Vec
    ##**#print(w2v_model)

    w2v_model.train(texts_sirna, # Train Neural Network over 5 epochs (NOTE: this can take some time)
                    total_examples=w2v_model.corpus_count,
                    epochs=50) # TODO: update epochs?

    # Save to Word2Vec Model (and the vector size
    w2v_model.save("embedding_dictionaries/word2vec_model")
    import pickle as pk
    pk.dump(vector_size_n_w2v, open('embedding_dictionaries/vector_size_w2v_metric.pkl', 'wb'))

    # # EXAMPLE: Output of the calculated vector for a given word (kmer) from the vocabulary:
    # single_example_kmer__ = texts_sirna[0][0]
    # w2v_model.wv[single_example_kmer__]
    # # EXAMPLE: Display the words that are most similar to a given word from the vocabulary:
    # w2v_model.wv.most_similar(single_example_kmer__)

    # Generate aggregate sentence vectors based on the kmer vectors for each kmer in the given sequence
    words = set(w2v_model.wv.index_to_key)
    import numpy as np
    text_vect_ls = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                                for ls in texts_sirna ])

    # NOTE: since specify vector initially no need to pad them (since all same length)
    # Generation of averaged Sentence Vectors
    text_vect_avg = []
    for v in text_vect_ls:
        if v.size:
            text_vect_avg.append(list(v.mean(axis=0)))
        else:
            text_vect_avg.append(list(np.zeros(vector_size_n_w2v, dtype=float))) # the same vector size must be used here as for model training

    # TODO: for troubleshotting (delete later)
    # #**#print("\n\n\n\n\n\n**********************************  WORD2VEC-NEW  ***************************************\n\n")
    # for t in text_vect_avg[0:10]:
    #     #**#print(t)
    # #**#print("\n\n**********************************  WORD2VEC-NEW  ***************************************\n\n\n\n\n")
    # NOTE: could move forward and train model based on siRNA efficacy, but not doing that here (just using word2vec to represent/embed data)

    return text_vect_avg




def embed_sequences_with_fasttext_cbow(seq_ls, kmer_size_, window_size_, word_freq_cutoff_):
    # https://fasttext.cc/docs/en/python-module.html
    import fasttext

    texts_sirna = [get_kmers_(x, kmer_size_, window_size_) for x in seq_ls]

    # export texts to file so fasttext.train_unsupervised can use
    temp_texts_sirna_file = 'embedding_dictionaries/fasttext_text_cbow_temp.txt'
    with open(temp_texts_sirna_file, 'w+') as f:
        for t in texts_sirna:
            f.write(' '.join(t) + '\n')
    f.close()


    # Train Fasttext Model:
    # CBOW model :
    model = fasttext.train_unsupervised(input =  temp_texts_sirna_file, # training file path (required)
                                        model = 'cbow', # unsupervised fasttext model {cbow, skipgram}
                                        lr = 0.05, # learning rate [0.05]
                                        dim = 120, # size of word vectors [100]
                                        ws = 15, # size of the context window [5]
                                        epoch = 15, # number of epochs [5]
                                        minCount = 1,# minimal number of word occurences [5]
                                        minn = kmer_size_,  # min length of char ngram [3]
                                        maxn = kmer_size_, # max length of char ngram [6]
                                        # neg # number of negatives sampled [5]
                                        # wordNgrams # max length of word ngram [1]
                                        # loss # loss function {ns, hs, softmax, ova} [ns]
                                        # bucket # number of buckets [2000000]
                                        # thread # number of threads [number of cpus]
                                        # lrUpdateRate # change the rate of updates for the learning rate [100]
                                        # t # sampling threshold [0.0001]
                                        # verbose # verbose [2]
                                        )

    # get the vectors for each sequence
    seq_vects_ls = [list(model[s]) for s in seq_ls]


    return seq_vects_ls


def embed_sequences_with_fasttext_skipgram(seq_ls, kmer_size_, window_size_, word_freq_cutoff_):
    # https://fasttext.cc/docs/en/python-module.html
    import fasttext

    texts_sirna = [get_kmers_(x, kmer_size_, window_size_) for x in seq_ls]

    # export texts to file so fasttext.train_unsupervised can use
    temp_texts_sirna_file = 'embedding_dictionaries/fasttext_text_skipgram_temp.txt'
    with open(temp_texts_sirna_file, 'w+') as f:
        for t in texts_sirna:
            f.write(' '.join(t)+'\n')
    f.close()

    # Train Fasttext Model:
    # Skipgram model :
    model = fasttext.train_unsupervised(
                                        input =  temp_texts_sirna_file, # training file path (required)
                                        model = 'skipgram', # unsupervised fasttext model {cbow, skipgram} [skipgram]
                                        lr = 0.05, # learning rate [0.05]
                                        dim = 120, # size of word vectors [100]
                                        ws = 15, # size of the context window [5]
                                        epoch = 15, # number of epochs [5]
                                        minCount = 1, # minimal number of word occurences [5]
                                        minn = kmer_size_, # min length of char ngram [3]
                                        maxn  = kmer_size_, # max length of char ngram [6]
                                        # neg # number of negatives sampled [5]
                                        # wordNgrams # max length of word ngram [1]
                                        # loss # loss function {ns, hs, softmax, ova} [ns]
                                        # bucket # number of buckets [2000000]
                                        # thread # number of threads [number of cpus]
                                        # lrUpdateRate # change the rate of updates for the learning rate [100]
                                        # t # sampling threshold [0.0001]
                                        # verbose # verbose [2]
                                        )

    # get the vectors for each sequence
    seq_vects_ls = [ list(model[s]) for s in seq_ls]

    return seq_vects_ls


def embed_sequences_with_fasttext_class_trained(seq_ls, kmer_size_, window_size_, word_freq_cutoff_,
                                       data_classes_, indxs_ext_data_):
    # https://fasttext.cc/docs/en/python-module.html
    # For utilizing class data, exclude indxs_ext_data_
    seq_ls_no_ext = []
    seq_ls_ext = []
    data_classes_no_ext = []
    for i in list(range(len(indxs_ext_data_))):
        if indxs_ext_data_[i] == True:
            seq_ls_ext.append(seq_ls[i])
        else:
            seq_ls_no_ext.append(seq_ls[i])
            data_classes_no_ext.append(data_classes_[i])

    import fasttext

    texts_sirna = [get_kmers_(x, kmer_size_, window_size_) for x in seq_ls_no_ext]

    # export texts to file so fasttext.train_unsupervised can use
    temp_texts_sirna_file = 'embedding_dictionaries/fasttext_text_cbow_temp_no_ext_labeled.txt'
    with open(temp_texts_sirna_file, 'w+') as f:
        for t, c in zip(texts_sirna, data_classes_no_ext):
            f.write('__label__'+str(c) + ' '+ (' '.join(t)) + '\n')
    f.close()


    # Train Fasttext Model:
    # CBOW model :
    model = fasttext.train_supervised(input =  temp_texts_sirna_file, # training file path (required)
                                        lr = 0.05, # learning rate [0.05]
                                        dim = 120, # size of word vectors [100]
                                        ws = 15, # size of the context window [5]
                                        epoch = 15, # number of epochs [5]
                                        minCount = 1,# minimal number of word occurences [5]
                                        minn = kmer_size_,  # min length of char ngram [3]
                                        maxn = kmer_size_, # max length of char ngram [6]
                                        # neg # number of negatives sampled [5]
                                        # wordNgrams # max length of word ngram [1]
                                        # loss # loss function {ns, hs, softmax, ova} [ns]
                                        # bucket # number of buckets [2000000]
                                        # thread # number of threads [number of cpus]
                                        # lrUpdateRate # change the rate of updates for the learning rate [100]
                                        # t # sampling threshold [0.0001]
                                        # verbose # verbose [2]
                                        )

    # get the vectors for each sequence
    seq_vects_ls = [list(model[s]) for s in seq_ls]


    return seq_vects_ls



def embed_sequences_with_keras_new(seq_ls, kmer_size_, window_size_, word_freq_cutoff_,
                                       data_classes_, indxs_ext_data_):
    from keras.models import Sequential
    from keras import layers

    embedding_dim = 50

    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size,
                               output_dim=embedding_dim,
                               input_length=maxlen))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()



# def embed_sequences_with_gensim_word2vec_new_class_trained(seq_ls, kmer_size_, window_size_, word_freq_cutoff_,
#                                              data_numeric_classes_, indxs_ext_data_):
#     ''' NEW UPDATED WORD2VEC METHOD'''
#
#     # For utilizing class data, exclude indxs_ext_data_
#     seq_ls_no_ext = []
#     seq_ls_ext = []
#     data_numeric_classes_no_ext = []
#     for i in list(range(len(indxs_ext_data_))):
#         if indxs_ext_data_[i] == True:
#             seq_ls_ext.append(seq_ls[i])
#         else:
#             seq_ls_no_ext.append(seq_ls[i])
#             data_numeric_classes_no_ext.append(data_numeric_classes_[i])
#
#     ## UPDATED to use Word2Vec correctly - using CBOW (Common Bag Of Words): Using the context to predict a target word
#     ## https://michael-fuchs-python.netlify.app/2021/09/01/nlp-word-embedding-with-gensim-for-text-classification/#gensim---word2vec
#     from gensim.models import Word2Vec
#
#     vector_size_n_w2v = 120 # TODO: update to something higher(?) 100?
#
#     w2v_model = Word2Vec(vector_size=vector_size_n_w2v,
#                          window=5, # number of words to consider on each side for the context TODO: update to larger value?
#                          min_count=word_freq_cutoff_,
#                          sg=1)  # 0=CBOW, 1=Skip-gram
#
#     texts_sirna = [get_kmers_(x, kmer_size_, window_size_) for x in seq_ls_no_ext] # seq_ls]
#
#     w2v_model.build_vocab(texts_sirna) # Create the vocabulary, which is to be learned by Word2Vec
#     ##**#print(w2v_model)
#
#     w2v_model.train(texts_sirna, # Train Neural Network over 5 epochs (NOTE: this can take some time)
#                     total_examples=w2v_model.corpus_count,
#                     epochs=50) # TODO: update epochs?
#
#     # Save to Word2Vec Model (and the vector size
#     w2v_model.save("embedding_dictionaries/word2vec_model")
#     import pickle as pk
#     pk.dump(vector_size_n_w2v, open('embedding_dictionaries/vector_size_w2v_metric.pkl', 'wb'))
#
#     # # EXAMPLE: Output of the calculated vector for a given word (kmer) from the vocabulary:
#     # single_example_kmer__ = texts_sirna[0][0]
#     # w2v_model.wv[single_example_kmer__]
#     # # EXAMPLE: Display the words that are most similar to a given word from the vocabulary:
#     # w2v_model.wv.most_similar(single_example_kmer__)
#
#     # Generate aggregate sentence vectors based on the kmer vectors for each kmer in the given sequence
#     words = set(w2v_model.wv.index_to_key)
#     import numpy as np
#     text_vect_ls = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
#                                 for ls in texts_sirna ])
#
#     # Generation of averaged Sentence Vectors
#     text_vect_avg = []
#     for v in text_vect_ls:
#         if v.size:
#             text_vect_avg.append(list(v.mean(axis=0)))
#         else:
#             text_vect_avg.append(list(np.zeros(vector_size_n_w2v, dtype=float))) # the same vector size must be used here as for model training
#
#     # # TODO: for troubleshotting (delete later)
#     # #**#print("\n\n\n\n\n\n**********************************  WORD2VEC-NEW  ***************************************\n\n")
#     # for t in text_vect_avg[0:10]:
#     #     #**#print(t)
#     # #**#print("\n\n**********************************  WORD2VEC-NEW  ***************************************\n\n\n\n\n")
#
#
#     # Add efficiacy data and train model
#     from sklearn.svm import SVC
#     clf_w2v = SVC(kernel='linear')
#     clf_w2v.fit(text_vect_avg, data_numeric_classes_no_ext)
#
#     pk.dump(clf, open('embedding_dictionaries/clf_model.pkl', 'wb'))
#
#
#     texts_ext = [get_kmers_(x, kmer_size_, window_size_) for x in seq_ls_ext]
#     words = set(w2v_model.wv.index_to_key)
#     text_vect_ext_ls = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
#                                                for ls in texts_ext])
#
#     text_vect_ext_avg = []
#     for v in text_vect_ext_ls:
#         if v.size:
#             text_vect_ext_avg.append(v.mean(axis=0))
#         else:
#             text_vect_ext_avg.append(np.zeros(vector_size_n_w2v, dtype=float))  # the same vector size must be used here as for model training
#
#     return text_vect_avg



def embed_sequences_with_gensim_word2vec (seq_ls, kmer_size_, window_size_, word_freq_cutoff_):
    '''
        ##
        ##      ~*~ Performs Sequence Data Word2Vec Deep Embedding using Gensim  ~*~
        ##
        ## Takes a list of sequences (seq_ls) as input and returns a list of encoded sequences
        ##    - Encodes sequence data as kmer integer words with sliding window
        ##
        ## word_freq_cutoff_ -->  Number of times a word must occur in the Bag-of-words Corpus
        ##      * when word_freq_cutoff = 1 only include words that occur more than once # TODO: adjust this parameter
        ##
        ## https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#sphx-glr-auto-examples-tutorials-run-word2vec-py
        ##
        #########################################################################################
        '''

    ## BELOW IS OLD METHOD:

    import numpy as np
    import pandas as pd

    texts_sirna = [get_kmers_(x,kmer_size_,window_size_) for x in seq_ls]

    pos_ = -2
    len_ = 14

    # Count word frequencies
    from collections import defaultdict
    kmer_freq = defaultdict(int)
    for text in texts_sirna:
        for token in text:
            kmer_freq[token] += 1



    max_freq_ = 0
    kmer_with_max_freq_ = ""
    cts_kmer_freq_0 = 0
    cts_kmer_freq_1 = 0
    cts_all_ls = []
    for k in kmer_freq:
        cts_all_ls.append(kmer_freq[k])
        if kmer_freq[k]>max_freq_:
            max_freq_ = kmer_freq[k]
            kmer_with_max_freq_ = k
        if kmer_freq[k] == 0:
            cts_kmer_freq_0+=1
        elif kmer_freq[k] == 1:
            cts_kmer_freq_1+=1
    import pandas as pd
    cts_all_ls = pd.Series(cts_all_ls)


    processed_corpus_sirna = [[token for token in text if kmer_freq[token] > word_freq_cutoff_] for text in texts_sirna]

    # Pad corpus
    rln_ = max([len(x) for x in processed_corpus_sirna])
    processed_corpus_sirna = [pad_corpus_(x, rln_) for x in processed_corpus_sirna]

    from gensim.models import Word2Vec

    # Build Dictionary of K-mers
    from gensim import corpora

    dictionary_sirna = corpora.Dictionary(processed_corpus_sirna)

    w2v_model = Word2Vec(sentences=processed_corpus_sirna, vector_size=20, window=window_size_, min_count=1, workers=4)
    #w2v_model = Word2Vec(sentences=processed_corpus_sirna, vector_size=100, window=7, min_count=1, workers=4)

    ## Vectorize Corpus
    # Using Bag-of-words representation
    #**NOTE**: it is possible for two unique seequences to have **IDENTICAL vectorization**
    # bow_corpus_sirna = [dictionary_sirna.doc2bow(seq) for seq in processed_corpus_sirna]

    w2v_model.train(processed_corpus_sirna, total_examples=len(seq_ls), epochs=1)

    #**#print("\n\n w2v_model shape:",w2v_model.wv[processed_corpus_sirna[0]].shape)
    #**#print("\n\n w2v_model shape:", w2v_model.wv[processed_corpus_sirna[-2]].shape)
    #**#print("\n\n w2v_model shape:", w2v_model.wv[processed_corpus_sirna[-1]].shape)
    #**#print("corpus size:",len(processed_corpus_sirna))

    # flatten vector 
    ls_ = [list(w2v_model.wv[ pc_ ].flatten()) for pc_ in processed_corpus_sirna]

    # Pad vector
    rln_ =max([len(x) for x in ls_])
    ls_padded_ = [pad_list_(x,rln_) for x in ls_]

    #ls_padded_ = [[(x - mn_) / (mx_ - mn_) for x in y ] for y in ls_padded_]

    #ls_padded_ = [[np.round(x,4) for x in y ] for y in ls_padded_]

    # Check that the length of a single vector isn't longer than what can be stored in a csv file
    if len(str(ls_padded_[0][0])) >= 32000:
        raise Exception('ERROR: exported vector for gensim word2vec embeddings is longer than 32000, update embedding parameters to shorten vector sizes')
    #**#print("Word2vec embedding complete!")
    return ls_padded_

        
    
    
    



    
    
def one_hot_encode_sequences(seq_ls):
    '''
    # Takes as input a list of sequences (seq_ls) and returns a flattened list of one-hot encoded sequences
    '''
    import numpy as np
    
    one_hot_dict = {
                'A':[1,0,0,0],
                'U':[0,1,0,0],'T':[0,1,0,0],
                'C':[0,0,1,0],
                'G':[0,0,0,1],
                'X':[0,0,0,0]
               }
    
    #encoded_seq_ls = [np.array( [item for sublist in [one_hot_dict[x] for x in list(seq)] for item in sublist] ) for seq in seq_ls]
    encoded_seq_ls = [list([item for sublist in [one_hot_dict[x] for x in list(seq)] for item in sublist]) for seq in seq_ls]
    # Check that the length of a single vector isn't longer than what can be stored in a csv file
    if len(str(encoded_seq_ls[0][0])) >= 32000:
        raise Exception('ERROR: exported vector for one-hot-encoding embeddings is longer than 32000, update embedding parameters to shorten vector sizes')
    #**#print('One-Hot Encoding Complete!')
    return encoded_seq_ls
    
    





#####################################################################################################################################################################################
##
##           *~* EXAMPLE RUNNING EMBEDDING METHODS *~*
##
#####################################################################################################################################################################################
##
## s = ['AAGAAGAAAACTCAACTCAGTGCCATTTTACGAATATATGCGTTTATATTTATACTTCCT', 'TCTCCACAGCCTGAAGAATGAAGACACGACAGAATAAAGACTCGATGTCAATGAGGAGTG', 'AAAGGAGAAAAAATACAATTTCTCACTTTGCATTTAGTCAAAAGAAAAAATGCTTTATAG', 'TCCCTACATGGAGTATATGTCAAGCCATAATTGTTCTTAGTTTGCAGTTACACTAAAAGG', 'TTACTATCTGTGGTTACGGTGGAGACATTGACATTATTACTGGAGTCAAGCCCTTATAAG', 'CTAAAGTTAGAAAGTTGATTTTAAGAATCCAAACGTTAAGAATTGTTAAAGGCTATGATT', 'TGCTAGGGAAGGCGGGAACCTTGGGTTGAGTAATGCTCGTCTGTGTGTTTTAGTTTCATC', 'AAAGGAATGTTTTGAAAGCCTCAGTAAAGAATGCGCTATGTTCTATTCCATCCGGAAGCA', 'CCAAAGAAGAAACCACTGGATGGAGAATATTTCACCCTTCAGATGCTACTTGACTTACGA', 'CAAAAGCGGACAAGGCCCGTTATGAAAGAGAAATGAAAACCTATATCCCTCCCAAAGGGG']
##
#### Bag-of-Words Embedding:
##
## from embedding_methods_code.embedding_methods import embed_sequences_with_gensim
## embed_sequences_with_gensim( s, kmer_size_ = 3, window_size_ = 1, word_freq_cutoff_ = 1)
##
##
##
#### Word Embedding using CNN
##
## from embedding_methods_code.embedding_methods import embed_sequences_with_ann
## embed_sequences_with_ann( s, kmer_size_ = 3, window_size_ = 1)
## 
## 
## 
#### One-Hot Encoding
## 
## from embedding_methods_code.embedding_methods import one_hot_encode_sequences
## one_hot_encode_sequences( s)
## 
          
