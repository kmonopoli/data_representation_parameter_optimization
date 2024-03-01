


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

    print('BOW embedding using CountVectorizer complete!')
    return out_ls_


def embed_sequences_with_gensim_doc2bow(seq_ls, kmer_size_, window_size_, word_freq_cutoff_, vector_output_ = 'weights'):
    '''
    ##
    ##      ~*~ Performs Sequence Data Embedding as Bag-of-Words using Gensim  ~*~
    ##
    ## Takes a list of sequences (seq_ls) as an input and returns a list of encoded sequences
    ##    - Encodes sequence data as kmer integer words with sliding window
    ## word_freq_cutoff_ -->  Number of times a word must occur in the Bag-of-words Corpus 
    ##      * when word_freq_cutoff = 1 only include words that occur more than once # TODO: adjust this parameter
    ##
    ## https://radimrehurek.com/gensim/auto_examples/core/run_core_concepts.html#core-concepts-vector
    ##
    #########################################################################################
    '''
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




    #########################################################################################
    ##     Transform Corpus Vectorization using tf-idf model                               ##
    ##        TODO: consider other transformations/models see:                             ##
    ##            `sphx_glr_auto_examples_core_run_topics_and_transformations.py`          ##
    ##                https://radimrehurek.com/gensim/auto_examples/core/run_topics_and_transformations.html
    #########################################################################################
    # # **tf-idf model**: transforms vectors from bag-of-words representation to a vector space where the frequency 
    # #                   #counts are weighted according to the relative rarity of each word in the corpus

    from gensim import models
    # train the model on the corpus
    tfidf_sirna = models.TfidfModel(bow_corpus_sirna) 

    # Transform all sequences in Corpus
    # 1) get kmers
    kmers_all = [get_kmers_(x,kmer_size_,window_size_) for x in seq_ls]
    # 2) vectorize
    vects_all = [dictionary_sirna.doc2bow(x) for x in kmers_all]
    # 3) Transform vectorization to tfidf vector space
    tfidf_vecs_all = [tfidf_sirna[x] for x in vects_all]

    # NOTE: vectorized sequences do not have uniform shapes (see below)
    # Get information about vectorized sequence lengths:
    tfidf_vecs_all_lens_ = pd.Series([len(x) for x in tfidf_vecs_all])
    tfidf_vecs_all_lens_df_ = pd.DataFrame(tfidf_vecs_all_lens_.value_counts())
    tfidf_vecs_all_lens_df_.index.name="Length of Vector"
    tfidf_vecs_all_lens_df_.columns = ['Number of Sequences']
    tfidf_vecs_all_lens_df_.reset_index(inplace=True)


    tfidf_vecs_all_lens_df_.sort_values(by=['Length of Vector'],ascending=False,inplace=True)
    tfidf_vecs_all_lens_df_.reset_index(drop=True,inplace=True)


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
    print('BOW embedding using Gensim doc2bow complete!')
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
    from tensorflow.keras import datasets,layers,models
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
    
    ##print(model.summary())

    # Model will take as input a matrix of size (batch, input_length)
    input_array = np.array(i_encoded_seq_ls)
    
    ##print('\nCNN Input matrix shape:',input_array.shape)

    # Transform kmer "sentence" into a dense feature vector matrix using CNN word Embedding layer
    output_array = model.predict(input_array)
    ##print('\nCNN Output matrix shape:',output_array.shape)

    # convert array to form that can be easily added to dataframe
    output_array = [list(x) for x in list(output_array)]
    
    if len(str(output_array[0][0])) >= 32000:
        raise Exception('ERROR: exported vector for keras deep ann embeddings is longer than 32000, update embedding parameters to shorten vector sizes')
    print('Keras Embedding Complete!')
    return output_array







def embed_sequences_with_gensim_word2vec(seq_ls, kmer_size_, window_size_, word_freq_cutoff_):
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

    print("\n\n w2v_model shape:",w2v_model.wv[processed_corpus_sirna[0]].shape)
    print("\n\n w2v_model shape:", w2v_model.wv[processed_corpus_sirna[-2]].shape)
    print("\n\n w2v_model shape:", w2v_model.wv[processed_corpus_sirna[-1]].shape)
    print("corpus size:",len(processed_corpus_sirna))

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
    print("Word2vec embedding complete!")
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
    print('One-Hot Encoding Complete!')
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
          
