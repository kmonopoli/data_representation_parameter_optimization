


# Function Definitions
def classify(expr,eff_co,ineff_co):
    if expr<eff_co:
        return 'efficient'
    elif expr>=ineff_co:
        return 'inefficient'
    else:
        return 'undefined'


def classify_no_undefined(expr,eff_co,ineff_co):
    if expr<eff_co:
        return 'efficient'
    elif expr>=ineff_co:
        return 'inefficient'
    else:
        return 'inefficient'

# def get_flank_by_substring(long, short, includes_target_region, flank_len):
#     try:
#         indx = long.replace('T','U').index(short.replace('T','U'))
#         if includes_target_region:
#             return long[((indx-3)-flank_len):((indx-3)+20+flank_len)]
#         else:
#             return long[((indx-3)-flank_len):(indx-3)] + 'X'*20 + long[((indx-3)+20):((indx-3)+20+flank_len)]
#     except:
#         #return False
#         return np.nan

def get_kmers(seq__, kmer_size__, window_size__):
    """ Returns kmers for a SINGLE sequence """
    return [''.join(seq__[i:kmer_size__+i]) for i in range(len(seq__)-kmer_size__+window_size__)]
## Testing:
## get_kmers('TTCAAAAATAGTGACTCAGAAAAGGACAATTCAAAAAGGACATTAC',kmer_size__ = 6,window_size__ = 1)


def one_hot(seq):
    one_hot_dict = {'A':[1,0,0,0],
                'U':[0,1,0,0],'T':[0,1,0,0],
                'C':[0,0,1,0],
                'G':[0,0,0,1],
                'X':[0,0,0,0]
               }
    return np.array([item for sublist in [one_hot_dict[x] for x in list(seq)] for item in sublist]) # flatten
## Testing:
## df['one-hot_encoded_flanking_sequence'] = df[flank_seq_working_key].apply(lambda x: one_hot(str(x)))


def get_20mer_sequence(seq_16mer, flank_seq):
    ix_ = flank_seq.index(seq_16mer)
    ## Get 20mer:
    return flank_seq[ ix_ - 3 : (ix_ + 16 + 1) ]
    
def get_flanking_sequence(seq_16mer, flank_seq, flank_len, include_20mer = True):
    # Uses 16mer "homology region" to find the 20mer
    import numpy as np
    seq_16mer = seq_16mer.replace('U','T')
    flank_seq = flank_seq.replace('U','T')
    try:
        ix_ = flank_seq.index(seq_16mer)
        
        # NOTE: this code gets just the 20mer --> flank_seq[ ix_ - 3 : (ix_ + 16 + 1) ]
        
        ## Get flanks with 20mer:
        if include_20mer:
            try:
                final_seq = flank_seq[ (ix_ - 3) - flank_len : ( ix_ + 16 + 1 )+ flank_len ] # based off 20mer
                if len(final_seq) != (flank_len*2) + 20 :
                    print("WARNING: final sequence did not match expected ("+str((flank_seq*2) + 20)+"nt) length for 16mer:",seq_16mer)                
                return final_seq
            except:
                print("WARNING: could not get flanking sequence for 16mer:",seq_16mer)
                return np.nan
        ## Get flanks WITHOUT 20mer :
        else:
            try: #UPDATE: 0CT-5-2023 Removed 'XXXXXXXXXXXXXXXXXXXX' from 20mer in cases where flanks only (hurts performance of methods that use kmers - possibly because using X's and finding other sequences were similar?)
                #final_seq = flank_seq[ (ix_ - 3 ) - flank_len : ix_ - 3 ] + 'X'*20 + flank_seq[ ix_ + 16 + 1 : (ix_ + 16 + 1) + flank_len ]# based off 20mer
                #if len(final_seq) !=  (flank_len*2) + 20 :
                final_seq = flank_seq[ (ix_ - 3 ) - flank_len : ix_ - 3 ] + flank_seq[ ix_ + 16 + 1 : (ix_ + 16 + 1) + flank_len ]# based off 20mer
                if len(final_seq) !=  (flank_len*2):# + 20 :
                    print("WARNING: final sequence did not match expected ("+str((flank_seq*2) + 20)+"nt) length for 16mer:",seq_16mer)
                return final_seq
            except:
                print("WARNING: could not get flanking sequence for 16mer:",seq_16mer)
                return np.nan
    # 16mer not found in flanking sequence 
    except: 
        print("WARNING: could not get flanking sequence for 16mer:",seq_16mer,'(16mer not found in flanking sequence)')
        return np.nan


def get_20mer_from_16mer(seq_16mer, flank_seq, seq_20mer_from_dataset, mismatch_to_flank ):
    # Uses 16mer "homology region" to find the 20mer
    import numpy as np
    # First check that 16mer can be used to find the 20mer, if not try returning given 20mer from starting dataset (NOTE: this is not done in find flanking region methods because 20mers are defined while flanking regions often are not -- i.e. finding flanks with cross-species reactive siRNAs where there might not be a perfect 20mer match)
    if mismatch_to_flank != '16mer perfect match to target':
        if type(seq_20mer_from_dataset) == str:
            if len(seq_20mer_from_dataset) == 20:
                return seq_20mer_from_dataset
        else:
            print("WARNING: when calling get_20mer_from_16mer() for 16mer ( "+str(seq_16mer)+" ) there was a mismatch_to_flank  so could not compute 20mer with get_20mer_from_16mer(), but '20mer_targeting_region' from starting dataset was: "+str(seq_20mer_from_dataset))
            return np.nan
    else:
        seq_16mer = seq_16mer.replace('U', 'T')
        flank_seq = flank_seq.replace('U', 'T')
        flank_len = 0 # since getting just the 20mer
        try:
            ix_ = flank_seq.index(seq_16mer)
            # NOTE: this code gets just the 20mer --> flank_seq[ ix_ - 3 : (ix_ + 16 + 1) ]
            try:
                final_seq = flank_seq[(ix_ - 3) - flank_len: (ix_ + 16 + 1) + flank_len]  # based off 20mer
                if len(final_seq) != 20:
                    print("WARNING: when calling get_20mer_from_16mer() final sequence did not match expected (20nt) length for 16mer:", seq_16mer)
                    return np.nan
                return final_seq
            except:
                print("WARNING: when calling get_20mer_from_16mer() could not get 20mer sequence for 16mer:", seq_16mer)
                return np.nan
        # 16mer not found in flanking sequence
        except:
            print("WARNING:when calling get_20mer_from_16mer() could not get 20mer sequence for 16mer:", seq_16mer, '(16mer not found in flanking sequence)')
            return np.nan
    
def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

# print(rgb_to_hex(255, 165, 1))

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    import seaborn as sns
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    rgb = colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
    rgb = [round(x*256) for x in list(rgb)]
    
    return rgb_to_hex(rgb[0],rgb[1],rgb[2])
    #'#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

def make_gradient_color_scheme(size,color_ls,plot_pal=True):
    '''
    Output list of hex colors for color scheme also displays a plot of the color pallette 
    Inputs:
        size = number of colors in color scheme
        color_ls = list of colors to be used in making gradient
            * can be any length < size
            * if only one color is given other color in gradient will be set to white)
        plot_pal = boolean if True will plot a palplot
    '''
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    norm = matplotlib.colors.Normalize(-1,1)
    
    if len(color_ls) == 1:
        color_ls.append("#ffffff")
    grad_cols = [[norm(-1.0), color_ls[0]] ] # first already determined
    if len(color_ls)>len(grad_cols)-1:
        i=1
        while i<len(color_ls)-1: 
            grad_cols.append([norm( i*(2/(len(color_ls)-1))+(-1)), color_ls[i]])
            i+=1
    grad_cols.append([norm(1.0), color_ls[-1]]) # last already determined
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", grad_cols)
    hex_cols = []
    for i in range(cmap.N): 
        if i%int(cmap.N/size)==0: # cmp1.N is 256 
            rgb = cmap(i)[:3] # will return rgba, we take only first 3 so we get rgb
            hex_cols.append(matplotlib.colors.rgb2hex(rgb))
    
    if plot_pal:
        sns.palplot(hex_cols)
        print(hex_cols)
    return(hex_cols)    
    
# # Testing:
# # s16_ = 'GGAAGAATCTATATGT' 
# # sflank_ = 'ACAATCTAGGCAAGGAAGTGAGAGCACATCTTGTGGTCTGCTGAGTTAGGAGGGTATGATTAAAAGGTAAAGTCTTATTTCCTAACAGTTTCACTTAATATTTACGGAAGAATCTATATGTAGCCTTTGTAAAGTGTAGGATTGTTATCATTTAAAAACATCATGTACACTTATATTTGTATTGTATACTTGGTAAGATAAAATTCCACAAAGTAGGAATGGGGCC' 
# i_ = 20
# flen_ = 10
# sflank_ = df.iloc[i_]['flanking_sequence_1']
# s16_ = df.iloc[i_]['16mer_complementary_region']
# print(s16_)
# print(df.iloc[i_]['20mer_targeting_region'].replace('U',"T"))

# print(get_flanking_sequence(s16_, sflank_, flen_, False))
# print(get_flanking_sequence(s16_, sflank_, flen_, True))