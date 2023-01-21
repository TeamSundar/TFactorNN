# Import package
import numpy as np
from numpy import random
import torch, os
from tqdm import tqdm
from tqdm import trange 
from multiprocessing import Pool
import pandas as pd
from transformers import BertModel, BertConfig, DNATokenizer
from Bio import SeqIO
from sklearn.model_selection import train_test_split as tts

ROOT = '/home/dell15/KING/Projects/202109_tfbind3/'
outPath = '/mnt/media/tfbind3/training_data_random/'

#seq_df = pd.read_csv(data, header=None)

print(torch.cuda.get_device_name(0))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Lines = seq_df[0]
dir_to_pretrained_model = ROOT+'DNABERT/6-new-12w-0/'
config = BertConfig.from_pretrained('https://raw.githubusercontent.com/jerryji1993/DNABERT/master/src/transformers/dnabert-config/bert-config-6/config.json')
tokenizer = DNATokenizer.from_pretrained('dna6')
model = BertModel.from_pretrained(dir_to_pretrained_model, config=config)
model.to(device)

def getEmb(seq, n):
    seq_ = ' '.join([seq[i:i+n] for i in range(0, len(seq), n)])
    model_input = tokenizer.encode_plus(seq_, add_special_tokens=False, max_length=512)["input_ids"]
    model_input = torch.tensor(model_input, dtype=torch.long).to(device)
    model_input = model_input.unsqueeze(0)   # to generate a fake batch with batch size one
    output = model(model_input)
    return output[1]   

def onehotEncode(data):
    # define universe of possible input values
    bases = 'ATGC'
    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(bases))
    int_to_char = dict((i, c) for i, c in enumerate(bases))
    # integer encode input data
    integer_encoded = [char_to_int[char] for char in data]
    # one hot encode
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(bases))]
        letter[value] = 1
        onehot_encoded.append(letter)
    #return list(map(list, zip(*onehot_encoded)))
    return(onehot_encoded)

# Encode data
def encode(record):
    count = 0
    a, bert = [], []
    for i in tqdm(record):
        if count>=10000:
            break
        data = str(i.seq).upper()
        try:
            bert_ = getEmb((data), 6)[0].detach().cpu().numpy().tolist()
            a_ = onehotEncode(data)
        except:
            continue
        bert.append(bert_)
        a.append(a_)
        count+=1
    #print('ENCODE:',len(a), len(bert))
    return a, bert

tfs = ['JUN']
#tfs = ['CEBPB']
for TF in tfs:
    print('************ Running for %s ************\n'%(TF))
    count = 0
    X1, X2, y = [], [], []
    if not os.path.exists('/mnt/media/tfbind3/training_v4/'):
        os.makedirs('/mnt/media/tfbind3/training_v4/')
    for i in os.listdir('../data/15tf/15tf_fasta_1000_pos/'+TF+'-human/'):
        record1 = SeqIO.parse("../data/15tf/15tf_fasta_1000_pos/"+TF+"-human/"+i, "fasta")
        record2 = SeqIO.parse("../data/15tf/15tf_fasta_1000_neg_with_motif/"+TF+"-human/"+i, "fasta")

        print('Running for', i)
        # Get X1, X2 positive
        p, p_bert= encode(record1)
        #p = encode(record1)
        X1.extend(p)
        X2.extend(p_bert)
        y_pos = np.ones(len(p),dtype=int)
        y.extend(y_pos)

        if len(y_pos)<3500:
            print('Skipping iteration. Data imbalance.\n')
            continue
        
        # Get X1 negative
        #n = encode(record2)
        n, n_bert= encode(record2)
        X1.extend(n)
        X2.extend(n_bert)
        y_neg = np.zeros(len(n),dtype=int)
        y.extend(y_neg)
        
        count = count+len(p)+len(n)
        print('Batch:',len(p),len(n),len(p)+len(n))
        print('Net Count:', count)
        print()
        if count>190000:
            break

    X1 = np.array(X1)
    X2 = np.array(X2)
    y =  np.array(y)
    print('Data shape:', X1.shape, X2.shape, y.shape)

    # TRAINTEST SPLIT
    print('\nSplitting data')
    X1_train, X1_test, X2_train, X2_test, y_train, y_test = tts(X1, X2, y, test_size=0.33, random_state=12)
    np.save('/mnt/media/tfbind3/training_v4/x1_train_'+TF+'.npy', X1_train)
    np.save('/mnt/media/tfbind3/training_v4/x1_test_'+TF+'.npy', X1_test)
    np.save('/mnt/media/tfbind3/training_v4/x2_train_'+TF+'.npy', X2_train)
    np.save('/mnt/media/tfbind3/training_v4/x2_test_'+TF+'.npy', X2_test)
    np.save('/mnt/media/tfbind3/training_v4/y_train_'+TF+'.npy', y_train)
    np.save('/mnt/media/tfbind3/training_v4/y_test_'+TF+'.npy', y_test)
    print(X1_train.shape, X1_test.shape, y_train.shape, y_test.shape)

    del X1, X2, y, X1_train, X1_test, X2_train, X2_test, y_train, y_test