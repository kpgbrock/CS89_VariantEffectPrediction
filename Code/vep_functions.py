import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import sklearn

######## Variable definitions ##############

# List out possible amino acids and format them for sklearn
aa_string = 'ACDEFGHIKLMNPQRSTVWY'
reshape_aa_list = np.array([a for a in aa_string]).reshape(-1,1)

# Call and fit sklearn's one hot encoder on our possible amino acids library
ohe_aa = sklearn.preprocessing.OneHotEncoder()
ohe_aa = ohe_aa.fit(reshape_aa_list)

# Call and fit sklearn's one hot encoder for our possible labels
possible_labels = np.array(['pathogenic','benign']).reshape(-1,1)
ohe_label = sklearn.preprocessing.OneHotEncoder()
ohe_label = ohe_label.fit(possible_labels)

# Kyte-Doolittle hydropathy encoding
# Here, we're going to use a biophysical coding for our amino acids. Each of the
# 20 possible amino acids can react more or less favorably to water - this 
# "hydrophobicity" of amino acids is one of the electrostatic forces governing
# how proteins fold in an aqueous solution. For this scale, we're using the 
# Kyte-Doolittle scale, as downloaded from Expasy 
# (https://web.expasy.org/protscale/)
aa_string = 'ACDEFGHIKLMNPQRSTVWY'
hydro_kd = {'A':  1.800,  'R': -4.500,  'N': -3.500,  'D': -3.500,  
                  'C':  2.500,  'Q': -3.500,  'E': -3.500,  'G': -0.400,
                  'H': -3.200,  'I':  4.500,  'L':  3.800,  'K': -3.900,
                  'M':  1.900,  'F':  2.800,  'P': -1.600,  'S': -0.800,  
                  'T': -0.700,  'W': -0.900,  'Y': -1.300,  'V':  4.200
                 }

# There are some redundant assignments in this mapping - I am going to manually
# change them so that the four residues (Q, G, D, N) have slightly different 
# values
hydro_kd['Q'] = hydro_kd['Q']+0.1
hydro_kd['D'] = hydro_kd['D']-0.1
hydro_kd['N'] = hydro_kd['N']-0.2

# For our gap character, make it the mean hydrophobicity value
hydro_kd['-'] = np.mean(list(hydro_kd.values()))

# Default window size for Model 2
default_window_size = 12

############## Functions ####################

# Function to return a one hot encoding given our amino acid character
def aa_one_hot_encoding(aa_char,ohe_function=ohe_aa):
  reshape_aa = np.array([aa_char]).reshape(1,-1)
  return ohe_function.transform(reshape_aa).toarray()
  
# Return a 40-length tensor corresponding to one hot encodings for wildtype
# and mutated amino acid in one concatenated vector
def aa_subs_input(wt_aa,subs_aa):
  wt_encoding = aa_one_hot_encoding(wt_aa)
  mut_encoding = aa_one_hot_encoding(subs_aa)
  return np.concatenate([wt_encoding,mut_encoding],axis=1)

# I want to be able to remember which entry corresponds to which catogery in our predictions
# so print it out :)
def print_categorical_labels(ohe_function=ohe_label):
    print('The categories are: {}'.format(ohe_function.categories_))
    
# Function to return one hot encoding for labels
def label_one_hot_encoding(label_val,ohe_function=ohe_label):
  reshape_label = np.array([label_val]).reshape(1,-1)
  return ohe_function.transform(reshape_label).toarray()

# Function to restrict dataset to just mutations that have clear labels 'pathogenic' or 'benign',
# and only return columns of interest
def restrict_pathogenic_or_benign(input_file):
    # Only keep columns that are directly relevant
    cols_to_keep = ['uniprot','Sequence','Length','mutant_pos','mutant_wt','mutant_mut','WT_match','Starry_Coarse_Grained_Clin_Sig']
    df = pd.read_csv(input_file)
    return df.query('(Starry_Coarse_Grained_Clin_Sig == "benign") or (Starry_Coarse_Grained_Clin_Sig=="pathogenic")')[cols_to_keep]

# Define functions to plot loss given Keras model history
def plot_loss(trained_history,title_val = 'Training and validation loss'):
  loss = trained_history.history['loss']
  val_loss = trained_history.history['val_loss']
  epochs = range(1, len(loss) + 1)

  plt.plot(epochs, loss, 'bo', label='Training loss')
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title(title_val)
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()

  plt.show()

# Define function to plot accuracy given model history
def plot_acc(trained_history,acc_string='accuracy',title_val = 'Training and validation accuracy'):
  acc = trained_history.history[acc_string]
  val_acc = trained_history.history['val_'+acc_string]
  epochs = range(1, len(acc) + 1)

  plt.plot(epochs, acc, 'bo', label='Training acc')
  plt.plot(epochs, val_acc, 'b', label='Validation acc')
  plt.title(title_val)
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()

  plt.show()
  
# Make a function for computing accuracy - the number of labels correctly 
# predicted (where the probability is highest) divided by the number of test 
# observations. This is so I make sure I understand the metric.
def calculate_accuracy_prediction_one_hot(model_val,test_input,test_output):

  predictions = model_val.predict(test_input)
  N = len(predictions)

  max_probs = [np.argmax(predictions[i]) for i in range(N)]
  output_at_max = [test_output[i][0][max_probs[i]] for i in range(N)]

  return np.sum(output_at_max)/N  

# Shuffle labels with respect to inputs and calculate this accuracy (used to 
# provide a baseline for how well we're doing accuracy-wise)  
def calculate_shuffled_labels_accuracy_one_hot(model_val,test_input,test_out):
    # For fun, let's calculate our best model's performance on a randomized set
    test_labels_copy = copy.copy(test_out)

    # Note: this is an in-place shuffle!
    np.random.shuffle(test_labels_copy)
    return calculate_accuracy_prediction_one_hot(model_val,test_input,test_labels_copy)
    
# Function to grab a window of the sequence centered around our amino acid 
# position that's changing. E.g., for sequence 'MSTLKRST' and position = 3 (zero-
# indexed), and window size w = 2, I will return a subsequence of 2*w+1 - in 
# this case, 'STLKR'. If our prescribed window size and position go beyond the
# boundaries of the sequence (beyond the start or end of the original sequence),
# we fill in the spaces with the gap '-' character. Note that our given position
# will always be at the center, with an equal number of amino acids flanking it.
def seq_with_window(seq,pos,window_size=default_window_size):

  # Get our sequence length and our proposed window start and window end. Note
  # that for sequence length, we're going ahead and subtracting one to make our
  # indexing later easier.
  seqlength = len(seq)-1
  n_left = pos-window_size
  n_right = pos+window_size

  # Grab what we can of the left side of the subsequence (before our input
  # position)
  left_seq = seq[max(n_left,0):pos]
  
  # If this goes beyond the start of the original sequence, fill with gaps
  if n_left < 0:
    left_seq = ''.join(['-' for i in range(abs(n_left))]) + left_seq
  
  # Now let's do the same for the right side - only this time, we care about not
  # exceeding the last amino acid in the sequence
  right_seq = seq[pos+1:min(seqlength,n_right)+1]
  if n_right > seqlength:
    right_seq = right_seq + ''.join(['-' for i in range(abs(n_right-seqlength))])
  
  # Return our combined subsequence
  return left_seq + seq[pos] + right_seq
  
# Translate a sequence into a vector of hydrophobicity values
def translate_seq_to_hydro(seq,hydroscale=hydro_kd):
  return np.array([hydroscale[aa] for aa in seq])
  
# Now, let's get our windowed sequence and translate it into our hydrophobicity
# vector for each sequence in the train, validate, and test sets
def get_seq_hydro_array(df_val,window_size=default_window_size):
  x = [translate_seq_to_hydro(seq_with_window(seq,pos,window_size)) for seq,pos in zip(df_val['Sequence'],df_val['mutant_pos']-1)]
  return np.array(x)
  
# Return a feature that is similar to a one-hot encoding - but instead of a 1
# at a particular location, input the hydrophobicity value
def make_hydrophobicity_and_location_vector(aa_char,window_size=default_window_size):
	
	# Our 'window size' is how many flank it on one side, so we need to get total 
	# length of window with both left and right flanks
	total_window_size = window_size*2+1
	
	# Initialize vector
	x = np.zeros(total_window_size)
	
	# Add in hydrophobicity measure for mutant amino acid
	hydro_mut_val = translate_seq_to_hydro(aa_char)[0]
	x[window_size] = hydro_mut_val
	return x
	
# For our model 1 (just amino acid transitions), process input and output data
def model_1_process_input_and_output(train_file,val_file,test_file):
    # For now, we only want to use mutations that have a clear label, pathogenic
    # or benign
    train_raw = restrict_pathogenic_or_benign(train_file)
    val_raw = restrict_pathogenic_or_benign(val_file)
    test_raw = restrict_pathogenic_or_benign(test_file)
    
    # Format wildtype and mutant amino acids for input to neural net
    X_train = np.array([aa_subs_input(w,m) for w,m in zip(train_raw['mutant_wt'],train_raw['mutant_mut'])])
    X_val = np.array([aa_subs_input(w,m) for w,m in zip(val_raw['mutant_wt'],val_raw['mutant_mut'])])
    X_test = np.array([aa_subs_input(w,m) for w,m in zip(test_raw['mutant_wt'],test_raw['mutant_mut'])])
    
    # Format labels (pathogenic or benign) for output data 
    Y_train = np.array([label_one_hot_encoding(x) for x in train_raw['Starry_Coarse_Grained_Clin_Sig']])
    Y_val = np.array([label_one_hot_encoding(x) for x in val_raw['Starry_Coarse_Grained_Clin_Sig']])
    Y_test = np.array([label_one_hot_encoding(x) for x in test_raw['Starry_Coarse_Grained_Clin_Sig']])
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test
    
# For our model 1 (just amino acid transitions), process input and output data
def model_2_process_input_and_output(train_file,val_file,test_file,window_size=default_window_size):
	
	# For now, we only want to use mutations that have a clear label, pathogenic
	# or benign
	train_raw = restrict_pathogenic_or_benign(train_file)
	val_raw = restrict_pathogenic_or_benign(val_file)
	test_raw = restrict_pathogenic_or_benign(test_file)
	
	# Get our N x (full window size) array for each of train, test, split
	seq_train = get_seq_hydro_array(train_raw,window_size)
	seq_val = get_seq_hydro_array(val_raw,window_size)
	seq_test = get_seq_hydro_array(test_raw,window_size)
	
	# Make our second feature vector - if our total sequence window is 'MSTLK', return
	# an array of [0,0,hydrophobicity(T),0,0] - this will give information both about
	# where the mutation is occurring in the sequence (in the center) and what the mutant
	# amino acid is
	mutant_train = np.array([make_hydrophobicity_and_location_vector(aa_char,window_size) for aa_char in train_raw['mutant_mut']])
	mutant_val = np.array([make_hydrophobicity_and_location_vector(aa_char,window_size) for aa_char in val_raw['mutant_mut']])
	mutant_test = np.array([make_hydrophobicity_and_location_vector(aa_char,window_size) for aa_char in test_raw['mutant_mut']])
	
	
	# We are going to concatenate this all into one vector for each example - the
	# sequence encoding + the wildtype hydrophobicity + the mutant hydrophobicity at
	# the end
	X_train = np.stack([seq_train,mutant_train],axis=2)
	X_val = np.stack([seq_val,mutant_val],axis=2)
	X_test = np.stack([seq_test,mutant_test],axis=2)
	
	# And finally, let's encode our output labels
	print_categorical_labels()
	Y_train = np.array([label_one_hot_encoding(x) for x in train_raw['Starry_Coarse_Grained_Clin_Sig']])
	Y_val = np.array([label_one_hot_encoding(x) for x in val_raw['Starry_Coarse_Grained_Clin_Sig']])
	Y_test = np.array([label_one_hot_encoding(x) for x in test_raw['Starry_Coarse_Grained_Clin_Sig']])

	Y_train = Y_train.reshape(len(Y_train),2,1)
	Y_val = Y_val.reshape(len(Y_val),2,1)
	Y_test = Y_test.reshape(len(Y_test),2,1)
	
	return X_train, X_val, X_test, Y_train, Y_val, Y_test
    
# For our model 3 (the one with the full-length sequence encoding based on Alley et al 2019),
# process our input files and return X and Y train, validate, and test splits
def model_3_process_input_and_output(train_file,val_file,test_file,d_proteins):
    # For now, we only want to use mutations that have a clear label, pathogenic
    # or benign
    train_raw = restrict_pathogenic_or_benign(train_file)
    val_raw = restrict_pathogenic_or_benign(val_file)
    test_raw = restrict_pathogenic_or_benign(test_file)

    # Get one-hot encoding of our missense mutation amino acid
    train_mut = np.array([aa_one_hot_encoding(aa_char) for aa_char in train_raw['mutant_mut']]).reshape(len(train_raw),20)
    val_mut = np.array([aa_one_hot_encoding(aa_char) for aa_char in val_raw['mutant_mut']]).reshape(len(val_raw),20)
    test_mut = np.array([aa_one_hot_encoding(aa_char) for aa_char in test_raw['mutant_mut']]).reshape(len(test_raw),20)

    # Get the normalized position of our missense mutation
    # (position / sequence length)
    train_pos = np.array([train_raw.iloc[i]['mutant_pos']/train_raw.iloc[i]['Length'] for i in range(len(train_raw))]).reshape(len(train_raw),1)
    val_pos = np.array([val_raw.iloc[i]['mutant_pos']/val_raw.iloc[i]['Length'] for i in range(len(val_raw))]).reshape(len(val_raw),1)
    test_pos = np.array([test_raw.iloc[i]['mutant_pos']/test_raw.iloc[i]['Length'] for i in range(len(test_raw))]).reshape(len(test_raw),1)
   
    # Get our one-hot encoding for the wildtype amino acid
    train_wt = np.array([aa_one_hot_encoding(aa_char) for aa_char in train_raw['mutant_wt']]).reshape(len(train_raw),20)
    val_wt = np.array([aa_one_hot_encoding(aa_char) for aa_char in val_raw['mutant_wt']]).reshape(len(val_raw),20)
    test_wt = np.array([aa_one_hot_encoding(aa_char) for aa_char in test_raw['mutant_wt']]).reshape(len(test_raw),20)
   
    # Get our 64 encodings per protein
    train_enc = np.array([d_proteins[uniprot] for uniprot in train_raw['uniprot']])
    #scaler = sklearn.preprocessing.StandardScaler()
    #scaler = scaler.fit(train_enc)
    #train_enc = scaler.transform(train_enc).reshape(len(train_raw),64)

    val_enc = np.array([d_proteins[uniprot] for uniprot in val_raw['uniprot']])
    #val_enc = scaler.transform(val_enc).reshape(len(val_raw),64)
    test_enc = np.array([d_proteins[uniprot] for uniprot in test_raw['uniprot']])
    #test_enc = scaler.transform(test_enc).reshape(len(test_raw),64)

    # Let's combine all of our inputs into one megatensor
    X_train = np.concatenate([train_pos,train_mut,train_wt,train_enc],axis=1)
    n_features = len(X_train[0])
    print(n_features)
    scaler = sklearn.preprocessing.StandardScaler()
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train).reshape(len(train_raw),n_features)
   
    X_val = np.concatenate([val_pos,val_mut,val_wt,val_enc],axis=1)
    X_val = scaler.transform(X_val).reshape(len(val_raw),n_features)
    X_test = np.concatenate([test_pos,test_mut,test_wt,test_enc],axis=1)
    X_test = scaler.transform(X_test).reshape(len(test_raw),n_features)

    # Let's encode our output labels
    print_categorical_labels()
    Y_train = np.array([label_one_hot_encoding(x) for x in train_raw['Starry_Coarse_Grained_Clin_Sig']])
    Y_val = np.array([label_one_hot_encoding(x) for x in val_raw['Starry_Coarse_Grained_Clin_Sig']])
    Y_test = np.array([label_one_hot_encoding(x) for x in test_raw['Starry_Coarse_Grained_Clin_Sig']])

    Y_train = Y_train.reshape(len(Y_train),2)
    Y_val = Y_val.reshape(len(Y_val),2)
    Y_test = Y_test.reshape(len(Y_test),2)
   
    return X_train, X_val, X_test, Y_train, Y_val, Y_test

