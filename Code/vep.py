import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import sklearn
import pickle
import requests
import seaborn as sns
import keras

# Get a one-hot encoding for amino acids
aa_string = 'ACDEFGHIKLMNPQRSTVWY'
reshape_aa_list = np.array([a for a in aa_string]).reshape(-1,1)

# Call and fit sklearn's one hot encoder on our possible amino acids library
ohe_aa_fx = sklearn.preprocessing.OneHotEncoder()
ohe_aa_fx = ohe_aa_fx.fit(reshape_aa_list)

# Call and fit sklearn's one hot encoder for our possible labels
possible_labels = np.array(['pathogenic','benign']).reshape(-1,1)
ohe_label_fx = sklearn.preprocessing.OneHotEncoder()
ohe_label_fx = ohe_label_fx.fit(possible_labels)

pathogenic_index = list(ohe_label_fx.categories_[0]).index('pathogenic')

# Class to handle predicting all possible single-site substitutions for
# a given protein, e.g. KCNQ1 in the demo notebook
class Protein():
    
    # List of amino acids
    aa = aa_string
    
    # Initializing
    X_predict = None
    Y_predict = None
    
    def __init__(self,uniprot_id):
        
        # Required: pass in Uniprot identifier name
        self.uniprot_id = uniprot_id
        
        # Query Uniprot through its API to get the sequence
        url = requests.get('https://rest.uniprot.org/uniprotkb/{}.fasta'.format(uniprot_id))
        if not url:
            raise Exception('Error: Could not retrieve sequence from Uniprot. Please check if this is a valid identifier.')
         
         # Format output from Uniprot API to get the sequence   
        fa = url.content.decode('utf-8')
        self.sequence = ''.join(fa.split('\n')[1:])
        
        self.length = len(self.sequence)
    
    # Function to generate all possible single amino acid substitutions of this 
    # protein and test them given a VEP_Model() instance
    def test_all_single_aa_subs(self,vep_nn,return_out=False):
        
        N = len(self.sequence)
        n_aa = len(self.aa)
        
        # Our input vector is sequence length * number of amino 
        # acids - note that this includes the wildtype (so a value for no change)
        # but we'll take that out later
        n_test = N*n_aa
        
        muts = []
        wts = []
        pos = []
        seq = [self.sequence for i in range(n_test)]
        lengths = [self.length for i in range(n_test)]
        
        # Generate all amino acid substitutions, skipping entries 
        # where wildtype = mutant
        for i in range(N):
            for aa_char in self.aa:
                muts.append(aa_char)
                wts.append(self.sequence[i])
                pos.append(i+1)
        
        # Store in dataframe analogous to X_train in VEP_Model()
        df_test = pd.DataFrame({'uniprot':self.uniprot_id,'mutant_wt':wts,
                                 'mutant_mut':muts,'mutant_pos':pos,
                                 'Sequence':seq,'Length':lengths})
        
        # Predict values of single site substitutions
        x_mat,x_df = vep_nn.process_input(df_test,is_file=False)
        self.X_predict = x_mat
        
        self.Y_predict = vep_nn.nn_model.predict(self.X_predict)
        
        # I have some weirdness with my input dimensions - hack for now
        output_shape = self.Y_predict.shape
        if len(output_shape)==3:
            df_test['pred_pathogenic'] = [self.Y_predict[i][0][pathogenic_index] for i in range(n_test)]
        else:
            df_test['pred_pathogenic'] = [self.Y_predict[i][pathogenic_index] for i in range(n_test)]
        
        self.predictions = df_test
        
        if return_out:
            return X_predict,Y_predict
    
    # Function to plot a heatmap of predicted pathogenicity    
    def plot_heatmap_pathogenicity(self,set_wt_to_zero=True):
        
        df_test = self.predictions[['mutant_mut','mutant_pos','pred_pathogenic']].copy()
        
        # Optionally censor out positions where we're not making an amino acid
        # change - e.g., if the wildtype amino acid = mutant amino acid
        if set_wt_to_zero:
            
            df_test[df_test[mutant_wt == mutant_mut]] = np.nan
        
        
        # Reshape our array to have amino acids on  y axis and position on x   
        df_test = df_test.pivot(index='mutant_mut',columns='mutant_pos',values='pred_pathogenic')
        
        # Finally, plot the heatmap!
        plt.figure(figsize=(15,10))
        sns.heatmap(df_test,cmap='coolwarm',yticklabels=[a for a in self.aa])
        plt.yticks(rotation=0)
        
        plt.show()
    

# Parent class for our 3 types of neural net architectures
class VEP_Model():
    
    # Initialize attributes 
    # A Keras neural net architecture and model
    nn_model = False
    
    # The training history associated with the Keras model
    nn_history = False
    
    # Which accuracy metric to use by default
    accuracy_metric = 'categorical_accuracy'
    
    # One-hot encoding functions for amino acid and for label
    ohe_aa = ohe_aa_fx
    ohe_label = ohe_label_fx
    
    # String of possible amino acids
    aa = aa_string
    
    # Can be overwritten by children classes
    def __init__(self):
        
        pass
        
    # Function to return a one hot encoding given our amino acid character
    def aa_one_hot_encoding(self,aa_char,ohe_function=None):
        
        if ohe_function is None:
            ohe_function = self.ohe_aa
            
        reshape_aa = np.array([aa_char]).reshape(1,-1)
        return ohe_function.transform(reshape_aa).toarray()
    
    # I want to be able to remember which entry corresponds to which catogery in 
    # our predictions so print it out :)
    def print_categorical_labels(self,ohe_function=None):
        
        if ohe_function is None:
            ohe_function = self.ohe_label
            
        print('The categories are: {}'.format(ohe_function.categories_))
        
    # Function to return one hot encoding for labels
    def label_one_hot_encoding(self,label_val,ohe_function=None):
    
        if ohe_function is None:
            ohe_function = self.ohe_label
        
        reshape_label = np.array([label_val]).reshape(1,-1)
        return ohe_function.transform(reshape_label).toarray()
          
    # Function to restrict dataset to just mutations that have clear labels 
    # 'pathogenic' or 'benign', and only return columns of interest
    def restrict_pathogenic_or_benign(self,input_file):
        
        # Only keep columns that are directly relevant
        cols_to_keep =  ['uniprot','Sequence','Length','mutant_pos','mutant_wt','mutant_mut','WT_match','Starry_Coarse_Grained_Clin_Sig']
        
        df = pd.read_csv(input_file)
        
        # Only keep rows with either a pathogenic or benign label
        return df.query('(Starry_Coarse_Grained_Clin_Sig == "benign") or (Starry_Coarse_Grained_Clin_Sig=="pathogenic")')[cols_to_keep]
        
    
    # Shuffle labels with respect to inputs and calculate this accuracy (used to 
    # provide a baseline for how well we're doing accuracy-wise). Plot a random
    # distribution to compare against the actual accuracy    
    def plot_shuffled_accuracies(self,test_input,test_out):
        
        _, base_accuracy = self.nn_model.evaluate(test_input,test_out,verbose=0)
        print('Test accuracy is {}'.format(base_accuracy))
        test_shuffle = copy.copy(test_out)
        
        shuffled_vals = []
        for i in range(100):
            
            np.random.shuffle(test_shuffle)
            _,na = self.nn_model.evaluate(test_input,test_shuffle,verbose=0)
            shuffled_vals.append(na)
        
        plt.figure(figsize=(10,10))
        plt.hist(shuffled_vals)
        plt.xlabel('Accuracy for Shuffled Labels Trials')
        plt.ylabel('Counts')
        plt.axvline(x=base_accuracy,label='Actual Accuracy')
        plt.legend(bbox_to_anchor=(1,1),loc='upper left')
        
        
    # Define functions to plot loss given Keras model history
    def plot_accuracy_and_loss(self,trained_history=None):
        
        if trained_history is None:
            trained_history=self.nn_history
        
        # Check to make sure we have a valid NN history.
        if not trained_history:
            print('NN history not loaded in. Please set at nn_history.')
            return False
            
        # Plot loss in the first slot
        loss = trained_history.history['loss']
        val_loss = trained_history.history['val_loss']
        
        # Get epochs for the x-axes
        epochs = range(1, len(loss) + 1)
        
        # Initialize figure
        plt.figure(figsize=(8,4), dpi=100)
        
        ax0 = plt.subplot(1,2,1)
        ax0.plot(epochs, loss, 'bo', label='Training loss')
        ax0.plot(epochs, val_loss, 'b', label='Validation loss')
        ax0.title.set_text('Training and validation loss')
        ax0.set_xlabel('Epochs')
        ax0.set_ylabel('Loss')
        ax0.legend()
          
        # Plot accuracy in the second slot
        acc = trained_history.history[self.accuracy_metric]
        val_acc = trained_history.history['val_'+self.accuracy_metric]
        
        ax1  = plt.subplot(1,2,2)
        ax1.plot(epochs, acc, 'bo', label='Training acc')
        ax1.plot(epochs, val_acc, 'b', label='Validation acc')
        ax1.title.set_text('Training and validation accuracy')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')
        ax1.legend()

        plt.show()
    
    # Placeholder - overwritten by children classes
    def process_input(self,test_file):
        pass
    
    # Placeholder - overwritten by children classes      
    def process_input_output_training(self,train_file,val_file,test_file):
        pass
    
    # Load and store Keras saved model (ie, load a previously trained model)
    def load_nn_model_from_weights(self,weights_file):
        self.nn_model = keras.models.load_model(weights_file)
        
class Model1(VEP_Model):
    
          
    # Return a 40-length tensor corresponding to one hot encodings for wildtype
    # and mutated amino acid in one concatenated vector
    def aa_subs_input(self,wt_aa,subs_aa):
          wt_encoding = self.aa_one_hot_encoding(wt_aa)
          mut_encoding = self.aa_one_hot_encoding(subs_aa)
          return np.concatenate([wt_encoding,mut_encoding],axis=1)
          
    
    # Process one file for input to model
    def process_input(self,train_file,is_file=True):
        
        train_raw = train_file
        
        if is_file:
            
            # For now, we only want to use mutations that have a clear label, pathogenic
            # or benign
            train_raw = self.restrict_pathogenic_or_benign(train_file)
        
        # Format wildtype and mutant amino acids for input to neural net
        X_train = np.array([self.aa_subs_input(w,m) for w,m in zip(train_raw['mutant_wt'],train_raw['mutant_mut'])])
        
        return X_train, train_raw
        
    
    # For our model 1 (just amino acid transitions), process input and output data
    def process_input_output_training(self,train_file,val_file,test_file,return_out=False):
        
        # Process inputs appropriately
        X_train, train_raw = self.process_input(train_file)
        X_val, val_raw = self.process_input(val_file)
        X_test, test_raw = self.process_input(test_file)
    
        # Format labels (pathogenic or benign) for output data 
        Y_train = np.array([self.label_one_hot_encoding(x) for x in train_raw['Starry_Coarse_Grained_Clin_Sig']])
        Y_val = np.array([self.label_one_hot_encoding(x) for x in val_raw['Starry_Coarse_Grained_Clin_Sig']])
        Y_test = np.array([self.label_one_hot_encoding(x) for x in test_raw['Starry_Coarse_Grained_Clin_Sig']])
        
        # Save to our model instance
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        
        self.Y_train = Y_train
        self.Y_val = Y_val
        self.Y_test = Y_test
        
        # Only return output if 'return_output' set to True
        if return_out:
            return X_train, X_val, X_test, Y_train, Y_val, Y_test
        else:
            return None

class Model2(VEP_Model):
    
    # This tells us how much of the surrounding protein sequence 
    # around the mutated position we should look at
    default_window_size = 6
    
    # Set which amino acid property columns we want to work with
    properties_of_interest = ['hydrophobicity','mass','abundance','pI']
    
    def __init__(self,amino_acid_properties_file):
        
        df = pd.read_csv(amino_acid_properties_file)
        
        self.aa_properties = df
        self.aa_properties = self.aa_properties.set_index('aa')
                
    # Return amino acid properties from our amino acid encoding file for
    # a particular character
    def get_amino_acid_properties(self,aa_char,properties = None):
        
        if properties is None:
            properties = self.properties_of_interest
            
        # Find our amino acid of interest in our saved table
        #aa_vals = self.aa_properties.query('aa == @aa_char').iloc[0]
        aa_vals = self.aa_properties.loc[aa_char]
        ret_val = np.array(aa_vals[properties])
        return ret_val
    
    # Sets window size if default is not desired
    def set_window_flanking_size(self,window_size):
        self.default_window_size = window_size
    
    # Function to grab a window of the sequence centered around our amino acid 
    # position that's changing. E.g., for sequence 'MSTLKRST' and position = 3 (zero-
    # indexed), and window size w = 2, I will return a subsequence of 2*w+1 - in 
    # this case, 'STLKR'. If our prescribed window size and position go beyond the
    # boundaries of the sequence (beyond the start or end of the original sequence),
    # we fill in the spaces with the gap '-' character. Note that our given position
    # will always be at the center, with an equal number of amino acids flanking it.
    def seq_with_window(self,seq,pos,window_size=None):
        
        if window_size is None:
            window_size=self.default_window_size

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
    def translate_seq_to_hydro(self,seq):
            
        # Get our features for each amino acid in sequence
        return np.array([self.get_amino_acid_properties(aa) for aa in seq])
  
    # Now, let's get our windowed sequence and translate it into our hydrophobicity
    # vector for each sequence in the train, validate, and test sets
    def get_seq_hydro_array(self,df_val,window_size=None):
        
        if window_size is None:
            window_size = self.default_window_size
        
        # Get all of our sequence windows
        seq_list = [self.seq_with_window(seq,pos,window_size) for seq,pos in zip(df_val['Sequence'],df_val['mutant_pos']-1)]
        
        # Pre-allocate for our vector with features for every window - this might not
        # actually do anything
        size_inputs = self.translate_seq_to_hydro(seq_list[0]).shape
        x = np.zeros((len(seq_list),size_inputs[0],size_inputs[1]))
        
        # Generate a tensor of size (batch_size,total_window_size,number_features)
        x = [self.translate_seq_to_hydro(subseq) for subseq in seq_list]
        
        return np.array(x)
              
    # Return a feature that is similar to a one-hot encoding - but instead of a 1
    # at a particular location, input the hydrophobicity value
    def make_hydrophobicity_and_location_vector(self,aa_char,window_size=None):
        
        if window_size is None:
            window_size = self.default_window_size
            
        # Our 'window size' is how many flank it on one side, so we need to get total 
        # length of window with both left and right flanks
        total_window_size = window_size*2+1
        
        # Initialize matrix with gap values 
        x = np.array([self.translate_seq_to_hydro('-') for i in range(total_window_size)])
        
        # Add in hydrophobicity measure for mutant amino acid
        hydro_mut_val = self.translate_seq_to_hydro(aa_char)[0]
        x[window_size] = hydro_mut_val
        
        return x
        
    # Process inputs one file at a time
    def process_input(self,train_file,window_size=None,scaler=None,is_file=True):
        
        # Load in dataframe if we're passing in file
        train_raw = train_file
        if is_file:
            
            # For now, we only want to use mutations that have a clear label, pathogenic
            # or benign
            train_raw = self.restrict_pathogenic_or_benign(train_file)
            
        if window_size is None:
            window_size = self.default_window_size
        
        # Get our N x (full window size) array for each of train, test, split
        seq_train = self.get_seq_hydro_array(train_raw,window_size)
        
        # Need code to get our second half of features - namely, zeros at 
        # all spots in sequence window except where our mutation is
        def _format_mutant_array(x):
            x2 = np.array([self.make_hydrophobicity_and_location_vector(aa_char,window_size) for aa_char in x['mutant_mut']])
            x2 = x2.reshape(len(x),2*window_size+1,len(self.properties_of_interest))
            return x2
        
        mutant_train = _format_mutant_array(train_raw)
        
        # We are going to concatenate this all into one vector for each example - the
        # sequence encoding + the wildtype hydrophobicity + the mutant hydrophobicity at
        # the end
        X_train = np.concatenate([seq_train,mutant_train],axis=2)
        
        # Now we need to worry about how we're scaling our input data
        # First things first - reformat our input so that we'll be scaling 
        # over each feature
        input_size = X_train.shape
        x = X_train.reshape(input_size[0]*input_size[1],input_size[2])
        
        # Create a new scaler if we're not passing one in
        if scaler is None:
            scaler = sklearn.preprocessing.MinMaxScaler()
            scaler = scaler.fit(x)
            
            self.scaler = scaler
        
        # Perform our scaling    
        X_train = scaler.transform(x).reshape(input_size[0],input_size[1],input_size[2])
        
        return X_train,train_raw
    
    # Our function to process inputs and outputs
    def process_input_and_output_training(self,
        train_file,val_file,test_file,window_size=None,return_out=False):
        
        if window_size is None:
            window_size = self.default_window_size
        
        # Process our input files
        X_train,train_raw = self.process_input(train_file,window_size,scaler=None)
        X_val,val_raw = self.process_input(val_file,window_size,scaler=self.scaler)
        X_test,test_raw = self.process_input(test_file,window_size,scaler=self.scaler)
        
        # And finally, let's encode our output labels
        self.print_categorical_labels()
        Y_train = np.array([self.label_one_hot_encoding(x) for x in train_raw['Starry_Coarse_Grained_Clin_Sig']])
        Y_val = np.array([self.label_one_hot_encoding(x) for x in val_raw['Starry_Coarse_Grained_Clin_Sig']])
        Y_test = np.array([self.label_one_hot_encoding(x) for x in test_raw['Starry_Coarse_Grained_Clin_Sig']])
        
        
        Y_train = Y_train.reshape(len(Y_train),2,1)
        Y_val = Y_val.reshape(len(Y_val),2,1)
        Y_test = Y_test.reshape(len(Y_test),2,1)
    
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
    
        self.Y_train = Y_train
        self.Y_val = Y_val
        self.Y_test = Y_test
    
        if return_out:
            return X_train, X_val, X_test, Y_train, Y_val, Y_test
        else:
            return None
            
            
class Model3(VEP_Model):
    
    # Require user to load in our dictionary of protein encodings
    def __init__(self,protein_encoding_file):
        
        # The protein ID -> 64 encodings mapping was stored as a pickled dictionary
        with open(protein_encoding_file,'rb') as f:
            d_prots = pickle.load(f)
            
            # Overwrite initialization with dictionary of protein encodings
            self.prot_encodings = d_prots
        
    # Process one input file
    def process_input(self,train_file,scaler=None,is_file=True):
        
        # Load in data if a file; otherwise, set the input dataframe as train_raw
        train_raw = train_file
        if is_file:
            
            # For now, we only want to use mutations that have a clear label, pathogenic
            # or benign
            train_raw = self.restrict_pathogenic_or_benign(train_file)
        
        # Get one-hot encoding of our missense mutation amino acid
        train_mut = np.array([self.aa_one_hot_encoding(aa_char) for aa_char in train_raw['mutant_mut']]).reshape(len(train_raw),20)
        
        # Get the normalized position of our missense mutation
        # (position / sequence length)
        train_pos = np.array([train_raw.iloc[i]['mutant_pos']/train_raw.iloc[i]['Length'] for i in range(len(train_raw))]).reshape(len(train_raw),1)
        
        # Get our one-hot encoding for the wildtype amino acid
        train_wt = np.array([self.aa_one_hot_encoding(aa_char) for aa_char in train_raw['mutant_wt']]).reshape(len(train_raw),20)
        
        # Get our 64 encodings per protein
        d_prots = self.prot_encodings
        
        train_enc = np.array([d_prots[uniprot] for uniprot in train_raw['uniprot']])
        
        # Let's combine all of our inputs into one megatensor
        X_train = np.concatenate([train_pos,train_mut,train_wt,train_enc],axis=1)
        n_features = len(X_train[0])
        
        # If we don't pass in a scaler, make one
        if scaler is None:
        
            # Use StandardScaler to scale based on X_train
            scaler = sklearn.preprocessing.StandardScaler()
            scaler = scaler.fit(X_train)
            self.scaler = scaler
        
        X_train = scaler.transform(X_train).reshape(len(train_raw),n_features)
        
        return X_train, train_raw
        
    
    # For our model 3 (the one with the full-length sequence encoding based on 
    # Alley et al 2019),process our input files and return 
    # X and Y train, validate, and test splits
    def process_input_and_output_training(self,train_file,val_file,test_file,return_out=False):
        
        # Process our levels of input
        X_train,train_raw = self.process_input(train_file)
        X_val,val_raw = self.process_input(val_file)
        X_test,test_raw = self.process_input(test_file)

        # Let's encode our output labels
        self.print_categorical_labels()
        Y_train = np.array([self.label_one_hot_encoding(x) for x in train_raw['Starry_Coarse_Grained_Clin_Sig']])
        Y_val = np.array([self.label_one_hot_encoding(x) for x in val_raw['Starry_Coarse_Grained_Clin_Sig']])
        Y_test = np.array([self.label_one_hot_encoding(x) for x in test_raw['Starry_Coarse_Grained_Clin_Sig']])

        Y_train = Y_train.reshape(len(Y_train),2)
        Y_val = Y_val.reshape(len(Y_val),2)
        Y_test = Y_test.reshape(len(Y_test),2)
        
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        
        self.Y_train = Y_train
        self.Y_val = Y_val
        self.Y_test = Y_test
   
        if return_out:
            return X_train, X_val, X_test, Y_train, Y_val, Y_test
        else:
            return None
    
        
    
    
    