import numpy as np
from tqdm import tqdm_notebook

class ELMO_tokenizer(object):
    """Bert tokenizer for raw text"""

    def __init__(self,
               max_seq_length = 256,
                PAD_WORD = '--PAD--'
               ):
        
        
        self.PAD_WORD = PAD_WORD
        self.max_seq_length = max_seq_length
    
    def predict(self, texts):

        split_texts = list(map(lambda x: np.array(x.split())[:self.max_seq_length], texts))
        
        def pad_sequence(sequence):
            return np.concatenate([sequence, np.array([self.PAD_WORD for i in range(self.max_seq_length - len(sequence))])])
        
        tokens = []
        for text in tqdm_notebook(split_texts, desc="Converting examples to tokens"):
            
            tokens.append(pad_sequence(text))
      
        return np.vstack(tokens)