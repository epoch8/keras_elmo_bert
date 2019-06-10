import tensorflow as tf
import tensorflow_hub as hub
import os
import re
import numpy as np
from tqdm import tqdm_notebook
#from tensorflow.keras import backend as K
from keras import backend as K
from keras.layers import Layer


class BertLayer(Layer):
    
    '''BertLayer which support next output_representation param (https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1):
    
    pooled_output: the first CLS token after adding projection layer () with shape [batch_size, 768]. 
    sequence_output: all tokens output with shape [batch_size, max_length, 768].
    
    Custom mean_pooling: mean pooling of all tokens output [batch_size, 768].
    
    
    You can simple fine-tune last n layers in BERT with n_fine_tune_layers parameter. For view trainable parameters call model.trainable_weights after creating model.
    
    '''
    
    def __init__(self, n_fine_tune_layers=10, tf_hub = None, output_representation = 'pooled_output', trainable = False, **kwargs):
        
        self.n_fine_tune_layers = n_fine_tune_layers
        self.is_trainble = trainable
        self.output_size = 768
        self.tf_hub = tf_hub
        self.output_representation = output_representation
        self.supports_masking = True
        
        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.bert = hub.Module(
            self.tf_hub,
            trainable=self.is_trainble,
            name="{}_module".format(self.name)
        )
        
        
        variables = list(self.bert.variable_map.values())
        if self.is_trainble:
            # 1 first remove unused layers
            trainable_vars = [var for var in variables if not "/cls/" in var.name]
            
            
            if self.output_representation == "sequence_output" or self.output_representation == "mean_pooling":
                # 1 first remove unused pooled layers
                trainable_vars = [var for var in trainable_vars if not "/pooler/" in var.name]
                
            # Select how many layers to fine tune
            trainable_vars = trainable_vars[-self.n_fine_tune_layers :]
            
            # Add to trainable weights
            for var in trainable_vars:
                self._trainable_weights.append(var)

            # Add non-trainable weights
            for var in self.bert.variables:
                if var not in self._trainable_weights:
                    self._non_trainable_weights.append(var)
                
        else:
             for var in variables:
                self._non_trainable_weights.append(var)
                

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)
        
        if self.output_representation == "pooled_output":
            pooled = result["pooled_output"]
            
        elif self.output_representation == "mean_pooling":
            result_tmp = result["sequence_output"]
        
            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
            input_mask = tf.cast(input_mask, tf.float32)
            pooled = masked_reduce_mean(result_tmp, input_mask)
            
        elif self.output_representation == "sequence_output":
            
            pooled = result["sequence_output"]
       
        return pooled
    
    def compute_mask(self, inputs, mask=None):
        
        if self.output_representation == 'sequence_output':
            inputs = [K.cast(x, dtype="bool") for x in inputs]
            mask = inputs[1]
            
            return mask
        else:
            return None
        
        
    def compute_output_shape(self, input_shape):
        if self.output_representation == "sequence_output":
            return (input_shape[0][0], input_shape[0][1], self.output_size)
        else:
            return (input_shape[0][0], self.output_size)
