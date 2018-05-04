import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

from keras.layers import Input, Dense, TimeDistributed, Embedding, concatenate, LSTM, Permute, Bidirectional, RepeatVector, Reshape, merge, Lambda
from keras.models import Model
from keras import backend as K
from keras.regularizers import l1_l2

def recall_score(y_true, y_pred):
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    return 0 if c3 == 0 else c1/c3

def precision_score(y_true, y_pred):
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    return 0 if c3 == 0 else c1/c2

def f1_score(y_true, y_pred):
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    precision = c1 / c2
    recall = 0 if c3 == 0 else c1 / c3
    f1_score = 2 * (precision * recall) / (precision + recall)
    return 0 if c3 ==0 else f1_score

'''
example:
inputs_wide = {'feature_demo':[0,0,1,1,0,0,0,1,0,1],'length':10,'name':'wide_input','wide_output_dim':32,'l1':1e-4,'l2':1e-4}
inputs_deepX = {'deep_hidden_dim':128,'deep_output_dim':32}
input_deep1={'feature_demo':[0,0,1,3,7,11,2,10,6],'length':9,'name':'shop_id_type','embedding_out_dim':16,'embedding_in_dim':20}
input_deep2={'feature_demo':[3],'length':1,'name':'user_type','embedding_out_dim':8,'embedding_in_dim':10}
model = model_wide_deep(inputs_wide,inputs_deepX,input_deep1,input_deep2)

inputs_deeps
[
    {'feature_demo(not necessary)':list,'length':len(list),'name':'str','embedding_out_dim':int,'embedding_in_dim':vacab_size},
    ...,
    {'feature_demo(not necessary)':list,'length':len(list),'name':'str','embedding_out_dim':int,'embedding_in_dim':vacab_size}
]
'''

def model_wide_deep(inputs_wide,inputs_deepX,*inputs_deeps):
    #wide
    input_wide = Input(shape=(inputs_wide['length'],), name=inputs_wide['name'],dtype='float32')
    output_wide = Dense(units=inputs_wide['wide_output_dim'], activation="relu",kernel_regularizer=l1_l2(l1=1e-4,l2=1e-4))(input_wide)

    #deep
    inputs_deep = []
    embedding_deep = []
    lambda_deep = []
    for input_deep_dict in inputs_deeps:
        inputs_deep.append(Input(shape=(input_deep_dict["length"],), name=input_deep_dict["name"],dtype='float32'))
        embedding_deep.append(Embedding(embeddings_initializer='uniform',output_dim=input_deep_dict["embedding_out_dim"], input_dim=input_deep_dict["embedding_in_dim"], input_length=input_deep_dict["length"], mask_zero=True, name='embedding_'+input_deep_dict["name"])(inputs_deep[-1]))
        lambda_deep.append(Lambda(function=lambda x: K.reshape(x, shape=(-1, input_deep_dict["length"] * input_deep_dict["embedding_out_dim"])))(embedding_deep[-1]))

    inputs_deep_model = concatenate(inputs = lambda_deep,axis=-1)
    output_deep_hidden = Dense(inputs_deepX['deep_hidden_dim'],activation='relu')(inputs_deep_model)
    output_deep = Dense(inputs_deepX['deep_output_dim'],activation='relu')(output_deep_hidden)

    #wide and deep
    input_wide_deep = concatenate(inputs=[output_wide,output_deep],axis=1)
    output_wide_deep = Dense(1, activation='sigmoid', name="output_wide_deep")(input_wide_deep)
    model_wide_deep = Model(inputs=[input_wide] + inputs_deep, outputs=[output_wide_deep])
    model_wide_deep.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_score, precision_score, recall_score])#
    model_wide_deep.summary()
    return model_wide_deep