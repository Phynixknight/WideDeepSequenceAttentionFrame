from keras.layers import Input, Dense, TimeDistributed, Embedding, concatenate, LSTM, Permute, Bidirectional, RepeatVector, Reshape, merge, Lambda
from keras.models import Model
from keras import backend as K
from keras.regularizers import l1_l2
from utils import f1_score,precision_score,recall_score

'''
example:
inputs_wide = {'feature_demo':[0,0,1,1,0,0,0,1,0,1],'length':10,'name':'wide_input','wide_output_dim':32,'l1':1e-4,'l2':1e-4}
inputs_deepX = {'deep_hidden_dim':128,'deep_output_dim':32}
inputs_sequenceX = {'sequence_out_dim':32}
input_deep1={'feature_demo':[0,0,1,3,7,11,2,10,6],'length':9,'name':'shop_id_type','embedding_out_dim':16,'embedding_in_dim':20}
input_deep2={'feature_demo':[3],'length':1,'name':'user_type','embedding_out_dim':8,'embedding_in_dim':10}
input_sequence1={'feature_demo':[1,2,0,1,4,3,0,2,0],'length':10,'name':'time_diff','embedding_out_dim':16,'embedding_in_dim':60}
input_sequence2={'feature_demo':[1,3,4,5,7,9,13,17,23],'length':10,'name':'url_seq','embedding_out_dim':16,'embedding_in_dim':60}
model = model_wide_deep(inputs_wide,inputs_deepX,input_deep1,input_deep2)

inputs_deeps
[
    {'feature_demo(not necessary)':list,'length':len(list),'name':'str','embedding_out_dim':int,'embedding_in_dim':vacab_size,'type':'deep'},
    ...,
    {'feature_demo(not necessary)':list,'length':len(list),'name':'str','embedding_out_dim':int,'embedding_in_dim':vacab_size, 'type':'deep'},
    {'feature_demo(not necessary)':list,'length':len(list),'name':'str','embedding_out_dim':int,'embedding_in_dim':vacab_size,'type':'sequence'},
    ...,
    {'feature_demo(not necessary)':list,'length':len(list),'name':'str','embedding_out_dim':int,'embedding_in_dim':vacab_size, 'type':'sequence'},
]
'''

def model_wide_deep(inputs_wide,inputs_deepX,inputs_sequenceX,*inputs_deeps):
    #wide model
    input_wide = Input(shape=(inputs_wide['length'],), name=inputs_wide['name'],dtype='float32')
    output_wide = Dense(units=inputs_wide['wide_output_dim'], activation="relu",kernel_regularizer=l1_l2(l1=1e-4,l2=1e-4))(input_wide)

    #deep
    inputs_deep = []
    embedding_deep = []
    lambda_deep = []

    #sequence
    inputs_sequence=[]
    embedding_sequence=[]
    sequence_length = 0
    sequence_num = 0

    for input_deep_dict in inputs_deeps:
        #deep
        if input_deep_dict['type'] == 'deep':
            inputs_deep.append(Input(shape=(input_deep_dict["length"],), name=input_deep_dict["name"],dtype='float32'))
            embedding_deep.append(Embedding(embeddings_initializer='uniform',output_dim=input_deep_dict["embedding_out_dim"], input_dim=input_deep_dict["embedding_in_dim"], input_length=input_deep_dict["length"], mask_zero=True, name='embedding_'+input_deep_dict["name"])(inputs_deep[-1]))
            lambda_deep.append(Lambda(function=lambda x: K.reshape(x, shape=(-1, input_deep_dict["length"] * input_deep_dict["embedding_out_dim"])))(embedding_deep[-1]))
        #sequence
        elif input_deep_dict['type'] == 'sequence':
            sequence_num += 0
            sequence_length = input_deep_dict['length']
            inputs_sequence.append(Input(shape=(input_deep_dict['length'],),name=input_deep_dict['name'],dtype='int32'))
            embedding_sequence.append(Embedding(output_dim=input_deep_dict['embedding_out_dim'],input_dim=input_deep_dict['embedding_in_dim'],input_length=input_deep_dict['length'],mask_zero=True)(inputs_sequence[-1]))

    #sequence and attention model
    input_sequence = concatenate(inputs=embedding_sequence,axis=2)
    bilstm = Bidirectional(LSTM(inputs_sequenceX['sequence_out_dim'], return_sequences=True,name='output_sequence'))(input_sequence)
    attention_implements = TimeDistributed(Dense(1, activation='tanh',name='attention_weights_cal'))(bilstm)

    #TimeDistributedDense layer will produce a 3D tensor shape of (batch_size,L,1), and when you apply the softmax activation, the output maybe not correct.
    lambda_attention = Lambda(lambda x: x, output_shape=lambda x: x)(attention_implements)
    reshape_attention = Reshape((sequence_length,))(lambda_attention)
    dense_attention = Dense(sequence_length, activation='softmax', use_bias=False, name="attetion_weights")(reshape_attention)
    repeatevector = RepeatVector(2 * inputs_sequenceX['sequence_out_dim'])(dense_attention)
    attention_probs = Permute([2, 1],name='attention_probs_flatten')(repeatevector)

    # attention = bilstm * attention_probs
    attention_mul = merge([bilstm, attention_probs], mode="mul")
    output_attention = Lambda(lambda x: K.sum(x, axis=1),name='output_attention')(attention_mul)

    #deep model
    inputs_deep_model = concatenate(inputs = lambda_deep,axis=-1)
    output_deep_hidden = Dense(inputs_deepX['deep_hidden_dim'],activation='relu')(inputs_deep_model)
    output_deep = Dense(inputs_deepX['deep_output_dim'],activation='relu')(output_deep_hidden)

    #wide and deep
    input_wide_deep = concatenate(inputs=[output_wide,output_deep,output_attention],axis=1)
    output_wide_deep = Dense(1, activation='sigmoid', name="output_wide_deep")(input_wide_deep)

    #all model
    model_wide_deep = Model(inputs=[input_wide] + inputs_deep + inputs_sequence, outputs=[output_wide_deep])
    model_wide_deep.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_score, precision_score, recall_score])
    model_wide_deep.summary()
    return model_wide_deep