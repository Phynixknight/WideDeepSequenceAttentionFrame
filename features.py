import numpy as np

def encode_list(feature_list):
    return dict(zip(feature_list,range(1,len(feature_list)+1)))

def encode_bucket_1darray(feature_1darray,base_value,max_bucket_index):
    '''太大的值进行log压缩'''
    for i in range(0,max_bucket_index):
        feature_1darray[np.where(np.logical_and(np.greater_equal(feature_1darray,base_value*pow(2,i)),np.less(feature_1darray,base_value *pow(2,i+1))))] = base_value+i

def encode_bucket_list(feature_list, bucket):
    

def onehot_encode_bucket_value(value, bucket):
    '''buckets is sorted ascending'''
    feature = np.zeros(len(buckets),dtype=int)

def onehot_cross_encode_list(value1,value2,bucket1,bucket2):
    feature = np.zeros()

def encode_cross(feature_list_A,feature_list_B):
