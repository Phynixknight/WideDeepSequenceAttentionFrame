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
    feature = np.zeros(len(bucket),dtype=int)

def onehot_cross_encode_list(value1,value2,bucket1,bucket2):
    feature = np.zeros()

def encode_cross(feature_list_A,feature_list_B):
    pass

# the method suit for the array that some maximum value is unnormal but common,
#       the tail numberic will be bucket as exponent
# input the nd_array ,change to the bucket 1d_array, the opration will in-place, and return the bucket_dict
# warning:
#       for the predict set, if you can not ensure every value will appear at least onces
#       you must bucket the feature as the same way but filter with the bucket_dict
# input:
#     ndarray_like:1d_array like object
#               like pandas Series/DataFrame and numpy 1darray/ndarray
#     min_pos: minimum bucket boundary
#     max_exp: maximum bucket boundary exponent
#               the max boundary will be min_pos.op(pow(2^max_exp))
#     strategy: the operation between the min_pos and max_exp
#               default will be '+' (add)
#               other choise can be '*' (mul)
# output:
#     bucket_dict: ensure predict feature will not appear the value that less than min_pos
#             you can use like this: np.vectorize(lambda x: x if x in bucket_dict else 0)(one_darray_like)
# bucket as exponent boundaries
#     the boundaries of bucket is [min_pos,min_pos.op(2^0),min_pos.op(2^1),min_pos.op(2^2),...]
# example:
#     if min_position = 10、strategy is ‘+’，then the boundaries of bucket is [10,10+0,10+2,10+4,10+8,10+16,10+32,...]
#     if min_position = 10、strategy is ‘*’，then the boundaries of bucket is [10,10*1,10*2,10*4,10*8,10*16,10*32,...]
def bucket_boundes_exp(ndarray_like,min_pos,max_exp,strategy,nd_type):
    if strategy == '*':
        op = min_pos.__mul__
    else:
        # '+' by default
        op = min_pos.__add__

    for i in range(max_exp):
        ndarray_like[np.where(np.logical_and(np.greater_equal(ndarray_like,op(pow(2,i))),np.less(ndarray_like,op(pow(2,i+1)))))] = min_pos+i

    if nd_type=='1d':
        return set(ndarray_like)
    else:
        # '2d' by default
        return set([x for y in ndarray_like for x in y])