# -*- coding: utf-8 -*-
# @Time    : 2018-01-01 00:00:00
# @Author  : gaishi

import numpy as np
import pandas as pd
import operator

def encode_list(feature_list):
    return dict(zip(feature_list,range(1,len(feature_list)+1)))

def encode_bucket_list(feature_list, bucket):
    pass

def onehot_encode_bucket_value(value, bucket):
    '''buckets is sorted ascending'''
    feature = np.zeros(len(bucket),dtype=int)

def onehot_cross_encode_list(value1,value2,bucket1,bucket2):
    feature = np.zeros()

def encode_cross(feature_list_A,feature_list_B):
    pass

# the method suit for the array that some maximum value is unnormal but common,
#       the tail numberic will be bucket as exponent
# input the nd_array ,change to the bucket 1d_array, the opration will in-place, and return the result and bucket_dict
# warning:
#       for the predict set, if you can not ensure every value will appear at least onces
#       you must bucket the feature as the same way but filter with the bucket_dict
#
#       [notice] the fuction change the value in place, but
#       if the real parameter is pandas_DataFrame,note that pd.iloc[] is a copy of orign,
#           ,it happened implicit, so I return the ndarray_like.
#       otherwise if the type of ndarray_like is np.ndarray, ndarray_like is itself, you can omit the first return value.
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
    if isinstance(ndarray_like,pd.core.frame.DataFrame) or isinstance(ndarray_like,pd.core.frame.Series):
        ndarray_like = ndarray_like.values
        # there's different between pandas dataframe and numpy.ndarray indexer[] in resolving tuple list like index
        # like np_array_like[(array([1,2,3,4]),array([0,0,0,0]))]
        # in pandas iloc can resolve the second list as columns, but like [1,[0,0,0,0]], and numpy treate it as a column [1,0] [2,0]


    if strategy == '*':
        op = min_pos.__mul__
    else:
        # '+' by default
        op = min_pos.__add__

    pow_list = [pow(2,i) for i in range(max_exp + 1)]

    for i in range(max_exp):
        ndarray_like[np.where(np.logical_and(np.greater_equal(ndarray_like,op(pow_list[i])),np.less(ndarray_like,op(pow_list[i+1]))))] = min_pos+i

    if nd_type=='1d':
        return ndarray_like,set(ndarray_like)
    else:
        # '2d' by default
        return ndarray_like,set([x for y in ndarray_like for x in y])


# bucket as exponent boundaries
#     the boundaries of bucket is [min_pos,int1,int2,...,max_pos]
#     must ensure the max ndarray_like is less than max_pos value
# example:
#     boundaries of bucket is [10,20,30,40,50,60,70]
#     the bucket will [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
#
# notice: the fuction change the value in place, but
#       if the real parameter is pandas_DataFrame,note that pd.iloc[] is a copy of orign,
#           ,it happened implicit, so I return the ndarray_like.
#       otherwise if the type of ndarray_like is np.ndarray, ndarray_like is itself, you can omit the first return value.
def bucket_boundes_has_min(ndarray_like,boundaries,nd_type):
    if isinstance(ndarray_like,pd.core.frame.DataFrame) or isinstance(ndarray_like,pd.core.frame.Series):
        ndarray_like = ndarray_like.values
    # there's different between pandas dataframe and numpy.ndarray indexer[] in resolving tuple list like index
    # like df[(array([1,2,3,4]),array([0,0,0,0]))]
    # pandas resolve the second list as columns like [1,[0,0,0,0]], but numpy treate it as a column [1,0] [2,0], neighter of iloc and loc

    min_pos = boundaries[0]

    for i in range(len(boundaries) - 1):
        ndarray_like[np.where(np.logical_and(np.greater_equal(ndarray_like,boundaries[i]),np.less(ndarray_like,boundaries[i+1])))] = min_pos+i

    if nd_type=='1d':
        return ndarray_like,set(ndarray_like)
    else:
        # '2d' by default
        return ndarray_like,set([x for y in ndarray_like for x in y])

# bucket as exponent boundaries
#     the boundaries of bucket is [0,upbounds_of_zero_bucket,int1,int2,...,max_pos]
#     must ensure the max ndarray_like is less than max_pos value
# example:
#     boundaries of bucket is [0,1,10,20,30,40,50,60,70]
#     the bucket will [0,1,2,3,4,5,6,7]
#     0:[0,1),1:[1,10),2:[10,20),3:[20,30),4:[30,40),5:[40,50),6:[50,60),7:[60:70)
# notice: the fuction change the value in place, but
#       if the real parameter is pandas_DataFrame,note that pd.iloc[] is a copy of orign,
#           ,it happened implicit, so I return the ndarray_like.
#       but you can also use the pd['colums_name'] method, it will return a Series reference in the Frame.
#       otherwise if the type of ndarray_like is np.ndarray, ndarray_like is itself, you can omit the first return value.
def bucket_boundes(ndarray_like,boundaries):
    if isinstance(ndarray_like,pd.core.frame.DataFrame) or isinstance(ndarray_like,pd.core.frame.Series):
        ndarray_like = ndarray_like.values
        # there's different between pandas dataframe and numpy.ndarray indexer[] in resolving tuple list like index
        # like df[(array([1,2,3,4]),array([0,0,0,0]))]
        # pandas resolve the second list as columns like [1,[0,0,0,0]], but numpy treate it as a column [1,0] [2,0]

    min_pos = boundaries[0]

    for i in range(len(boundaries) - 1):
        ndarray_like[np.where(np.logical_and(np.greater_equal(ndarray_like,boundaries[i]),np.less(ndarray_like,boundaries[i+1])))] = min_pos+i

    return ndarray_like

# encode 2darray to one_hot
# return new one_hot 2darray features
# for example:
# array[[0, 0, 3],
#       [1, 1, 0],
#       [0, 2, 1],
#       [1, 0, 2]]
# dim : 3
# n_values: [2, 3, 4]
# for dim 0:
#            0:[0,1],1:[1,0]
# for dim 1:
#            0:[0,0,1],1:[0,1,0],2:[1,0,0]
# for dim 2:
#            0:[0,0,0,1],1:[0,0,1,0],2:[0,1,0,0],3:[1,0,0,0]
# return:
#        array([[ 1.,  0.,      1.,  0.,  0.,      0.,  0.,  0.,  1.],
#               [ 0.,  1.,      0.,  1.,  0.,      1.,  0.,  0.,  0.],
#               [ 1.,  0.,      0.,  0.,  1.,      0.,  1.,  0.,  0.],
#               [ 0.,  1.,      1.,  0.,  0.,      0.,  0.,  1.,  0.]])
def encode_one_hot(ndarray_like):
    if isinstance(ndarray_like,pd.core.frame.DataFrame) or isinstance(ndarray_like,pd.core.frame.Series):
        ndarray_like = ndarray_like.values
    N = len(ndarray_like)
    dim = len(ndarray_like[0])
    diff_values = [set([]) for i in range(dim)]
    n_values = [0 for i in range(dim)]
    dic = [{} for i in range(dim)]
    for i in range(dim):
        diff_values[i] = set(ndarray_like[:,i])
        n_values[i] = len(diff_values[i])
        dic[i] = dict(zip(diff_values[i],range(n_values[i])))
        offset = sum(n_values[:i])
        for k in dic[i]: dic[i][k] = dic[i][k] + offset
    encode_ndarray = np.zeros(shape=(N,sum(n_values)))
    for n in range(N):
        for i in range(dim):
            encode_ndarray[n][dic[i][ndarray_like[n][i]]] = 1
    return encode_ndarray


# encode 2-column 2darray to multi_hot
# return new one_hot 2darray features and dict
# input dim must be 2
# n_values : different dim featrues number
# output dim will be n_values[0] * n_values[1] ,even not all will access
# example 1:
# array[[0, 0],
#       [1, 1],
#       [0, 2],
#       [1, 0]]
# dim : 2
# n_values: [2, 3]
# out_dim : 2 * 3 = 6
#         no matter whether all this six can be accessable
# all the dict:
#            0,0:[1,0,0,0,0,0];
#            0,1:[0,1,0,0,0,0];
#            0,2:[0,0,1,0,0,0];
#            1,0:[0,0,0,1,0,0];
#            1,1:[0,0,0,0,1,0];
#            1,2:[0,0,0,0,0,1];
# for this example:
#       (0,1);(1,2) is not accessable, but we also encode it
#       for the reason that we assume all independent feature will access all
#              ,but 2-gram can be missing in train_set/valid_set
# return:
#        array([[1,0,0,0,0,0],
#               [0,0,0,0,1,0],
#               [0,0,1,0,0,0],
#               [0,0,0,1,0,0]])
# example 2:
#     np.array([['a', 'a'], ['b','e'], ['c','d']]
# return:
#     array([[ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#            [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
#            [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.]])
def encode_bi_multi_hot(ndarray_like):
    N = len(ndarray_like)
    dim = len(ndarray_like[0])
    assert dim == 2
    diff_values = [set([]) for i in range(dim)]
    n_values = [0 for i in range(dim)]
    dic = [{} for i in range(dim)]
    for i in range(dim):
        diff_values[i] = set(ndarray_like[:,i])
        n_values[i] = len(diff_values[i])
        dic[i] = dict(zip(diff_values[i],range(n_values[i])))
    encode_ndarray = np.zeros(shape=(N,reduce(operator.mul,n_values)))
    for n in range(N):
        encode_ndarray[n][dic[0][ndarray_like[n][0]] + dic[1][ndarray_like[n][1]] * n_values[0]] = 1
    return encode_ndarray

# encode n-column 2darray to multi_hot
# return new one_hot 2darray features and dict
# n_values : different dim featrues number
# output dim will be n_values[0] * n_values[1] ,even not all will access
# for example:
# np.array([['a', 'a','x'], ['b','e','e'], ['c','d','y']]
# dim : 3
# n_values: [3, 3, 3]
# out_dim : 3 * 3 * 3 = 27
#         no matter whether all this six can be accessable
# return:
#        array([[ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
#                0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.],
#               [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
#                0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.],
#               [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,
#                0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.]])
def encode_n_multi_hot(ndarray_like):
    N = len(ndarray_like)
    dim = len(ndarray_like[0])
    diff_values = [set([]) for i in range(dim)]
    n_values = [0 for i in range(dim)]
    dic = [{} for i in range(dim)]
    for i in range(dim):
        diff_values[i] = set(ndarray_like[:,i])
        n_values[i] = len(diff_values[i])
        dic[i] = dict(zip(diff_values[i],range(n_values[i])))
    encode_ndarray = np.zeros(shape=(N,reduce(operator.mul,n_values)))
    for n in range(N):
        pos = 0
        for i in range(dim):
            pos += dic[i][ndarray_like[n][i]] * sum(n_values[0:i])
        encode_ndarray[n][dic[0][ndarray_like[n][0]] + dic[1][ndarray_like[n][1]] * n_values[0]] = 1
    return encode_ndarray