# -*- coding: utf-8 -*-
# @Time    : 8/6/18 10:34 AM
# @Author  : gaishi
# @Site    : 
# @File    : Step2_data_process.py
# @Software: IntelliJ IDEA

from features import bucket_boundes
from features import encode_one_hot

# expose_bucket_boundaries = [0,1,10,100,200,300,400,500,600,700,800,900,1500,3000,5000,10000,1000000]
# columns = [['expose_ad', 'expose_natrual_ad','expose_non_ad','expose_live','expose_topic','expose_content','expose_mofang_zhibo_kuaiqiang']
#           ,['expose_num', 'mogujie_expose', 'meilishuo_expose', 'pc_expose', 'h5_expose']
#           ]
# expose_biz_featrue,expose_plat_featrue = process_data(expose_bucket_boundaries,columns)
def deep_features(pandas_df, expose_bucket_boundaries, columns):
    features = []

    for columns_i in columns:
        deep_featrue(pandas_df, expose_bucket_boundaries, columns_i) # using side-effect of the method
        for colum in columns_i:
            bucket_boundes(pandas_df[colum], expose_bucket_boundaries)
        features.append(pandas_df[columns_i])

    return features

# using side-effect
def deep_featrue(pandas_df,expose_bucket_boundaries,columns):
    for colum in columns:
        bucket_boundes(pandas_df[colum], expose_bucket_boundaries)

def wide_features(pandas_df, columns):
    features = encode_one_hot(pandas_df[columns])
    return features

