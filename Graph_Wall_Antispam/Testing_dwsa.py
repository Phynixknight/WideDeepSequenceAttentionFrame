# -*- coding: utf-8 -*-
# @Time    : 8/7/18 11:57 AM
# @Author  : gaishi
# @Site    : 
# @File    : Testing_dwsa.py.py
# @Software: IntelliJ IDEA

from Hive_Interaction import Hive

# Step1 get data from hive

app_name = 'Graph_Wall_Antispam_Train'
data_time = '2018-07-05'

table = { 'type' : 'PN'
    , 'fields':'*'
    , 'name':'gaishi_nrt_u_f_ytd_trainset'
    , 'label_name': 'cheat'
    , 'N_pos': 50000
    , 'N_neg': 50000
    , 'start_column': 2
    , 'end_column': -1 }

hive = Hive(app_name)

X_train,y_train = hive.get_data_from_hive(table,data_time)

# Step2 data_process

from features import deep_features
from features import wide_features_dict
from DeepAndWideModel import model_wide_deep

expose_bucket_boundaries = [0,1,10,100,200,300,400,500,600,700,800,900,1500,3000,5000,10000,1000000]
deep_columns = [['expose_ad', 'expose_natrual_ad','expose_non_ad','expose_live','expose_topic'
                    ,'expose_content','expose_mofang_zhibo_kuaiqiang']
    , ['expose_num', 'mogujie_expose', 'meilishuo_expose', 'pc_expose', 'h5_expose']
    , ['click_num', 'click_ad', 'click_natrual_ad', 'click_non_ad', 'click_live', 'click_topic', 'click_content',
       'mogujie_click', 'meilishuo_click', 'pc_click', 'h5_click', 'click_mofang_zhibo_kuaiqiang']
    , ['cart_num', 'cart_ad', 'cart_natrual_ad', 'cart_non_ad', 'cart_live', 'cart_topic', 'cart_content',
       'mogujie_cart', 'meilishuo_cart', 'pc_cart', 'h5_cart', 'cart_mofang_zhibo_kuaiqiang']
    , ['fav_num', 'fav_ad', 'fav_natrual_ad', 'fav_non_ad', 'fav_live', 'fav_topic', 'fav_content',
       'mogujie_fav', 'meilishuo_fav', 'pc_fav', 'h5_fav', 'fav_mofang_zhibo_kuaiqiang']
    , ['user_active_p_cnt_login', 'active_nonvalid_uuid_p_cnt_login', 'diff_p_login', 'diff_platform_p_login',
       'diff_valid_device_p_login', 'diff_device_p_login', 'average_plat_p_cnt_login', 'platform_0_p_cnt_login',
       'platform_1_p_cnt_login', 'platform_2_p_cnt_login', 'platform_3_p_cnt_login', 'platform_4_p_cnt_login',
       'platform_5_p_cnt_login', 'platform_6_p_cnt_login', 'platform_9_10_p_cnt_login',
       'platform_23_24_p_cnt_login', 'platform_25_29_p_cnt_login']
    , ['active_nonvalid_uuid_p_cnt_nonlogin', 'diff_p_nonlogin', 'diff_platform_p_nonlogin',
       'diff_valid_device_p_nonlogin', 'diff_device_p_nonlogin', 'average_plat_p_cnt_nonlogin',
       'platform_0_p_cnt_nonlogin', 'platform_1_p_cnt_nonlogin', 'platform_2_p_cnt_nonlogin',
       'platform_3_p_cnt_nonlogin', 'platform_4_p_cnt_nonlogin', 'platform_5_p_cnt_nonlogin',
       'platform_6_p_cnt_nonlogin', 'platform_9_10_p_cnt_nonlogin', 'platform_23_24_p_cnt_nonlogin',
       'platform_25_29_p_cnt_nonlogin']
    , ['user_active_e_cnt_login', 'active_nonvalid_uuid_e_cnt_login', 'diff_e_login', 'diff_platform_e_login',
       'diff_valid_device_e_login', 'diff_device_e_login', 'average_plat_e_cnt_login', 'platform_0_e_cnt_login',
       'platform_1_e_cnt_login', 'platform_2_e_cnt_login', 'platform_3_e_cnt_login', 'platform_4_e_cnt_login',
       'platform_5_e_cnt_login', 'platform_6_e_cnt_login', 'platform_9_10_e_cnt_login',
       'platform_23_24_e_cnt_login', 'platform_25_29_e_cnt_login', 'user_active_e_cnt_nonlogin']
    , ['user_active_e_cnt_nonlogin', 'active_nonvalid_uuid_e_cnt_nonlogin', 'diff_e_nonlogin',
       'diff_platform_e_nonlogin', 'diff_valid_device_e_nonlogin', 'diff_device_e_nonlogin',
       'average_plat_e_cnt_nonlogin', 'platform_0_e_cnt_nonlogin', 'platform_1_e_cnt_nonlogin',
       'platform_2_e_cnt_nonlogin', 'platform_3_e_cnt_nonlogin', 'platform_4_e_cnt_nonlogin',
       'platform_5_e_cnt_nonlogin', 'platform_6_e_cnt_nonlogin', 'platform_9_10_e_cnt_nonlogin',
       'platform_23_24_e_cnt_nonlogin', 'platform_25_29_e_cnt_nonlogin']
    , ['diff_items', 'diff_platform', 'diff_valid_device', 'diff_device', 'diff_valid_ip', 'diff_ip']
                ]

expose_biz_deep_featrue \
    , expose_plat_deep_feature \
    , click_deep_feature \
    , cart_deep_feature \
    , fav_deep_feature \
    , page_login_deep_feature \
    , page_nonlogin_deep_feature \
    , event_login_deep_feature \
    , event_nonlogin_deep_feature \
    , diff_deep_featrue = deep_features(X_train,expose_bucket_boundaries,deep_columns)

wide_columns = ['mogujie_order','meilishuo_order','weixin_qq_order','order_zb_kq_mf'
    ,'diff_platform','diff_valid_device','diff_device','diff_valid_ip','diff_ip']
wide_feature = wide_features_dict(X_train,wide_columns, 'one_hot.dic')

inputs_wide = {'feature': wide_feature,'length':len(wide_feature[0]),'name':'wide_input','wide_output_dim':32,'l1':1e-4,'l2':1e-4}
inputs_deepX = {'deep_hidden_dim':128,'deep_output_dim':32}
input_deep1={'feature': expose_biz_deep_featrue,'length':expose_biz_deep_featrue.columns.size,'name':'expose_biz_deep_featrue','embedding_out_dim':16,'embedding_in_dim':20}
input_deep2={'feature': expose_plat_deep_feature,'length':expose_plat_deep_feature.columns.size,'name':'expose_plat_deep_feature','embedding_out_dim':16,'embedding_in_dim':20}
input_deep3={'feature': click_deep_feature,'length':click_deep_feature.columns.size,'name':'click_deep_feature','embedding_out_dim':16,'embedding_in_dim':20}
input_deep4={'feature': cart_deep_feature,'length':cart_deep_feature.columns.size,'name':'cart_deep_feature','embedding_out_dim':16,'embedding_in_dim':20}
input_deep5={'feature': fav_deep_feature,'length':fav_deep_feature.columns.size,'name':'fav_deep_feature','embedding_out_dim':16,'embedding_in_dim':20}
input_deep6={'feature': page_login_deep_feature,'length':page_login_deep_feature.columns.size,'name':'page_login_deep_feature','embedding_out_dim':16,'embedding_in_dim':20}
input_deep7={'feature': page_nonlogin_deep_feature,'length':page_nonlogin_deep_feature.columns.size,'name':'page_nonlogin_deep_feature','embedding_out_dim':16,'embedding_in_dim':20}
input_deep8={'feature': event_login_deep_feature,'length':event_login_deep_feature.columns.size,'name':'event_login_deep_feature','embedding_out_dim':16,'embedding_in_dim':20}
input_deep9={'feature': event_nonlogin_deep_feature,'length':event_nonlogin_deep_feature.columns.size,'name':'event_nonlogin_deep_feature','embedding_out_dim':16,'embedding_in_dim':20}
input_deep10={'feature':diff_deep_featrue,'length':diff_deep_featrue.columns.size,'name':'diff_deep_featrue','embedding_out_dim':8,'embedding_in_dim':10}


# Step3 predicting can use CPU

x = {
    inputs_wide['name']: inputs_wide['feature'],
    input_deep1['name']: input_deep1['feature'],
    input_deep2['name']: input_deep2['feature'],
    input_deep3['name']: input_deep3['feature'],
    input_deep4['name']: input_deep4['feature'],
    input_deep5['name']: input_deep5['feature'],
    input_deep6['name']: input_deep6['feature'],
    input_deep7['name']: input_deep7['feature'],
    input_deep8['name']: input_deep8['feature'],
    input_deep9['name']: input_deep9['feature'],
    input_deep10['name']: input_deep10['feature']
}


# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # set before tensorflow
#
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# set_session(tf.Session(config=config))

from keras.layers import Input, Dense, TimeDistributed, Embedding, concatenate, LSTM, Permute, Bidirectional, RepeatVector, Reshape, merge, Lambda, Flatten
from keras.models import Model
from keras import backend as K
from keras.regularizers import l1_l2

model = model_wide_deep(inputs_wide,inputs_deepX,input_deep1,input_deep2,input_deep3,input_deep4,input_deep5,input_deep6,input_deep7,input_deep8,input_deep9)

model.load_weights("antispam_graph_wall_weights.h5")

classes = model.predict(x, batch_size=128, verbose=1)

# Step 4 KPI

import numpy as np
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,recall_score,precision_score,classification_report

print "AUC Score (Test): %f" % roc_auc_score(y_train, classes[:,0])

classes2 = np.asarray([0 if x < .90 else 1 for x in classes[:,0]],dtype='int32')

print "预测第二天\n精准率Accuracy : %.4g" % accuracy_score(y_train, classes2)
print "准确率Precision : %.4g" % precision_score(y_train, classes2)
print '召回率Recall:%.4g' % recall_score(y_train, classes2)
print 'F1:%.4g' % f1_score(y_train, classes2)