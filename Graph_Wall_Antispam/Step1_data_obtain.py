# -*- coding: utf-8 -*-
# @Time    : 8/6/18 10:02 AM
# @Author  : gaishi
# @Site    : 
# @File    : Step1_data_obtain.py
# @Software: IntelliJ IDEA

from Hive_Interaction import Hive

# data_time='2018-07-04'
# N = 100
# table_name = 'gaishi_nrt_u_f_ytd_trainset'
# start_column = 2
# end_column = -1
def get_data_from_hive(data_time,N,table_name,start_column,end_column):
    hive = Hive('graph_wall_wide_deep')
    table = {'fields':'*','name':'gaishi_nrt_u_f_ytd_trainset'}

    pandas_df = hive.Get_Pandas_From_Table(table,data_time,N)

    y = pandas_df.cheated
    X = pandas_df.fillna(0,inplace=True)
    X = pandas_df.iloc[:,start_column:end_column]

    print X.head()

    X_train = X
    y_train = y
    X_test = X # 不推荐，应该使用新数据，调试代码时使用
    y_test = y

    return X_train,y_train,X_test,y_test