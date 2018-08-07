# -*- coding: utf-8 -*-
# @Time    : 3/1/18 10:00 AM
# @Author  : gaishi
# @Site    :
# @File    : Hive_Interaction.py
# @Software: IntelliJ IDEA

# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')

from pyspark.sql import SparkSession
from pyspark import SparkConf

# from pyspark import SparkContext
# from pyspark.sql import HiveContext

# .config("spark.storage.memoryFraction", "0.5") \
# .config("spark.memory.storageFraction","0.5")\
# .config("spark.memory.fraction","32000") \
#.config("spark.files.overwrite", "true") \

import pandas as pd

class Hive:

    def __init__(self,app_name):
        self.spark = SparkSession \
            .builder \
            .appName(app_name) \
            .enableHiveSupport() \
            .config("spark.rpc.message.maxSize", "60") \
            .config("spark.files.overwrite", "true") \
            .config("spark.hadoop.validateOutputSpecs", "false") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.kryoserializer.buffer.max", "2000m") \
            .config("fs.defaultFS", "hdfs://mgjcluster") \
            .getOrCreate()

    def Get_Pandas_From_Table(self,table_fields,table_name,date_time,limit):
        print "[INFO] Data Time：",date_time

        sql = '''
            select
                %s
            from
                %s
            where created_date='%s'
            limit 100
            ''' % (table_fields, table_name, date_time,limit)

        print sql

        df = self.spark.sql(sql)

        pandas_df = df.toPandas()

        print pandas_df.head()

        return pandas_df


    def Get_Pandas_From_Table_PN(self, table_fields, table_name, label_name, date_time, N_positive, N_negtive):
        print "[INFO] Data Time：",date_time

        sql_pos = '''
            select
                %s
            from
                %s
            where created_date='%s' and %s = 1
            limit %d
            ''' % (table_fields, table_name, date_time, label_name, N_positive)

        sql_neg = '''
            select
                %s
            from
                %s
            where created_date='%s' and %s = 0
            limit %d
            ''' % (table_fields, table_name, date_time, label_name, N_negtive)

        print sql_pos

        df1 = self.spark.sql(sql_pos)
        df2 = self.spark.sql(sql_pos)

        df = df1.union(df2)

        pandas_df = df.toPandas()

        print pandas_df.head()

        return pandas_df

    # table = { 'type' : 'all'
    # , 'fields':'*'
    # , 'name':'gaishi_nrt_u_f_ytd_trainset'
    # , 'N': 100000
    # , 'start_column': 2
    # , 'end_column': -1}
    #
    # postive set and negtive set
    # table = { 'type' : 'PN'
    # , 'fields':'*'
    # , 'label_name': 'cheat'
    # , 'name':'gaishi_nrt_u_f_ytd_trainset'
    # , 'N_pos': 50000
    # , 'N_neg': 100000
    # , 'start_column': 2
    # , 'end_column': -1}
    # retrun X_train,y_train
    def get_data_from_hive(self, table, data_time):
        if table['type'] == 'all':
            pandas_df = self.Get_Pandas_From_Table(table['fields'],table['name'],data_time,table['N'])
        elif table['type'] == 'PN':
            pandas_df = self.Get_Pandas_From_Table_PN(table['fields'],table['name'],table['label_name'],data_time,table['N_pos'],table['N_neg'])

        y = pandas_df.cheated
        X = pandas_df.fillna(0,inplace=True)
        X = pandas_df.iloc[:,table['start_column']:table['end_column']]

        print X.head()

        X_train = X
        y_train = y

        return X_train,y_train

    def __del__(self):
        self.spark.stop()

