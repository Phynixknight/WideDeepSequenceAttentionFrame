#coding: utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf-8')


from pyspark.sql import SparkSession
from pyspark import SparkConf

import pandas as pd

class Hive:

    def __init__(self):
        self.spark = SparkSession \
            .builder \
            .appName("anti_spam_spark_u_p_predict_%s" % (date_time)) \
            .enableHiveSupport() \
            .config("spark.rpc.message.maxSize", "60") \
            .config("spark.files.overwrite", "true") \
            .config("spark.hadoop.validateOutputSpecs", "false") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.kryoserializer.buffer.max", "2000m") \
            .config("fs.defaultFS", "hdfs://mgjcluster") \
            .getOrCreate()

    def Get_Pandas_From_Table(self,table,date_time):
        print "[INFO] Data Time：",date_time

        sql = '''
            select 
                %s
            from  
                %s 
                where created_date='%s'
            ''' % (table['fields'],table['name'],date_time)

        df = self.spark.sql(sql)

        pandas_df = df.toPandas()

        return pandas_df

    def __del__(self):
        self.spark.stop()

