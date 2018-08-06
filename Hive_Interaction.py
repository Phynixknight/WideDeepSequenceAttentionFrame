#coding: utf-8

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

    def Get_Pandas_From_Table(self,table,date_time,limit):
        print "[INFO] Data Time：",date_time

        sql = '''
            select
                %s
            from
                %s
            where created_date='%s'
            limit 100
            ''' % (table['fields'],table['name'],date_time,limit)

        print sql

        df = self.spark.sql(sql)

        pandas_df = df.toPandas()

        print pandas_df.head()

        return pandas_df

    def __del__(self):
        self.spark.stop()

