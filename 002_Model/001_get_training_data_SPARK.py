import pickle
import boto3
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Vale
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
#sc = spark.sparkContext         # Since Spark 2.0 'spark' is a SparkSession object that is by default created upfront and available in Spark shell, you need to explicitly create SparkSession object by using builder
spark = SparkSession.builder.getOrCreate()
#sc = SparkContext().getOrCreate()
sc = SparkContext._active_spark_context #devuelve la instancia existente
from pyspark.sql import SQLContext
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, StringType, FloatType, ArrayType, DoubleType, StructType, StructField
sqlContext = SQLContext(sc,spark)

# These paths should be changed to wherever you want to save the general data and where you want to save
# iteration specific data
base_save_path = "./"
iteration_save_path = "./institutional_affiliation_classification"
rutaDatos = "../Datos"

institutions = spark.read.parquet(f"{rutaDatos}/OA_static_institutions_single_file.parquet") \
    .filter(F.col('ror_id')!='')

print('institutions.cache().count() --------------------------------------')
print(institutions.cache().count())

df_affil = pd.read_csv(f"{rutaDatos}/Insumos_M1/file_parts_20231201_arg.csv")
df_affil = df_affil.rename(columns={'affiliation_ids': 'affiliation_id'})
df_affil.to_parquet(f"{rutaDatos}/static_affiliations.parquet")

print('Se crea el archivo static_affiliations.parquet con los datos de Insumos_M1/file_parts_20231201_arg.csv')

print('FINALIZADO OK')
