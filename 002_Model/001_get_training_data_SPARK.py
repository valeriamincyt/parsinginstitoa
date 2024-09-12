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
iteration_save_path = "./institutional_affiliation_classification/"
rutaDatos = "../Datos/"

institutions = spark.read.parquet(f"{rutaDatos}OA_static_institutions_single_file.parquet") \
    .filter(F.col('ror_id')!='')

print('institutions.cache().count() --------------------------------------')
print(institutions.cache().count())

df_affil = pd.read_csv(f"{rutaDatos}Insumos_M1/file_parts_20231201_arg.csv")
df_affil = df_affil.rename(columns={'affiliation_ids': 'affiliation_id'})
df_affil.to_parquet(f"{rutaDatos}static_affiliations.parquet")

print('Se crea el archivo static_affiliations.parquet con los datos de Insumos_M1/file_parts_20231201_arg.csv')

print('El archivo contiene: --------------------------------')
print(df_affil.shape)
print('Columnas: -----------------------------')
print(df_affil.columns)
print('Primer registro: -------------------------------')
df_affil.head(1)

affiliations = spark.read.parquet(f"{rutaDatos}static_affiliations.parquet")

print('affiliations.cache().count() --------------------------------------')
print(affiliations.cache().count())
print(affiliations.head(1))

#### Getting ROR aff strings

dedup_affs = affiliations.select(F.trim(F.col('original_affiliation')).alias('original_affiliation'), 'affiliation_id')\
.filter(F.col('original_affiliation').isNotNull())\
.filter(F.col('original_affiliation')!='')\
.withColumn('aff_len', F.length(F.col('original_affiliation')))\
.filter(F.col('aff_len')>2)\
.groupby(['original_affiliation','affiliation_id']) \
.agg(F.count(F.col('affiliation_id')).alias('aff_string_counts'))

print('dedup_affs.columns: -----------------------------')
print(dedup_affs.columns)
dedup_affs.cache().count()

ror_data = spark.read.parquet(f"{rutaDatos}ror_strings.parquet").select('original_affiliation','affiliation_id')

print('ror_data.cache().count(): ---------------------------------')
print(ror_data.cache().count())

### Gathering training data

num_samples_to_get = 50

w1 = Window.partitionBy('affiliation_id')

filled_affiliations = dedup_affs \
    .join(ror_data.select('affiliation_id'), how='inner', on='affiliation_id') \
    .select('original_affiliation','affiliation_id') \
    .union(ror_data.select('original_affiliation','affiliation_id')) \
    .filter(~F.col('affiliation_id').isNull()) \
    .dropDuplicates() \
    .withColumn('random_prob', F.rand(seed=20)) \
    .withColumn('id_count', F.count(F.col('affiliation_id')).over(w1)) \
    .withColumn('scaled_count', F.lit(1)-((F.col('id_count') - F.lit(num_samples_to_get))/(F.lit(3500) - F.lit(num_samples_to_get)))) \
    .withColumn('final_prob', F.col('random_prob')*F.col('scaled_count'))

print('filled_affiliations.select(affiliation_id).distinct().count()')
print(filled_affiliations.select('affiliation_id').distinct().count())

less_than = filled_affiliations.dropDuplicates(subset=['affiliation_id']).filter(F.col('id_count') < num_samples_to_get).toPandas()
print('less_than.shape: --------------------------------------')
print(less_than.shape)

print(less_than.sample(10))

temp_df_list = []
for aff_id in less_than['affiliation_id'].unique():
    temp_df = less_than[less_than['affiliation_id']==aff_id].copy()
    help_df = temp_df.sample(num_samples_to_get - temp_df.shape[0], replace=True)
    temp_df_list.append(pd.concat([temp_df, help_df], axis=0))
less_than_df = pd.concat(temp_df_list, axis=0)

print('less_than_df.shape: --------------------------------')
print(less_than_df.shape)

# only install fsspec and s3fs
less_than_df[['original_affiliation', 'affiliation_id']].to_parquet(f"{iteration_save_path}lower_than_{num_samples_to_get}.parquet")
print('Se crea el archivo' + f"{iteration_save_path}lower_than_{num_samples_to_get}.parquet")

w1 = Window.partitionBy('affiliation_id').orderBy('random_prob')

more_than = filled_affiliations.filter(F.col('id_count') >= num_samples_to_get) \
.withColumn('row_number', F.row_number().over(w1)) \
.filter(F.col('row_number') <= num_samples_to_get+25)

print('more_than.cache().count(): ---------------------------------')
print(more_than.cache().count())

more_than.select('original_affiliation', 'affiliation_id').coalesce(1).write.mode('overwrite').parquet(f"{iteration_save_path}more_than_{num_samples_to_get}.parquet")
print('Se crea el archivo ' + f"{iteration_save_path}more_than_{num_samples_to_get}.parquet")


print('FINALIZADO OK')
