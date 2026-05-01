from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder.appName("CyberbullyingDetection").getOrCreate()

df = spark.read.csv("data/raw/ghc-train.tsv", sep="\t", header=True)
df.printSchema()
toxic_df = df.filter(df["label"] == 1)
toxic_df.show(5)


df.groupBy("label").count().show()
toxic_df.write.csv("data/processed/spark_output", header=True)

# Stop Spark
spark.stop()
