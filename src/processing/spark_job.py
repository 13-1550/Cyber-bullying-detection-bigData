from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder.appName("CyberbullyingDetection").getOrCreate()

# Load dataset (adjust path if needed)
df = spark.read.csv("data/raw/ghc-train.tsv", sep="\t", header=True)

# Show schema
df.printSchema()

# Basic processing: filter toxic comments (label = 1)
toxic_df = df.filter(df["label"] == 1)

# Show sample
toxic_df.show(5)

# Count toxic vs non-toxic
df.groupBy("label").count().show()

# Save processed output
toxic_df.write.csv("data/processed/spark_output", header=True)

# Stop Spark
spark.stop()
