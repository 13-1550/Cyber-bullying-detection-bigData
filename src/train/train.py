
import pandas as pd
from pyspark.sql import SparkSession

from preprocessing import preprocess, clean_labels
from pipeline import build_pipeline

# _____Load data_______
train = pd.read_csv("data/train_sample.csv")

# ____spaCy _______
train["lemmatized"] = train["text"].apply(preprocess)

# ______Spark____
spark = SparkSession.builder.appName("Cyberbullying").getOrCreate()
train_spark = spark.createDataFrame(train)

# ______Clean labels_____
train_spark = clean_labels(train_spark)

# ____pipeline____
pipeline = build_pipeline()

# ____Fit and transform___
pipeline_model = pipeline.fit(train_spark)
train_spark = pipeline_model.transform(train_spark)

# ____ Train model______
from pyspark.ml.classification import LogisticRegression

lr_hd = LogisticRegression(featuresCol="features", labelCol="hd")
model_hd = lr_hd.fit(train_spark)

lr_vo = LogisticRegression(featuresCol="features", labelCol="vo")
model_vo = lr_vo.fit(train_spark)

lr_cv = LogisticRegression(featuresCol="features", labelCol="cv")
model_cv = lr_cv.fit(train_spark)

# Save  
model_hd.save("models/model_hd")
model_vo.save("models/model_vo")
model_cv.save("models/model_cv")

print("Training complete ✅")
