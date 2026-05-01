import spacy
from pyspark.sql.functions import expr

nlp = spacy.load("en_core_web_sm")

# --- TEXT PREPROCESSING (pandas) ---
def preprocess(text):
    doc = nlp(text)
    return " ".join([
        token.lemma_
        for token in doc
        if not token.is_stop and token.is_alpha
    ])

# --- LABEL CLEANING (Spark) ---
def clean_labels(df):
    df = df.withColumn("hd", expr("try_cast(hd as int)"))
    df = df.withColumn("cv", expr("try_cast(cv as int)"))
    df = df.withColumn("vo", expr("try_cast(vo as int)"))

    return df.dropna(subset=["hd", "cv", "vo"])
