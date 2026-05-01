from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, NGram, HashingTF, IDF


def build_pipeline():
    tokenizer = Tokenizer(inputCol="lemmatized", outputCol="words")
    ngram = NGram(n=2, inputCol="words", outputCol="bigrams")
    hashingTF = HashingTF(inputCol="bigrams", outputCol="rawFeatures")
    idf = IDF(inputCol="rawFeatures", outputCol="features")

    return Pipeline(stages=[tokenizer, ngram, hashingTF, idf])
