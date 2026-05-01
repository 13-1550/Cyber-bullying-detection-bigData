import spacy
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType

nlp = spacy.load("en_core_web_sm")

def lemmatize_spacy(words):
    doc = nlp(" ".join(words))
    return [token.lemma_ for token in doc]
def preprocess(text):
    doc = nlp(text)
    return " ".join([
        token.lemma_
        for token in doc
        if not token.is_stop and token.is_alpha
    ])
lemma_udf = udf(lemmatize_spacy, ArrayType(StringType()))

train['lemmatized'] = train['text'].apply(preprocess)
