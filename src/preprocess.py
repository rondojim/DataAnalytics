import re
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from collections import Counter
from IPython.display import display
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd


def remove_html(text):
    return re.sub("<.*?>", "", text)

def clean_html(df, modify_columns):
    for col in modify_columns:
        df[col] = df[col].apply(remove_html)
    return df

def clean_urls(df, modify_columns):
    for col in modify_columns:
        df[col] = df[col].str.replace(r"http\S+", "", regex=True)
        df[col] = df[col].str.replace(r"www\.\S+", "", regex=True)
    return df

def clean_numbers(df, modify_columns):
    for col in modify_columns:
        df[col] = df[col].str.replace(r"\d+", "", regex=True)
    return df

def clean_punctuation(df, modify_columns):
    for col in modify_columns:
        df[col] = df[col].str.replace(r"[^\w\s]+", "", regex=True)
    return df

def clean_uppercase(df, modify_columns):
    for col in modify_columns:
        df[col] = df[col].str.lower()
    return df

def clean_tokenize(df, modify_columns):
    for col in modify_columns:
        df[col] = df[col].apply(word_tokenize)
    return df

def clean_stopwords(df, modify_columns):
    stop_words = set(stopwords.words("english"))
    for col in modify_columns:
        df[col] = df[col].apply(lambda words: ' '.join([w for w in words if w not in stop_words]))
    return df

def lemmatize_with_pos_tags(word, pos):
    if not pos:
        return word
    pos_mapping = {
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
        "J": wordnet.ADJ,
    }
    wordnet_pos = pos_mapping.get(pos[0], wordnet.NOUN)
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word, pos=wordnet_pos)

def lemmatize_cell(cell):
    pos_tags = pos_tag(cell)
    lemmatized_words = [lemmatize_with_pos_tags(word, pos) for word, pos in pos_tags]
    return lemmatized_words

def clean_lemmatize(df, modify_columns):
    for col in modify_columns:
        df[col] = df[col].apply(lambda cell: ' '.join(lemmatize_cell(word_tokenize(cell))))
    return df

def remove_single_char_words(text):
    return ' '.join([word for word in text.split() if len(word) > 1])

def clean_single_char_words(df, modify_columns):
    for col in modify_columns:
        df[col] = df[col].apply(remove_single_char_words)
    return df

def remove_common_rare_words(text_tokens, word_freq, min_df): 
    return ' '.join(word for word in text_tokens if word_freq[word] >= min_df)

def clean_common_rare_words(df, modify_columns, min_df):
    for col in modify_columns:
        tokenized_texts = df[col].apply(word_tokenize)
        word_freq = Counter(word for tokens in tokenized_texts for word in tokens)

        df[col] = tokenized_texts.apply(lambda tokens: remove_common_rare_words(tokens, word_freq, min_df)) 
        
    return df

def transform(df, modify_columns):
    df = clean_html(df, modify_columns)
    print("HTML clean done")

    df = clean_numbers(df, modify_columns)
    print("Numbers clean done")

    df = clean_urls(df, modify_columns)
    print("URLs clean done")

    df = clean_punctuation(df, modify_columns)
    print("Punctation clean done")

    df = clean_uppercase(df, modify_columns)
    print("Uppercase clean done")

    df = clean_tokenize(df, modify_columns)
    print("Tokenize done")

    #nltk.download('stopwords')
    df = clean_stopwords(df, modify_columns)
    print("Stopwords clean done")

    df = clean_lemmatize(df, modify_columns)
    print("Lemmatize done")

    df = clean_single_char_words(df, modify_columns)
    print("Clean single char words done")

    return df
