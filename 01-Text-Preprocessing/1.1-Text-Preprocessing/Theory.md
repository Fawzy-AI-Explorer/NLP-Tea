# Text Preprocessing

## What ... ?
- the process of cleaning and transforming raw text into a format suitable for NLP tasks

## Why ... ?
- Text data often contains noise such as punctuation, special characters, and irrelevant symbols. Preprocessing helps remove these elements.
- Different forms of words (e.g., “run,” “running,” “ran”) can convey the same meaning but appear in different forms. Preprocessing techniques like stemming and lemmatization help standardize these variations.
- raw text has Mixed cases ("Hello" , "hello") Models treat "Hello" and "hello" as different words
and more...


# How ... ?

- Lowercase
-   Converts text to lowercase ("Hello" → "hello")
- Remove URLs, mentions, hashtags
- Remove punctuation & numbers
- Tokenize
- Remove stopwords
- Lemmatization
- Spacy for Better Lemmatization
- Custom Rules (e.g., replace emojis with text ":)" → "\[smile]")
- 
