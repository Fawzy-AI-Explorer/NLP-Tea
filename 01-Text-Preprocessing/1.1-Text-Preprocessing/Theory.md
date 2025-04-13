# Text Preprocessing

# What ... ?
-  the process of cleaning and transforming raw text into a format suitable for NLP tasks
-  first step of NLP projects
  
# Why ... ?
-  Text data often contains noise such as punctuation, special characters, and irrelevant symbols. Preprocessing helps remove these elements
-  Different forms of words (e.g., “run,” “running,” “ran”) can convey the same meaning but appear in different forms. Preprocessing techniques like stemming and lemmatization help standardize these variations
-  raw text has Mixed cases ("Hello" , "hello") Models treat "Hello" and "hello" as different words
and more...

# How ... ?

## Lowercase
Converts text to lowercase ("Hello WORLD" =>>> "hello world")
#### Apply & Avoid for:
-  apply If the case (Capital or lower) does not contain information
    -  Search engines (to normalize queries)
    -  If your goal is just to classify
      -  Sentiment analysis, Spam Detection, Topic Classification (NLP, nlp) are Same
-  Avoid : <br>
    -  Machine translation
    -  POS (Parts-of-speech tagging (like noun, verb, adjective))
      
Chat GPT Said:  <br>
If you're not sure, just ask: <br>
         || “Does capitalization change the meaning in my task?” || <br>
If no, lowercase away. If yes, preserve it <br>

## Remove URLs, mentions, hashtags
Deletes symbols like !@#,. and urls
### Apply & Avoid for:
- Apply for : Social media analysis, Topic modeling
- Avoid for: If URLs/hashtags carry meaning (trend analysis)

## Remove punctuation & numbers & White Spaces
  - Deletes noise like . , ! ? ) : " 123
#### Apply & Avoid for: 
- Apply for : Sentiment analysis (if numbers are irrelevant), Document classification
- Avoid : If punctuation carries emotion, number-sensitive
    - emotion detection : "Sad :("
    - math problems
    - Financial/medical texts ("COVID-19")

## Tokenize
Splits text into words or tokens ("I love NLP" → ["I", "love", "NLP"]) 
  
## Remove stopwords
Deletes (Stop Words) common words ("is", "the", "and").
#### Apply & Avoid for:
- Apply for : Topic modeling
- Avoid : If stop words carries Informations 
    -  Sentiment analysis ("not", "never" are stopwords but means negation)
    -  Machine translation (stopwords are Important)
   
      
## Stemming & Lemmatization
  - return Word Base ("playing" => Play)
#### Apply & Avoid for:
- Apply for : Spam detection, Search engines, Sentiment analysis
- Avoid for : generative tasks (Summarization or translation) 
  
## Custom Rules
  - replace emojis with text "🙂" → "[smile]") Social media sentiment, reviews analysis




---
Text preprocessing is task-specific <Br> 
The preprocessing steps you choose should always depend on: <Br>

-  NLP Task
    -  Sentiment Analysis
      -  Lowercasing, remove URLs, emojis to text
      -  Avoid removing negations ("not") or emojis 
  -  Topic Classification
      -  Lowercasing, stopword removal, stemming/lemmatizing 
  -  Machine Translation
      -  keep sentence structure
      -  Avoid remove punctuation, stopwords
  -  Text Generation (GPT)
      -  Avoid changing text
      
-  Model
    -  Traditional ML (SVM, Regression)
      -  advanced: lowercase, stopwords, stemming  
    -  Transformers (BERT)
      -  minimal cleaning
 
- Dataset
    -  Tweets
    -  Product reviews
    -  Scientific texts

---
chat GPT Said : <br>
Always Ask Yourself <br>
Before preprocessing, ask: <br>
-  What is the goal of my task?
-  Will this step remove or distort useful information?
-  What model am I using — does it need clean or natural text?
---


## Stemming & Lemmatization

The goal of both stemming and lemmatization is to reduce:

-  inflectional forms and derivationally related forms of a word to a common base form

  
![image](https://github.com/user-attachments/assets/5e647b23-f61d-4a14-b1b4-da60ca14137c)

#### Stemming

-  the process of reducing infected words to their stem (removing common affixes (prefixes, suffixes) from words)
-  the process of removing the last few characters of a given word, to obtain a shorter form, even if that form doesn’t have any meaning in machine learning.
-  rule Based Algorithm
  
![image](https://github.com/user-attachments/assets/8594aa9d-4acb-4930-8ca0-3e3c5b59e3e9)




#### Lemmatization

The purpose of lemmatization is same as that of stemming but overcomes the drawbacks of stemming <br>
use of a vocabulary and morphological analysis of words. <br>

the token saw <br>
-  stemming might return just s, (remove aw)
-  lemmatization would attempt to return either see or saw
  -  depending on whether the use of the token was as a verb or a noun.


![image](https://github.com/user-attachments/assets/faca7b47-8096-45e8-8b11-0b7025c81bbe)


- Tokenization :
- POS Tagging: Parts-of-speech tagging (like noun, verb, adjective, etc.)
- Lemmatization:
  -  Simple dictionary lookup. This works well for straightforward inflected forms,
  -  Hand-crafted rule based system
  -  Rules learned automatically from an annotated corpus.






-  Stemming: Faster, but may create Wrong root for words and lose meaning. This is known as "over stemming."

-  Lemmatization: slower, More accurate, preserves meaning and grammatical function.








 
