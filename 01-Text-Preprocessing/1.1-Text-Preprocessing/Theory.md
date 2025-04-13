# Text Preprocessing

# What ... ?
-  the process of cleaning and transforming raw text into a format suitable for NLP tasks.
-  first step of NLP projects.
  
# Why ... ?
-  Text data often contains noise such as punctuation, special characters, and irrelevant symbols. Preprocessing helps remove these elements.
-  Different forms of words (e.g., “run,” “running,” “ran”) can convey the same meaning but appear in different forms. Preprocessing techniques like stemming and lemmatization help standardize these variations.
-  raw text has Mixed cases ("Hello" , "hello") Models treat "Hello" and "hello" as different words.
and more...

# When ... ? 




# How ... ?

## Lowercase
Converts text to lowercase ("Hello WORLD" =>>> "hello world")
### Apply & Avoid for:
apply If the case (Capital ar lower) does not contain information.
- Search engines (to normalize queries)
- If your goal is just to classify
  - Sentiment analysis, Spam Detection, Topic Classification (NLP, nlp) are Same
Avoid : <br>
- Machine translation
  
Chat GPT Said:  <br>
If you're not sure, just ask: <br>
         || “Does capitalization change the meaning in my task?” <br>
If no, lowercase away. If yes, preserve it. <br>

## Remove URLs, mentions, hashtags
Deletes symbols like !@#,. and urls.
### Apply & Avoid for:
- Apply for : Social media analysis, Topic modeling
- Avoid for: If URLs/hashtags carry meaning (trend analysis)

  
## Remove punctuation & numbers % White Spaces.
  - Deletes noise like . , ! ? ) : " 123
### Apply & Avoid for: 
- Apply for : Sentiment analysis (if numbers are irrelevant), Document classification
- Avoid : If punctuation carries emotion, number-sensitive 
    - emotion detection : "Sad :(":
    - math problems
    - Financial/medical texts ("COVID-19")

## Tokenize
Splits text into words or tokens ("I love NLP" → ["I", "love", "NLP"]) 
  
## Remove stopwords
Deletes (Stop Words) common words ("is", "the", "and").
### Apply & Avoid for:
- Apply for : Topic modeling
- Avoid : If stop words carries Informations 
    -  Sentiment analysis ("not", "never" are stopwords but means negation)
    -  Machine translation (grammar depends on stopwords)
   
      
## Stemming & Lemmatization
  - return Word Base ("playing" => Play)
### Apply & Avoid for:
- Apply for : Spam detection, Search engines
- Avoid for : generative tasks (Summarization or translation) 
  
## Custom Rules
  - replace emojis with text "🙂" → "[smile]") Social media sentiment, reviews analysis





Text preprocessing is task-specific
there's no one-size-fits-all. 
The preprocessing steps you choose should always depend on:

- NLP Task
  - Sentiment Analysis
    - Lowercasing, remove URLs, emojis to text
    - Avoid removing negations ("not") or emojis if they carry sentiment 
  - Topic Classification
    - Lowercasing, stopword removal, stemming/lemmatizing 
  - Machine Translation
    - keep sentence structure
    - Avoid remova punctuation, stopwords
  - Text Generation (GPT)
    - Avoid changing text
      
- Model
  -  Traditional ML (SVM, Regression)
    -  advanced: lowercase, stopwords, stemming  
  -  Transformers (BERT)
    -  minimal cleaning
 
- Dataset
  -  Tweets
  -  Product reviews
  -  Scientific texts


chat GPT Said : <br>
Always Ask Yourself <br>
Before preprocessing, ask: <br>
- What is the goal of my task?
- Will this step remove or distort useful information?
- What model am I using — does it need clean or natural text?



### Stemming & 

- the process of reducing infected words to their stem
- the process of removing the last few characters of a given word, to obtain a shorter form, even if that form doesn’t have any meaning in machine learning.

![image](https://github.com/user-attachments/assets/8594aa9d-4acb-4930-8ca0-3e3c5b59e3e9)









  
 
