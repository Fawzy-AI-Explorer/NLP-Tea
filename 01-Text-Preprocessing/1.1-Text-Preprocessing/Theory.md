# Text Preprocessing

# What ... ?
- the process of cleaning and transforming raw text into a format suitable for NLP tasks

# Why ... ?
- Text data often contains noise such as punctuation, special characters, and irrelevant symbols. Preprocessing helps remove these elements.
- Different forms of words (e.g., “run,” “running,” “ran”) can convey the same meaning but appear in different forms. Preprocessing techniques like stemming and lemmatization help standardize these variations.
- raw text has Mixed cases ("Hello" , "hello") Models treat "Hello" and "hello" as different words.
and more...

# When ... ? 




# How ... ?

## Lowercase
Converts text to lowercase ("Hello WORLD" =>>> "hello world")
### Apply for:

gjkbnkgjnb 
  - hhh
    
## Remove URLs, mentions, hashtags
  - Deletes symbols like !@#,. and urls.
  - ⚠️ don't Remove If URLs/hashtags carry meaning (trend analysis)

  
## Remove punctuation & numbers % White Spaces.
  - Deletes noise like . , ! ? ) : " 123
  -  ⚠️ don't Remove If punctuation carries emotion, number-sensitive 
    - emotion detection : "Sad :(":
    - math problems

## Tokenize
  - Splits text into words or tokens ("I love NLP" → ["I", "love", "NLP"]) 
  
## Remove stopwords
  - Deletes (Stop Words) common words ("is", "the", "and").
  - ⚠️ don't Remove If stop words carries Informations 
    -  Sentiment analysis ("not", "never" are stopwords but means negation)
   
      
## Stemming & Lemmatization
  - return Word Base ("playing" => Play)
  - ⚠️ don't Use
    - generative tasks : Summarization or translation. 
  
## Spacy for Better Lemmatization
  
## Custom Rules
  - replace emojis with text "🙂" → "[smile]") Social media sentiment, reviews analysis



















  
 
