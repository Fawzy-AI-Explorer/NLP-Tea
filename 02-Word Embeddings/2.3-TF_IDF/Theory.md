# Word Embeddings

[01- Label Encoder & One Hot Encoder](https://github.com/Fawzy-AI-Explorer/NLP-Tea/tree/main/02-Word%20Embeddings/2.1-Label%20Encoder%20and%20One%20Hot%20Encoder)  

[02 - BOW](https://github.com/Fawzy-AI-Explorer/NLP-Tea/tree/main/02-Word%20Embeddings/2.2-BOW)  


# 03 - TF-IDF

## What is TF-IDF?
TF-IDF (Term Frequency–Inverse Document Frequency) is a numerical statistic that reflects how important a word is to a document in a collection (corpus).

Terms like ` “the”, “on”, “at” ` may appear many times in the documents. their large counts means LOW discrimination power between documents.  




- Term Frequency (TF):    
how often a given word appears within a document, term importance within a single document.
  - `TF(t,d) = f(t,d) / SUM(f(t,d))`
  -  `f(t,d)` =>> Number of times term t appears in document d
  -  `SUM(f(t,d))` =>>>  total number of terms in d = len(d)  
Same term has different TF values in different documents accoarding to How many times appears in this document.  
Term `t` appears 5 times in doc1 and 90 in doc2    
  
- Document Frequency (DF):
  - How many documents that a given term appear in it  
  - `DF(t) = number of documents where the term "t" appears`  
term `t` appears in 10 DOCS 
- Inverse Document Frequency (IDF):
  - down scales words that appear a lot across documents.
  - `IDF(t) = N/n`
  - `N` =>> total number of documents
  - `n` =>> number of documents where the term "t" appears  
term `t` appears in 1 DOCS over 80 Docs  `DF = 1` `IDF = 80/1 = 80`  
    - Low DF => High IDF  
term `t` appears in 80 DOCS over 80 Docs `DF = 80` `IDF = 80/80 = 1`  
    - High DF => Low IDF  

- TF-IDF
  - highlight words that are `Frequent in a document` (High TF(t,d)) and `Less frequent across documents` (High IDF(t) = Low DF(t))   
  - `TF-IDF = TF * IDF`

A high weight in tf–idf is reached by:  
- a high term frequency (in the given document)     
- a low document frequency of the term in the whole collection of documents (high Inverse
Document Frequency)  

## Steps
1. Corpus : A list of text documents.
2. Vocabulary : Unique Words
3. Calculate Term Frequency (TF) (BOW)
   - For Each Word (t) in Vocab  :
     - For Each Doc (d) in Corpus
        - Calc TF(t,d)  
4. Calculate Inverse Document Frequency (IDF)
   -  For Each Word (t) in Vocab  :
     -  Clac IDF(t) = Log(N/n)  
5. Construct TF-IDF Matrix
   - Rows : Docs
   - Cols : Vocab terms
 


- For Each Word (t) in Vocab  :
     - Clac IDF(t) 
     - For Each Doc (d) in Corpus
        - Calc TF(t,d)
        - Calc TF-IDF(t,d) = TF(t,d) * IDF(t)  



![image](https://github.com/user-attachments/assets/0be29ba1-2fc2-4fce-8aea-4600a827fcdd)
![image](https://github.com/user-attachments/assets/5f2f708e-6090-4198-827e-0019315d7b45)
![image](https://github.com/user-attachments/assets/7c1fc473-17c3-4755-80b4-20d5eb9a5301)
![image](https://github.com/user-attachments/assets/cfe56ff0-3cfa-4185-85fa-489b00008f2a)
![image](https://github.com/user-attachments/assets/248c97ce-197b-4974-9900-7bdf3f97d9b7)
![image](https://github.com/user-attachments/assets/5001a6ed-f0a6-48f7-814f-9bbee9bf3301)





## Limitations of TF-IDF

- No semantic
  - TF-IDF treats words independently, so it doesn't capture meaning or word order. 
- Sparse
  - For large vocabularies, TF-IDF creates large sparse matrices. 
- OOV
  - Can't handle words that weren’t seen during training 






