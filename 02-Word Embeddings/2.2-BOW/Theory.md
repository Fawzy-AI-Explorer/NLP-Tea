# Word Embeddings

[01- Label Encoder & One Hot Encoder](https://github.com/Fawzy-AI-Explorer/NLP-Tea/tree/main/02-Word%20Embeddings/2.1-Label%20Encoder%20and%20One%20Hot%20Encoder)
<br>

# 02 - Bag Of Words

## What is Bag of Words (BoW)?

convert text into numerical. It treats a document as an unordered collection (or "bag") of words, ignoring word order and structure. Each document is represented as a vector where each dimension corresponds to the frequency (or presence) of a word from a vocabulary(Unique Words). 



## Steps

1. Prepare your corpus  
2. Preprocessing (corpus)  
3. Create Vocabulary (unique words in the corpus)  
4. Calculate count of vocab words (histogram) in each document  
   - For Each Doc: create a vector of word counts
     - Calculate the count of each vocab word 
     - Each position in the vector corresponds to a word in the vocabulary (number of times that word appears in the document)


For documents not considered during Vocab design , they may contain some words not in vocabulary (Out of Vocab). Those words are ignored.


## Limitations:
-  No context
    -  Ignores word order, syntax, and semantic relationships 
-  High dimensionality
    -  large vocabulary (large Number of Unique Words) => 
-  Sparse data
    -  Most values are zeros
-  BoW is designed for representing entire documents (or sentences) as vectors, not individual words



              W1     W2     W3     W4     ...............Wv ==> Vocab (Unique Words)
        Doc1 [                                             ]  => len = len(vocab) = len (Unique words)                  
        Doc2 [                                             ]
        Doc3 [                                             ] 

---
---
---












