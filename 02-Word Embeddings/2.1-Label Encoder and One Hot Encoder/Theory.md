# Word Embeddings

## What is Word Embedding ?

Inputs to Machine learning algorithms are Numbers (Scalars, Vectors). <br>
Text must be converted into vectors.<br>

a way of representing words as vectors in a multi-dimensional space, where the distance between vectors reflect the similarity and relationships between the words.<br>

representing words in a way that machines can understand. <br>

There are two main Approaches for word embedding:
-  Frequency Based Embedding
    -  Label (integer) Encoding 
    -  One-Hot encoded vector.
    -  Bag of Word (BOW) Count Vector.
    -  Term Frequency- Inverse Document frequency (TF-IDF) Vector.  
-  Prediction Based Embedding
    -  Word2Vec
        -  CBOW
        -  Skip Gram
            -  Negative Sampling  
    -  Fast Text    



## 01- Label Encoder

### What It Is

Represent text data as an integr values (mapping each unique Word to a unique integer (scalar) )

Used more in classical ML Algorithms (Structure Data) Features with Ordinal Characteristics ("small", "medium", "large")

Suitable for tree-based models (e.g., Decision Trees, Random Forests) that do not assume ordinal relationships.

Limitations  : 

-  Lack of Semantic Information:
  
    -  Since each word is mapped to a single integer, the numeric distance between two encoded words depends solely on the integer values of the two given words, not on the semantic similarity between the words.
      
-  Unsuitable for NLP
    -  we want representations that capture the meaning and relationships between words. Label encoding fails to capture semantic and contextual information because it encodes each word independently as a scalar.
       
### Steps

Giving Data (Doc1, Doc2, Doc3, ......)

1.     create corpus = list of all Dos
2.     preprocessing (take Doc as an input, out Tokens) List[List[str]]
3.     Build a Vocabulary (Unique Words)
         -    Combine tokens from all documents and create a set of unique words
4.     Integer Mapping
         -    Map each unique word (or category) to a unique integer    

5.     Transform the Documents
         -    Replace each word in each document with its corresponding integer according to the mapping.
6.     Post-Processing
         -    Pad sequences: Ensure all sequences have the same length.
             -    add Padding to ensure that all Docs has the same Lenght.





## 02- One Hot Encoder

Represent text as an Binary vectors.

Distance between two vectors of two words that are One-Hot Encoded is the same (either “2” for different words and “0” for same words)



1. gbngh
2. fvfvrf
3. fdvcrfdv
4. vddv a
    4.1. fvmifv m

   






