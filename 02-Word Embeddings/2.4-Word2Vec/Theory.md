# Word Embeddings

[01- Label Encoder & One Hot Encoder](https://github.com/Fawzy-AI-Explorer/NLP-Tea/tree/main/02-Word%20Embeddings/2.1-Label%20Encoder%20and%20One%20Hot%20Encoder)   

[02 - BOW](https://github.com/Fawzy-AI-Explorer/NLP-Tea/tree/main/02-Word%20Embeddings/2.2-BOW)   

[03 - TF-IDF](https://github.com/Fawzy-AI-Explorer/NLP-Tea/tree/main/02-Word%20Embeddings/2.3-TF_IDF)   

## 04 - Word2Vec

Word2Vec is a neural network-based method that learns to represent words as vectors in a continuous vector (word embeddings). where words with similar meanings have similar vectors. Word2Vec provides a way to capture the semantic relationships between words through neural networks.  

Map each word to a dense vector such that words with common contexts in the corpus have similar vector representations.  


- Dimensionality Reduction:
  - Instead of representing words as one-hot vectors (High Dimension len vec = len(Vocab) = Number of unique words and sparse), Word2Vec produces dense vectors, low-dimensional vectors.  
- Semantic:
  - By Using `Context Words` Based on `window size` 
- Computational Efficiency:
  - By Using `Negative Sampling Method`  

## Goal

Not predict Context But to Leaern Vector representation of target words  
By predicting or using the context Words, Word2Vec `learns` the structure of the language.
The main objective of Word2Vec is to learn word embeddings that:
- Capture Semantic Relationships: Words with similar meanings are represented by similar vectors.
- 
## Target, Context, Negative sampling

- Target Word:
  - in a particular step: It is the `center` word that the model wants to learn a good representation for.
  - Each word will be a target in a specific step  
- Context Words:
  - The words surrounding the target word within window size.

`my name is mohammad fawzy`                
window size = 1    
my => name    
name => my, is       
is => name, mohammad         
mohammad => is, fawzy         
fawzy => mohammad          
window size = 2           
my => name , is          
name => my, is, mohammad          
is => my, name, mohammad, fawzy          
mohammad => name, is, fawzy            
fawzy => is, mohammad           
     




## How Does Word2Vec Work?  

Word2Vec Uses a shallow neural network that consists of an input layer, a hidden layer, and an output layer.  

- Use One Hot Encoding
- defining target words and for each one define it's Context words

- One Input Layer  (Number of Neurons = Number of unique words = len Vocab)
- One Hidden Layer (Embedding Size)
- One output Layer (Number of Neurons = Number of unique words = len Vocab)


## Types of Word2Vec
1. Continuous Bag of Words (CBOW):
   - Given the `Context` words, Predict `Target` word
   - works well with large datasets and It is computationally more efficient.
2. Skip-Gram:
   -  Given the `Target` word, Predict `Context` words
   -  Works well with smaller datasets and is particularly good at capturing rare words.
















