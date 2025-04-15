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
`Target` => `Context`     
`my` => `name`    
`name` => `my, is`       
`is` => `name, mohammad`         
`mohammad` => `is, fawzy`         
`fawzy` => `mohammad`          
window size = 2           
`my` => `name , is`          
`name` => `my, is, mohammad`          
`is` => `my, name, mohammad, fawzy`          
`mohammad` => `name, is, fawzy`            
`fawzy` => `is, mohammad`           
     




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


# Skip-Gram

- Input Layer:
  - Number of Neurons: Equal to the number of unique words (vocabulary size).
  - Representation: One-hot encoded vector representing the target word.

- Hidden Layer:
  - Number of Neurons: Equal to the chosen embedding size.
  - Purpose: This layer learns to project the one-hot vector into a lower-dimensional space. The learned weights of this layer become the word embeddings.

- Output Layer:
  - Number of Neurons: Equal to the number of unique words (vocabulary size).
  - Representation: Produces a probability (SoftMax) of all words in the vocabulary to predict which words are context.



Vocab Size = 100 , Window Size = 1
Each Word represented in Binary Vector (Len = 100) All 0 except the index
Suppose the   
- target Has 1 in position 3 [0 0 0 1 0 0 0 0 0 0.......]
- context [0 0 1 0 0 0 0 0 0 0.......], [0 0 0 0 1 0 0 0 0 0.......]



1. One-Hot Encoding
   - Every word in the vocabulary is represented as a Binary vector
2. Defining Target and Context Words
   - Target Word: The central word for which an embedding is learned
   - Context Words: The words that surround the target word in a sentence
3. Input Layer
   - take the one-hot encoded vector for the target word
4. Hidden Layer (Embedding Layer)
   - The one-hot vector is multiplied by a weight matrix W (vocab size × embedding size).
   - Since only one element in the one-hot vector is 1, the output is simply the row of W corresponding to that word. This row becomes the word embedding for the target word.
   - The training process adjusts the weights in W so that similar words (appearing in similar contexts) end up with similar vectors.
5. Output Layer
   - The hidden layer output (the word embedding) is then passed through another weight matrix W′(embedding size × vocab size) to produce logits for every word in the vocabulary.
   - A softmax function is applied to these Logits to get a probability distribution over all words. This distribution reflects the probability of each word being a context word for the given target word.
6. Training Objective
  - The error between the predicted probabilities and the actual context words (represented as one-hot vectors) is computed.
The network uses backpropagation to adjust both weight matrices W and W′
7. Extracting the Embeddings
   - Once training is complete, the weights in the hidden layer matrix W are used as the word embeddings.
   - These embeddings capture the relationships between words based on their context in the training text.










