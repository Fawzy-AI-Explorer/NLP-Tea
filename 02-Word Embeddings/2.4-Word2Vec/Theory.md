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
  
## Target, Context
![image](https://github.com/user-attachments/assets/425a0cf4-68c1-476c-9061-fa13ae335f18)

- Target Word:
  - in a particular step: It is the `center` word that the model wants to learn a good representation for.
  - Each word will be a target in a specific step  
- Context Words:
  - The words surrounding the target word within window size.
![image](https://github.com/user-attachments/assets/d2155ea2-aadc-41f1-9d31-ab0e32ab1a47)

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
   - ![image](https://github.com/user-attachments/assets/9ef1f8ad-df42-4b6a-b7c1-068db91398d5)

2. Skip-Gram:
   -  Given the `Target` word, Predict `Context` words
   -  Works well with smaller datasets and is particularly good at capturing rare words.
   - ![image](https://github.com/user-attachments/assets/98926dee-066c-4c9a-8566-a54724f4058e)






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
  - maximize the probability of all context words together, given a center word
  - goal is not to predict context words, but to learn vector representation of words, It just happens that predicting context words
![image](https://github.com/user-attachments/assets/24b44754-51d4-407e-af57-5936b4795840)



Vocab Size = 100 , Window Size = 1   
Each Word represented in Binary Vector (Len = 100) All 0 except the index  
Suppose the     
- target  
   - Has 1 in position 3 [0 0 0 1 0 0 0 0 0 0.......]  
- context      
   - Has 1 in position 2 [0 0 1 0 0 0 0 0 0 0.......]    
   - Has 1 in position 4 [0 0 0 0 1 0 0 0 0 0.......]    

1. One-Hot Encoding (Neurons = Voc size)     

2. Defining Target and Context Words

3. Input Layer
   - Feed the Target Word (one-hot encoded vector for the target word)

4. Hidden Layer (Embedding Layer)
   - The one-hot vector is multiplied by a weight matrix W (vocab size × embedding size).
   - Since only one element in the one-hot vector is 1, the output is simply the row of W corresponding to that word. This row becomes the word embedding for the target word.
   - The training process adjusts the weights in W so that similar words (appearing in similar contexts) end up with similar vectors.
   - ![image](https://github.com/user-attachments/assets/635aa4dd-c693-40a6-8894-6c2c76d5d004)


5. Output Layer
   - The hidden layer output (the word embedding) is then passed through another weight matrix W′(embedding size × vocab size) to produce logits for every word in the vocabulary.
   - A softmax function is applied to these Logits to get a probability distribution over all words. This distribution reflects the probability of each word being a context word for the given target word.
   - we want to maximize the probability of all context words together, given a center word
   - compute The error between the predicted probabilities and the actual context words represented as one-hot vectors (Sum all context in one vextor).
   - [0 0 1 0 1 0 0 0 0 0.......] 
   - The network uses backpropagation to adjust both weight matrices W and W′
6. Extracting the Embeddings
   - Once training is complete, the weights in the hidden layer matrix W are used as the word embeddings.
   - These embeddings capture the relationships between words based on their context in the training text.

## Negative Sampling 

Problems with Skip-Gram   
![image](https://github.com/user-attachments/assets/5960d9ba-3be6-4486-b282-94d0283395d0)

Softmax is computationally very expensive, as it requires scanning through the entire
output to compute the probability distribution of all Vocab (V) words, Vocab size may be millions or more  

Multi class classification Problems Number of classes = V = 10,000 Classes   
we want to convert from multi class classification (Soft max) to Binary classification (Sigmoid)   

Negative Sampling:  
For each training sample define:  
- Conyext Word 
- Context Words (Positive Context Sample Cpos)
   - For each Context Word
   - K Number of Words not in the context (Negative Samples).
The new objective is to predict, for any given target-word pair, whether the word is in the context words or not.
Give the Network Two Words => It predict 1 if target-Context, 0 If target-Negative
this is a Binary Classification

![image](https://github.com/user-attachments/assets/a7e7b459-1160-43a7-ab3d-1b2fc7eac1dd)







