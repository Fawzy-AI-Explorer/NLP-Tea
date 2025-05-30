# NLP Questions

## Preprocessing

<details><summary><h3>Q1. What are preprocessing steps/techniques in NLP?</h3></summary>

- Remove tags
- Case normalization
- Tokenization
- Stopword removal
- Stemming
- Lemmatization
- Punctuation removal

</details>

---

## Feature Extraction

<details><summary><h3>Q2. What are two main approaches for word embedding?</h3></summary>

- Frequency-based methods
  - One-hot encoding
  - Bag-of-words (BOW)
  - Term frequency-inverse document frequency (TF-IDF)
  - Co-occurrence matrix
- Prediction-based methods
  - Word2Vec
    - Skip-gram
    - Continuous bag of words (CBOW)
  - GloVe

</details>

<details><summary><h3>Q3. What are the steps for one-hot encoding?</h3></summary>

1. Create a vocabulary of unique words from the text.
2. Integer Encoding: Assign a unique index to each word in the vocabulary.
3. Create a binary vector for each word:
   - The length of the vector is equal to the size of the vocabulary.
   - Set the index corresponding to the word to 1, and all other indices to 0.

</details>

<details><summary><h3>Q4. What is the distance between any two vectors in one-hot and integere encoding?</h3></summary>

- The distance between any two vectors in one-hot encoding is 2 or 0.
- The distance between any two vectors in integer encoding depends on the difference in their indices.

</details>

<details><summary><h3>Q5. What is disadvantage of Bag-of-Words (BOW) model?</h3></summary>

Any information about the order or structure of the words in the document is discarded.

</details>

<details><summary><h3>Q6. What are BoW involves?</h3></summary>

- A vocabulary of known words. (AKA Vocab)
- A measure of the presence of known words in the document. (Histogram)

</details>

<details><summary><h3>Q7. What is corpus?</h3></summary>

A collection of selected documents.

</details>

<details><summary><h3>Q8. What are the steps to represent documents using BoW?</h3></summary>

1. Prepare a corpus of documents.
2. Tokenize the documents into words.
3. Apply "Stemming" or "Lemmatization" to the words.
4. Create a vocabulary of unique words from the corpus.
5. Omit stop words from the vocabulary.
6. Create a histogram for each document, counting the occurrences of each word in the vocabulary.

</details>

<details><summary><h3>Q9. What is a potential problem with using BoW?</h3></summary>

For documents not considered during vocab desing, they may contain some words not in the vocabular (out-of-vocabulary words). **Those words are ignored**.

</details>

<details><summary><h3>Q10. Why not using Words Frequencies? (Distadvantages of BoW)</h3></summary>

- Word Counts are very basic.
- Stop words (e.g., "the", "is", "and") may appear many times in the documents. Their **large counts** means **LOW discrimination** power between documents.

</details>

<details><summary><h3>Q11. What is TF-IDF Tries to do?</h3></summary>

Tries to highlight words that are more interesting:
Terms that are (1) Frequent in a document and (2) Less frequent  across documents

</details>

<details><summary><h3>Q12.What is the difference between CBOW and Skip-gram in Word2Vec?</h3></summary>

- CBOW:
  - predicts the target word from the context words.
  - Used when you have a lot of data.
  - Order of words in the context is not important
  - Faster to train.
  - Good for frequent words.
  - Architecture:
  ![alt text](assets/image.png)
- Skip-gram:
  - predicts the context words from the target word.
  - Used when you have less data.
  - Slower to train.
  - Good for infrequent words.
  - Architecture:
  ![alt text](assets/image2.png)

</details>

<details><summary><h3>Q13.What is Window Size in Word2Vec?</h3></summary>

- Window Size:
  - The number of words to consider on either side of the target word.
  - Number of context words to consider = 2 * window size.

</details>

<details><summary><h3>Q14.what is the goal of Word2Vec?</h3></summary>

- The goal of Word2Vec is to learn vector representations of words such that words with similar meanings are close together in the vector space.

- Not to predict the next word in a sentence.

</details>

<details><summary><h3>Q15.What is Negative Sampling in Word2Vec?</h3></summary>

- Negative Sampling:
  - Words that are not in the context of the target word are treated as negative samples.
  - A technique used to train Word2Vec models more efficiently.
  - it converts the multi-class classification problem into a binary classification problem.
  - it uses sigmoid instead of softmax.
  - for each target word, K negative samples are randomly selected from the vocabulary.
- Architecture:
  ![alt text](assets/image3.png)

</details>

<details><summary><h3>Q16.How many pairs of training data are generated if the window size is 2 and the sentence is "I love natural language processing" negative samples=3?</h3></summary>

- For the sentence "I love natural language processing" with a window size of 2, the pairs of training data generated would be:
  - (I, love), (I, natural)   we have 6 negative samples  all = 6 + 2 = 8 pairs of training data.
  - (love, I), (love, natural), (love, language)  we have 9 negative samples  all = 9 + 3 = 12 pairs of training data.
  - (natural, I), (natural, love), (natural, language), (natural, processing) we have 12 negative samples  all = 12 + 4 = 16 pairs of training data.
  - (language, love), (language, natural), (language, processing)  we have 9 negative samples  all = 9 + 3 = 12 pairs of training data.
  - (processing, natural), (processing, language) we have 6 negative samples   all = 6 + 2 = 8 pairs of training data.
  - Total pairs = 8 + 12 + 16 + 12 + 8 = 56 pairs of training data.

</details>

<details><summary><h3>Q17. What are the Training steps for a Negative Sampling mode?</h3></summary>

Training steps for a Negative Sampling model:

1. **Input:**
   - One input target word + one positive/negative word (one-hot encoded).

2. **Label Assignment:**
   - Positive sample → **Y = 1**
   - Negative sample → **Y = 0**

3. **Embedding:**
   - Pass both words through **separate embedding layers** (output: dense vectors of size `embed_size`).

4. **Similarity Calculation:**
   - Compute **dot product** of the two embeddings.

5. **Activation:**
   - Apply **sigmoid** to dot product → outputs a value between 0 and 1.

6. **Loss and Backpropagation:**
   - Compare sigmoid output to actual label Y.
   - Compute loss and **backpropagate** error to update weights.

7. **Repeat:**
   - Perform the above steps for all (target, context) pairs across multiple epochs.

</details>

<details><summary><h3>Q18. What is a limitation of One-Hot Encoding regarding word relationships?</h3></summary>

A limitation of One-Hot Encoding is that the distance between the one-hot encoded vectors of any two different words is the same. This means it does not capture any semantic similarity or relationship between words, only their distinct identity.

</details>

<details><summary><h3>Q19. Derive Cost Function for Skip-Gram Model</h3></summary>

- The network predicts the probability of a context word given a target word:  

```math
P(w_\text{context}|w_\text{center};\theta)=\frac{e^{W_{\text{output}_{(c)}} \cdot h}}{\sum_{i=1}^{V} e^{W_{\text{output}_{i}} \cdot h}}
```  

- Objective is to maximize the log-likelihood of the context words given the target word:

```math
J(\theta) = arg\max_\theta \log \prod_{w_\text{context} \in C(w_\text{center})} P(w_\text{context}|w_\text{center};\theta)=arg\max_\theta \log \prod_{c=1}^{C} \frac{e^{W_{\text{output}_{(c)}} \cdot h}}{\sum_{i=1}^{V} e^{W_{\text{output}_{i}} \cdot h}}
```  

- Converting objective to a loss function:  

```math
  L(\theta; w^{(t)}) = - \log \prod_{c=1}^{C} \frac{e^{W_{\text{output}_{(c)}} \cdot h}}{\sum_{i=1}^{V} e^{W_{\text{output}_{i}} \cdot h}} = -\sum_{c=1}^{C} \log \frac{e^{W_{\text{output}_{(c)}} \cdot h}}{\sum_{i=1}^{V} e^{W_{\text{output}_{i}} \cdot h}}
```

```math
  L(\theta; w^{(t)}) = - \sum \log P(w_\text{context}|w_\text{center};\theta)
```

</details>


<details><summary><h3>Q20. What is the problems with Skip-Gram?</h3></summary>

Softmax is applied to the values of the output layer, which can be computationally very expensive for large vocabularies.

</details>

<details><summary><h3>Q21. Why using sigmoid instead of softmax in Negative Sampling?</h3></summary>

Negative sampling converts multi-classification task into binary-classification task.
The new objective is to predict, for any given word-context pair (w,c), whether the word (c) 
is in the context window of the the center word (w) or not.

</details>

<details><summary><h3>Q22. How to Get “Embeddings” from Negative Sampling Model?</h3></summary>

To get word embeddings for entire vocabulary:
Extract the embeddings from  word_model embedding layer.
(use Target wordEmbedding NOT Context wordembedding)

</details>

<details><summary><h3>Q23. What are the limitations of Word2Vec?</h3></summary>

- Out-of-vocabulary (OOV) words: Words not seen during training are ignored.
- Morpohological variations: Different forms of a word (e.g., "run", "running") are treated as distinct words.

</details>

<details><summary><h3>Q24. What is the FastText Objective?</h3></summary>

Utilizing the internal structure of words to make the process of learning word embeddings more efficient.

</details>

<details><summary><h3>Q25. What is the main difference between Word2Vec and FastText?</h3></summary>

- Word2Vec representation is word-based
- FastText representation is character n-gram based.

</details>

<details><summary><h3>Q26. What is the main advantage of FastText over Word2Vec?</h3></summary>

In FastText, Representing out-of-vocab words by summing their sub-words has performance than assigning null vectors like Word2Vec.  

FastText generally has better performance than Baseline Word2Vec algorithms (CBOW and SkipGram) even with OOV.

</details>

<details><summary><h3>Q27. What are the steps for Skip-Gram Representation Of FastText Embedding?</h3></summary>

1. **Select Target and Context Words**
   - Using a sliding window of size 2C+1, where C is the context size.
2. **Select Negative Samples for each context word**
   - For one actual context word, “N” random negative words are sampled  (e.g. N=5)
3. **Sub-word generation**
   - By creating character n-grams (length 3–6) from words enclosed in angle brackets.
4. **Prepare Vectors for Target, Context, and Negative Words**
   - Target word vector: Sum of vectors of all sub-words.
5. **Train Neural Network with:** two inputs, one hidden layer and one segmoid output
   - By maximizing similarity with context words and minimizing similarity with negative samples using a neural network.
   - Architecture:
   - ![alt text](assets/image4.png)

</details>

<details><summary><h3>Q28. What is the goal of stemming and lemmatization and what is the difference between them?</h3></summary>

- **Goal**: Reduce words to a common base form.

- **Stemming**: is the process that stems or removes last few characters from a word, often leading to incorrect meanings and spelling.
  - For example, Caring becomes Car, and Cars becomes Car.
  - Used in case of large datasets where performance is an issue.
- **Lemmatization**: considers the context and converts the word to its meaningful base form.
  - For example, Caring becomes Care, and Cars becomes Car.
  - Used in case of small datasets where accuracy is an issue, because its computationally expensive.

</details>

---

## Classification

<details><summary><h3>Q1. What are two types of time-idexed data?</h3></summary>

- **Equally spaced**: Data points are collected at regular intervals (e.g., daily, monthly).
- **Ordinal time steps**: Data points are collected at irregular intervals, but the order of events is preserved (e.g., word sequences).

</details>

<details><summary><h3>Q2. Given the following architecture, what is the output shape and number of parameters of each layer?</h3>

- **Architecture**:
  - ![alt text](assets/image5.png)

</summary>

- **Answer**:
  - ![alt text](assets/image6.png)

</details>

<details><summary><h3>Q3. What is the difference between 1D, 2D, and 3D convolution?</h3></summary>

- ID CNN,
  - kernel moves in 1 direction.
  - Input and output data of ID CNN is 2 dimensional.
  - Mostly used on Time-Series data.
- 2D CNN,
  - kernel moves in 2 directions.
  - Input and output data of 2D CNN is 3 dimensional.
  - Mostly used on Image data
- 3D CNN,
  - kernel moves in 3 directions.
  - Input and output data of 3D CNN is 4 dimensional.
  - Mostly used on 3D Image data (MRI, CT Scans).

</details>

<details><summary><h3>Q4. What are the steps to apply 1D CNN for text classification?</h3></summary>

1. **Word Tokenization**:
   - Represented in a dictionary that maps each unique word to their token, {'the': 0, 'of': 1, 'so': 2, 'then': 3, 'you': 4, … }
   - Vocab: all unique words in the text.
   - Token: a unique integer assigned to each word in the vocab.
2. **Word One-Hot Encoding**:
   - ![alt text](assets/image7.png)
3. **Word Embedding**:
   - ![alt text](assets/image8.png)
4. **Convolutional Kernel**:
   - ![alt text](assets/image9.png)

</details>

<details><summary><h3>Q5. Why using Max-Pooling over Time></h3></summary>

In order to indicate the presence of these high-level features, we need a way to identify them in a vector, regardless of the location within the larger input sequence. One way to identify important features, no matter their location in a sequence, is to discard less-relevant, locational information.

![alt text](assets/image10.png)

</details>

<details><summary><h3>Q6. Draw the architecture for using multiple kernels with different scales in 1D CNN.</h3></summary>

![alt text](assets/image11.png)

</details>

<details><summary><h3>Q7. What are the Traditional Language Models?</h3></summary>

A language model computes a probability for a sequence of words: $P(w_1, ..., w_T)$

Example: Zein is playing football

```math
P(\text{Zein is playing football}) = P(w_1, w_2, w_3, w_4)\\
P(\text{Zein is playing football}) = P(w_1) * P(w_2|w_1) * P(w_3|w_1, w_2) * P(w_4|w_1, w_2, w_3)\\
P(\text{Zein is playing football}) = P(\text{Zein}) * P(\text{is}|\text{Zein}) * P(\text{playing}|\text{Zein}, \text{is}) * P(\text{football}|\text{Zein}, \text{is}, \text{playing})
```

```math
P(w_2|w_1) = \frac{\text{count}(w_1, w_2)}{\text{count}(w_1)}\\
```

</details>

<details><summary><h3>Q8. Draw RNN architecture (Folded and Unfolded)</h3></summary>

![alt text](assets/image12.png)

</details>

<details><summary><h3>Q9. Draw Unfolded RNN architecture with multiple hidden layers</h3></summary>

![alt text](assets/image13.png)

</details>

<details><summary><h3>Q10. Give 4 examples of RNN/LSTM applications specific the type of the task and draw the architecture for each.</h3></summary>

![alt text](assets/image14.png)

</details>

<details><summary><h3>Q11. Draw the **Generation** Model Architecture for RNN</h3></summary>

![alt text](assets/image15.png)

</details>

<details><summary><h3>Q12. Complete the following table</h3>

![alt text](assets/image16.png)

</summary>

**Answer**:
![alt text](assets/image17.png)

</details>

<details><summary><h3>Q13. For the following RNN architecture, Derive the Backpropagation equations (BPTT), hidden activation is `tanh` output activation is `softmax`</h3>

![alt text](assets/image18.png)

</summary>

**Answer**:

**Forawrd Calculation:**
```math
s_t = \tanh{(W s_{t-1} + U x_t)}\\
\hat{y}_t = \text{softmax}(o_t) = \text{softmax}(V s_t)\\
o_t = V s_t\\
```

**Loss and Cost Function:**
```math
E_t(y_t,\hat{y}_t) = -y_t \log(\hat{y}_t)\\
C = \sum_{t=1}^{T} L_t\\
```

**Gradients:**
```math
\frac{\partial E}{\partial V} = \sum_{t=1}^{T} \frac{\partial E_t}{\partial V},\space
\frac{\partial E}{\partial U} = \sum_{t=1}^{T} \frac{\partial E_t}{\partial U}, \space
\frac{\partial E}{\partial W} = \sum_{t=1}^{T} \frac{\partial E_t}{\partial W}
```

**Gradients by Chain Rule:**

```math
\frac{\partial E_t}{\partial V} = \frac{\partial E_t}{\partial \hat{y}_t} \cdot \frac{\partial \hat{y}_t}{\partial o_t} \cdot \frac{\partial o_t}{\partial V}\ (1)\\

\frac{\partial E_t}{\partial U} = \frac{\partial E_t}{\partial \hat{y}_t} \cdot \frac{\partial \hat{y}_t}{\partial s_t} \cdot \frac{\partial s_t}{\partial U}\\

\frac{\partial s_t}{\partial U} = \sum_{t'=1}^{T} \frac{\partial s_{t}}{\partial s_t'} \cdot \frac{\partial s_t'}{\partial U}\\

\frac{\partial E_t}{\partial U} = \sum_{t'=1}^{T} \frac{\partial E_t}{\partial \hat{y}_t} \cdot \frac{\partial \hat{y}_t}{\partial s_t} \cdot \frac{\partial s_{t}}{\partial s_t'} \cdot \frac{\partial s_t'}{\partial U}\ (2)\\

\frac{\partial E_t}{\partial W} = \frac{\partial E_t}{\partial \hat{y}_t} \cdot \frac{\partial \hat{y}_t}{\partial s_t} \cdot \frac{\partial s_t}{\partial W}\\

\frac{\partial s_t}{\partial W} = \sum_{t'=1}^{T} \frac{\partial s_{t}}{\partial s_t'} \cdot \frac{\partial s_t'}{\partial W}\\

\frac{\partial E_t}{\partial W} = \sum_{t'=1}^{T} \frac{\partial E_t}{\partial \hat{y}_t} \cdot \frac{\partial \hat{y}_t}{\partial s_t} \cdot \frac{\partial s_{t}}{\partial s_t'} \cdot \frac{\partial s_t'}{\partial W}\ (3)\\

```

</details>

<details><summary><h3>Q14. What is the problems with RNNs?</h3></summary>

Error Gradients pass through nonlinearity every step  
Saturation at both ends ==> zero gradient  
Vanishing completely after a few time steps.  

</details>

<details><summary><h3>Q15. What is the problem with tanh activation in RNNs?</h3></summary>

- The **derivative of tanh** ranges between **0 and 1**.

- When the **tanh derivative is small**, the gradients **shrink** exponentially over time steps $\rightarrow$ this leads to the **Vanishing Gradient Problem**.

- If the gradients **grow exponentially**, it leads to the **Exploding Gradient Problem** — this is **not** due to the tanh function itself, but rather due to how gradients accumulate through time.

</details>
