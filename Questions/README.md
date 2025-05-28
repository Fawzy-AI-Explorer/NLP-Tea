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

  * Pass both words through **separate embedding layers** (output: dense vectors of size `embed_size`).

4. **Similarity Calculation:**

  * Compute **dot product** of the two embeddings.

5. **Activation:**

  * Apply **sigmoid** to dot product → outputs a value between 0 and 1.

6. **Loss and Backpropagation:**

  * Compare sigmoid output to actual label Y.
  * Compute loss and **backpropagate** error to update weights.

7. **Repeat:**

  * Perform the above steps for all (target, context) pairs across multiple epochs.

</details>
