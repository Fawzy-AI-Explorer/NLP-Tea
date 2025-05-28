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

<details><summary><h3>Question</h3></summary>

Answer

</details>
