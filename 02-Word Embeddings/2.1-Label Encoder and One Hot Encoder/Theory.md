# Word Embeddings

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

1.     create corpus = list of all Dos [Doc1, Doc2, Doc3, ......]
2.     preprocessing (take Doc as an input, out Tokens) List[List[str]]
3.     Build a Vocabulary (Unique Words)
         -  Combine tokens from all documents and create a set of unique words
4.     Integer Mapping
         -  Map each unique word (or category) to a unique integer    

5.     Transform the Documents
         -  Replace each word in each document with its corresponding integer according to the mapping.
6.     Post-Processing
         -  Pad sequences: Ensure all sequences have the same length.
             -  add Padding to ensure that all Docs has the same Lenght.





## 02- One Hot Encoder

Represent text as an Binary vectors.
-  The vector’s length equals the number of unique categories.
-  All elements of the vector are 0 except for one element, which is set to 1 to indicate the presence of that category.

-  Distance between two vectors of two words that are One-Hot Encoded is the same (either “2” for different words and “0” for same words)

-  High Dimensionality , length of Each Vector equal lenght voab (Unique words) (e.g. Unique=10000)
  
-  the Vector is a binaly (All 0 except one position only is 1)


   
### Steps

1.     Apply Label Encoder (Mapp Each Unique Word to Integer Value)
2.     Create Binary Vectors
        -  For each unique Word, create a binary vector all 0 except the index of the integer, (lenght = len(Vocab) = len(Unique_Words)).
3.     Transform
        -  Replace each Word with its corresponding binary vector.

'''

    Doc1: "cat sat on the mat"
    Doc2: "dog barked at the cat"
    ---------------------
    corpus = ["cat sat on the mat", "dog barked at the cat" ]
    processed_corpus = [ [ "cat", "sat", "on", "the", "mat" ], [ "dog", "barked", "at", "the", "cat" ]]
    Vocabulary (Unique Words) = ["at", "barked", "cat", "dog", "mat", "on", "sat", "the"]
    Label Encoding : 
    ["at":0, "barked":1, "cat":2, "dog":3, "mat":4, "on":5, "sat":6, "the":7]
     One-Hot Encoding :
     "at" (index 0)    : [1, 0, 0, 0, 0, 0, 0, 0]
     "barked" (index 1): [0, 1, 0, 0, 0, 0, 0, 0]
     "cat" (index 2)   : [0, 0, 1, 0, 0, 0, 0, 0]
     "dog" (index 3)   : [0, 0, 0, 1, 0, 0, 0, 0]
     "mat" (index 4)   : [0, 0, 0, 0, 1, 0, 0, 0]
     "on" (index 5)    : [0, 0, 0, 0, 0, 1, 0, 0]
     "sat" (index 6)   : [0, 0, 0, 0, 0, 0, 1, 0]
     "the" (index 7)   : [0, 0, 0, 0, 0, 0, 0, 1]
     
    Doc1 : "cat sat on the mat"
    = [ 
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0] , 
    [0, 0, 0, 0, 0, 1, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 0]    
    ]
    
   
'''




