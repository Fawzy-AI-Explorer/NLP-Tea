# RNNs (Recurrent Neural Networks)

## Sequential Data
- Sequential data is any data where the order of elements matters.
  - Examples: Text, Videos, Speech   
- Sequence Models are designed for sequential data
  - Examples: RNNs, LSTMs, GRUs, Transformers

## Temporal vs. Spatial Data
- Temporal Data: sequences that changr over time (related to **time**)
  - Time Series : Stock prices, Weather data
  - Speech signals: (audio changes over time)
  - Video: sequence of frames over time
  - Text:  (sequence of words or characters)
- Spatial Data: Refers to data associated with spatial locations.(structure in space(2-D grid))
  - Images: Pixels arranged in a grid.
  - Video: Each video frame is essentially an image (Pixels)
  - maps: Geographic data for specific areas.

## What is RNNs ? 
- RNNs are neural networks specially designed for sequential data.
- They remember past information using a "history vector".
- Great for tasks where order and context matter (e.g. language, time series)

## Why RNN Not FC?
- FC networks:
  - Expect fixed-size input and output
    - Can’t handle variable-length sequences well 
  - Don’t remember previous inputs
    - Doesn’t retain info from earlier words/time steps. 
  - Ignore the order of inputs
    - No temporal structure. 
  - Traditional neural networks process input data without considering sequence or time-based dependencies.

- RNNs
  - Sequential Data Handling (where the order of inputs matters)
  - RNNs retain information from previous steps, making them suitable for tasks that require understanding context or history
  - Efficiency with Variable-Length Inputs: RNNs can handle variable-length sequences naturally. FCNs require fixed-length inputs
    - Translation  
  - "This is good" vs. "I can't say this is good."
    - FC treats "good" the same in both sentences.
    - RNN understands the context around "good".  

### How FC and RNN Process Video or Text:

- FC
  - dataset (Videos)
    - For video with 100 frames, each 256x256:
      - Whole video = 3D tensor (100 x 256 x 256) fed at once 
  - dataset (sentences)
    - For text:
      - Join all words into one long vector and feed at once
  - FC doesn’t see relationships between frames or words
FC diesn't take relation Between Pixels or Words , Video or Sentence Feed to the Network 
- RNNs
  - Processes data step-by-step (Sequential):
    - For video: frame-by-frame
    - For text: word-by-word
  - At each step (t) Take 2 vecctor as an input:
     1. Takes current input vector : Represent Cur Frame (t) 
     2. history vector (summary of all previous inputs) : Represent Frame 0 to Frame t-1)
  - Updates the history after each step 

### How RNNs Handle Different Sizes Input.
Sentences With n Words , Each Word Represented as vector ( len = 90 )
- Input Layer >> Number of Neurons : 90 Nodes
- Hidden Layer (RNN) >> Number of Neurons:  براحتك = History Vector Size
- Output Layer >> Number of Neurons :Depends on the Task
لي بقا مش مهم كل فيديو فيه كام فريم ؟ لانك مثلا اول فيديو فيه 25 فريم ف انت هتخش ع النتورك اول مرة باول فريم وتاني مرة ب (تاني فريم وهستوري) وهكذا مش فارقه معاك عدد الفريمز لانه كدا كدا مش هتدخلهم كلهم مرة واحده انت شغال فريم فريم انظر للكود اللي تحت لمزيد من التفاصيل .
فكر ف الموضوع انه معاك فيديو هتدخله ل RNN Layer فريم فريم ومش هتخش ع الفيديو اللي بعده غير لما يخلص خالص كل الفريمات .
- Each step depends on ALL the previous steps
  - (The k-th frame depends on all previous k-1 frames !!!! )
  - at each step we have 2 inputs only :
    - The k-th feature vector for the k-th frame
    - History vector representing the frames from 1 to k-1
- After each k-th step, the History vector will be updated to represent inputs from 1 to k !!!!!



## RNN Architecture

ال RNN Layer هيشتغل على كل Frame لوحده ويعرف يستغل ال History او ال Frames السابقة.   
لو معاك 100 frame  يعني السامبل الواحد فيه 100 Frame ==>   
هيبقى معاك 100 Vec هيدخله واحد واحد و مع كل vec داخل للنتورك بتدخل معاه vec 1 بيعبر عن ملخص كل اللي فات History      
لو انت عند ال Frame K هتدخل للنتورك ===> vec K بيعبر عن الحالى و Vec from 1 to K-1 ده ال History وطبعا كل مرة هتعمل Update لل History ده تضيف عليه ال frame  الحالي   
![image](https://github.com/user-attachments/assets/6af25a3e-68de-4f79-b16e-a7a3f9fa0db8)

```
in FC : Input x(Video/Sentence) fed at once 

a = g ( Wax . X + ba )
y = g ( Wya . a + by )
----------------------------------------
----------------------------------------

in RNNs : Work on Steps (X[0], X[1], ......., X[t])    

a(t) = g ( Wax . X(t) + Waa . a(t-1) + ba)
y(t) = g ( Wya . a(t) + by)
-----------
a(0) = 0 ===> this is the History >>>>> History Vector Size = Number of Hidden State on RNN Layer
*****
a(0) = 0
a1 = g(Wax * X1 + Waa * a0 + ba) 
y1 = g(Wya * a1 + by)
*****
a2 = g(Wax * X2 + Waa * a1 + ba)
y2 = g(Wya * a2 + by)
*****
a3 = g(Wax * X3 + Waa * a2 + ba)
   = g(Wax * X3 + Waa * (g(Wax * X2 + Waa * (g(Wax * X1 + Waa * a0 + ba)) + ba)) + ba)
Wax ==> Shared Weights Through Time
Waa  ==> Shared Weights Through Time

y3 = g(Way * a3 + b)
y3 = g(X3, X2, X1

```
![image](https://github.com/user-attachments/assets/42614b7e-1e26-4a48-868f-9307161879c1)


## RNNS Types
- Ont to One 
- One to Many
  - Image Caption   
- Many to One
  - Sentiement Analysis 
- Many to Many
  - Translation 

