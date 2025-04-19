# RNNs

## Sequential Data
- Sequential data : refers to any data where the order of elements matters.
  - Text, Videoes   
- Sequence Models : Models designed for sequential data
  - RNNs, LSTMs, GRUs, Transformers

## Temporal vs. Spatial Data
- Temporal Data: Involves sequences that vary over time. (زمان)
  - Time Series : Stock prices: time. Weather data: during the day.
  - Speech signals: Audio waveforms over time.
  - Video: sequence of frames over time.
  - Text: Words or characters appear in a specific order.
- Spatial Data: Refers to data associated with spatial locations.(مواقع مكانية)
  - Images: Pixels arranged in a grid.
  - Video: Each video frame is essentially an image (Pixels)
  - maps: Geographic data for specific areas.

## What is RNNs ? 
Recurrent Neural Networks (RNNs) are a class of neural networks designed for processing sequential data. They capture temporal dependencies and maintain a "memory" of previous inputs.


## Why RNN Not FC?

- Fixed-size input/output: Can’t process variable-length sequences.
- No memory of past: Doesn’t retain info from earlier words/time steps.
- Order doesn’t matter: No temporal structure.
- Traditional neural networks process input data without considering sequence or time-based dependencies.

- Sequential Data Handling: (where the order of inputs matters)
- RNNs retain information from previous steps, making them suitable for tasks that require understanding context or history
- Efficiency with Variable-Length Inputs: RNNs can handle variable-length sequences naturally. FCNs require fixed-length inputs
- This is Good. _ I can't say This is Good.
الاتنين Good لكن ال Context مختلف تمامًا FC هتفشل طبعا هي بالنسبة ليها كلمة Good واحده ف الاتنين إنما RNN هتفهم ال Context
عندك حجم ال Input, output مش ثابت دايما بيتغير مثلا في الترجمه ال FC مش هتنفع انما ال RNN شغال.



- FC
  - dataset (Videos)
    - Each video => 100 Frames, Each Frame (image) $256 * 256$
    - For Each Video : 3D Tensor $(100 * 256 * 256 )$
  - dataset (sentences)
    - each sentence 30 Words ===> concatenate as a single string.
FC diesn't take relation Between Pixels or Words , Video or Sentence Feed to the Network 
هو مش هياخد علاقة الكلمات ببعض ولا الفريمات ببعض. الفيديو او الجملة هتدخل للنتورك مرة واحدة وخلاص .
- RNNs
  - dataset (Videos)
    - Each video => 100 Frames, Each Frame (image) $256 * 256$
    - Take Frame By Frame (Sequential)
    - At each time Step (t) Take 2 vecctor as an input
       - Current Vector (t) (Represent :Cur Frame (t) ), History Vector (Represent :Frame 0 to Frame t-1)
هنا بقا هياخد Frame By Frame مش كلهم مرا واحده. لا ده بيشتغل Sequential
كل مرة هياخد الفريم بتاعها + فريم بيعبر عن اللي فات كله History يعني فكر ف الهستوري كانه فكتور واحد شايل معلومات ملخصة عن كل اللي فات
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



## Architecture

ال RNN Layer هيشتغل على كل Frame لوحده ويعرف يستغل ال History او ال Frames السابقة.   
لو معاك 100 frame  يعني السامبل الواحد فيه 100 Frame ==>   
هيبقى معاك 100 Vec هيدخله واحد واحد و مع كل vec داخل للنتورك بتدخل معاه vec 1 بيعبر عن ملخص كل اللي فات History      
لو انت عند ال Frame K هتدخل للنتورك ===> vec K بيعبر عن الحالى و Vec from 1 to K-1 ده ال History وطبعا كل مرة هتعمل Update لل History ده تضيف عليه ال frame  الحالي   

```
in FC : 
كله مرة واحده input X هنا بتاخد ال 
a = g ( Wx_a . X + bx_a )
y = g ( Wa_y . a + ba_y )

in RNNs :
هنا هتاخده على مراحل
 في كل مرة هتاخد كلمه واحده فقط طبعا مع ال History
كل مرة هتاخد كلمه واحده فقط vector واحد بس يعني ويتجمع عليه ال History   

a(t) = g ( Wx_a . X(t) + a(t-1) + bx_a)
y(t) = g ( Wa_y . a(t) + ba_y)
----------------------------------------
a(0) = 0 ===> this is the History >>>>> History Vector Size = Number of Hidden State on RNN Layer
*****
a(0) = 0
a1 = g(Wx_a * X1 + Waa * a0 + bx_a) = g(Wx_a * X1 + bx_a), a(0) = 0 
y1 = g(Way * a1 + ba_y)
*****
a2 = g(Wx_a * X2 + Waa * a1 + bx_a)
y2 = g(Way * a2 + ba_y)
*****
a3 = g(Wx_a * X3 + Waa * a2 + bx_a)
   = g(Wx_a * X3 + Waa * (g(Wx_a * X2 + Waa * (g(Wx_a * X1 + Waa * a0 + bx_a)) + bx_a)) + bx_a)
Wx_a ==> Shared Weights Through Time
Waa  ==> Shared Weights Through Time

y3 = g(Way * a3 + b)
y3 = g(X3, X2, X1

```


## RNNS Types
- Ont to One 
- One to Many
  - Image Caption   
- Many to One
  - Sentiement Analysis 
- Many to Many
  - Translation 






