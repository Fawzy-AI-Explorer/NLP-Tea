# 1D CNN
1D CNN (1-dimensional convolutional neural network) is a type of neural network that learns patterns in 1D data. It’s often used for:
- Time series data
- Text or word sequences
- Sensor data
- Audio signals


Imagine you have a row of numbers (Word embeddings)
- A 1D CNN uses a small filter (like a window) that slides over the row and detects patterns


Benefits of 1D CNN
- Fast and efficient
- Good at finding local patterns
- Needs fewer parameters than RNNs or LSTMs
- Can handle long sequences if combined with pooling


Shape of Input
A 1D CNN expects input like this:
- (samples, sequence_length, channels)
- (200, 100, 30) => 200 Sentences , each one 100 word, each word vec 30



(10, 3)  →  10 words per sentence, 3 features per word   
- ![image](https://github.com/user-attachments/assets/d9ef5745-7585-44b9-9bcb-81839013731a)

Conv1D layer: 1 filter, kernel size = 3 (3*3(As number of features = 3))   
- ![image](https://github.com/user-attachments/assets/2c1799dd-becc-496b-a01c-d61d985556a1)

- output shape = 10-3+1=8 => (8, 1)

Conv1D layer 2 filters, kernel size = 3 (3*3(As number of features = 3))
- output shape = 10-3+1=8 => (8, 2)
- ![image](https://github.com/user-attachments/assets/cfd84896-5c1c-4b7d-a835-7fcc34d4e959)



- Input shape (99, 30)
- Conv1D (10 Filter, Shape = 3, padding = "Valid", stride = 1)
  - (99 - 3)/1 + 1 = 97  
  - Shape = (97, 10)  
- Conv1D (20 Filter, Shape = 3, padding = "Valid", stride = 2)
  - (97 - 3)/2 + 1 = 48
  - Shape = (48, 20)
- MaxPooling1D layer: pool size = 2
  - output shape = (24, 20) 

![image](https://github.com/user-attachments/assets/b33793be-3f17-43aa-a655-30221d9e43cf)
































