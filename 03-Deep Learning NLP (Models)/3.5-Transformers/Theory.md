# Transformers 


## Encoder-Decoder Sequence-to-Sequence Model
The **Encoder-Decoder** architecture commonly used for tasks that involve transforming one sequence into another, such as **machine translation, text summarization.

### two main components:
1. **Encoder**: Processes the input sequence and converts it into a fixed-length context vector (hidden state). This vector captures the Whole of the input.
	compressed summary of the entire input sequence, capturing its meaning and structure. This vector is then passed to the **decoder**, which generates the output sequence.   
	a(t) = F(Wxa * X + Waa * a(t-1) + ba)   
    size of hidden =  number of nodes in RNN
2. Decoder: Takes the context vector and generates the output sequence, step by step.   

This architecture was originally built using **Recurrent Neural Networks (RNNs)**, specifically **Long Short-Term Memory (LSTM)** and **Gated Recurrent Unit (GRU)** networks.

---
**Cons:**
- **Bottleneck issue**:
    - Single, fixed-size context vector Capture the meaning of Entire Input Sequence.
    - Single, fixed-size context vector limits the ability to store long Sequencies.
- **Sequential Processing (NO parallelize)**: Since RNNs process sequences step-by-step, they cannot be easily parallelized.
- **Struggles with very long sequences**: LSTMs and GRUs still struggle with very long dependencies, even though they improve over simple RNNs.

![image](https://github.com/user-attachments/assets/b969347c-30b1-4968-a513-a906c9d80cb4)
