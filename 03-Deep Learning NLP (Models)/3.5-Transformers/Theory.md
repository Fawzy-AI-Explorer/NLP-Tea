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

```
Encoder :          
h0 = 0                    
h1 = f (Wxh.X1 + Whh.h0)        
h2 = f (Wxh.X2 + Whh.h1)             
h3 = f (Wxh.X3 + Whh.h2)              
Decoder :                                           
s0 = h3                   || y0 = <SOS>                
s1 = f (Wys.Y0 + Wss.S0)  || Y1 = softmax (Wsy.S1)            
s2 = f (Wys.Y1 + Wss.S1)  || Y2 = softmax (Wsy.S2)                 
s3 = f (Wys.Y2 + Wss.S2)  || Y3 = softmax (Wsy.S3)                       
```

![image](https://github.com/user-attachments/assets/ea62fdc0-7289-4aa6-bd03-1ba27ced51c4)



---
## Attention Mechanism
To solve the bottleneck issue, the **Attention mechanism** was introduced. Instead of relying on a single context vector, Attention assigns different weights to different parts of the input sequence, allowing the decoder to focus on relevant words at each step.
- enabling the decoder to look at all encoder outputs.
- Reduces the reliance on a single context vector.

**Benefits of Attention**
- **Improves performance on long sequences** by dynamically selecting relevant parts of the input.
-  **Eliminates the fixed-size bottleneck** by allowing the decoder to access all hidden states of the encoder.

![image](https://github.com/user-attachments/assets/78f2ca58-ddb5-4d22-9a82-4b95f37f6cb0)

![image](https://github.com/user-attachments/assets/8f592f51-285b-41fa-a1b4-9af2a7041526)

### Attention Block Calculations
                                                  
- Inputs : (S(i), h1,h2,h3,.....,hn)                          
- Output : S(i)~                     
                                 
1. Calc Score                                      
Score (a,b) = a.b or f(W.a + W.b)                      
        - Score (s0, h1) = α1                       
	- Score (s0, h2) = α2                                
	- Score (s0, h3) = α3                                 
2. Soft max over Scores                          
	- (α1 + α2 + α3 = 1)                           
3. context vector =>>                    
	- c(0) = α1.h1 + α2.h2 + α3.h3            
5. Combine Context and Decoder State             
	- s(0)~ = tanh (s0, c0)                   
- α(i) = Score (s0, hi)               
- softmax (α)                      
- C(0) = SUM (α(i).h(i))                      
- s(0)~ = tanh (s0, c0)             
```

