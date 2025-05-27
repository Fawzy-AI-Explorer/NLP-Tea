# GRU

GRUs are an improved version of Recurrent Neural Networks (RNNs) designed to better capture long-term dependencies in sequential data.
**RNNs**
-  RNNs maintain a hidden state a(t)​ that is updated at each time step t based on the input x(t)​ and the previous hidden state h(t-1)​.
-  a(t​) = g(W. x(t)​ + W​ ⋅ h(t−1) ​+ b )
-  Challenges with Long-Term Dependencies:
	-  Poor memory of long-term dependencies in sequences.
    -  Vanishing & Exploding Gradient Problem
    -  always update a if u work on videos 50 frames (ads appear from t=5 to t=8) network doesn't want to take these frames in history 
مش عايزة تاخدها معاها ملهاش لازمة يعني time steps وخلاص النتورك ممكن تبقا فيه Update انا مش عايز كل مرة اعمل 

GRUs introduce gates to control the flow of information, solving the vanishing gradient problem and improving long-term dependency handling.



```
RNNs

a(t) = g (Wax.X(t) + Waa.a(t-1) + ba)
---------------------------
GRU
a~(t) = g (Wax.X(t) + Waa.a(t-1) + ba)
a(t) = Gu.a~(t) + (1-Gu).a(t-1)

if Gu=1 
a~(t) = g (Wax.X(t) + Waa.a(t-1) + ba)
a(t) = Gu.a~(t) = g (Wax.X(t) + Waa.a(t-1) + ba) ==> RNN || Update History with current input

if Gu=0 
a~(t) = g (Wax.X(t) + Waa.a(t-1) + ba)
a(t) = a(t-1) ==> do not Update History || Drop the current input

if Gu = 0.6 
a~(t) = g (Wax.X(t) + Waa.a(t-1) + ba)
a(t) = 0.6.a~(t) + 0.4.a(t-1) ==> take 60% from a~(t) and 40% from a(t-1)


The update gate U decides how much of the previous hidden state (a(t−1) needs to be retained and how much of the new candidate hidden state a~(t) should replace it. 

```
Gu => u will take the current time step in history or not ?
if u need to forget All history and start from current time step

```
GRU
Gu = Sig ( Wxu.X(t) + Wau.a(t-1) + bu )
Gr = Sig ( Wxr.X(t) + War.a(t-1) + br )

a~(t) = g (Wax.X(t) + Gr[Waa.a(t-1)] + ba)
a(t) = Gu.a~(t) + (1-Gu).a(t-1)
---------------
if Gr = 0, Gu = 1  ==> Traditional NN 
a~(t) = g (Wax.X(t) + ba)
a(t) = a~(t)  

if Gr = 1, Gu = 1  ==> RNN 
a~(t) = g (Wax.X(t) + Waa.a(t-1) + ba)
a(t) = Gu.a~(t) 

```

GRUs use two gates:
 - Update Gate (Gu​): Decides how much of the new information to use.
	- Balances **new information** a~(t) and **past information** a(t−1).
	- Gu = Sig ( Wxu.X(t) + Wau.a(t-1) + bu )
- Relevance Gate (Rt): Decides how much of the past information to forget.
	- Controls **how much past information to forget** while computing the new candidate activation.
	- Gr = Sig ( Wxr.X(t) + War.a(t-1) + br )

![image](https://github.com/user-attachments/assets/95737e76-f42a-4389-996e-2d662509f5f3)





