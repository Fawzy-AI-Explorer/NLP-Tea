# LSTM


```
GRU 
----
Gu = Sig ( Wxu.X(t) + Wcu.C(t-1) + bu )
Gr = Sig ( Wxr.X(t) + Wcr.C(t-1) + br )

C~(t) = g (Wax.X(t) + Gr[Waa.C(t-1)] + ba)
C(t) = Gu.C~(t) + (1-Gu).C(t-1)

Y(t) = g (Wcy.a(t) + by)
```

## LSTM 

1. Removing  Relevance Gate Gr
```
Gu = Sig ( Wxu.X(t) + Wcu.C(t-1) + bu )

C~(t) = tanh (Wax.X(t) + Waa.C(t-1) + ba)
C(t) = Gu.C~(t) + (1-Gu).C(t-1)

Y(t) = g (Wcy.C(t) + by)
```

2. Split “Update Gate” into two gates: “Update Gate”, “Forget Gate”
- Why Apply Constrain
    - C(t) = Gu.C~(t) + (1-Gu).C(t-1) 
    - if take 40% from C~(t) must take 60% from C(t-1)
    - what if you nedd to take 70% and 70% 
```
Gu = Sig ( Wxu.X(t) + Wcu.C(t-1) + bu )
Gf = Sig ( Wxf.X(t) + Wcf.C(t-1) + bf )
C~(t) = tanh (Wax.X(t) + Waa.C(t-1) + ba)
C(t) = Gu.C~(t) + Gf.C(t-1)   => Range Not Bounded 

Y(t) = g (Wcy.C(t) + by)
```
C~(t) => [-1, +1]  
C(t-1) => [-1, +1]  
if you do C~(t) + C(t-1) Range Not Bounded   
if you do 60% . C~(t) + 40% . C(t-1) Range Bounded from [-1, +1]


3. Bounded a

```
Gu = Sig ( Wxu.X(t) + Wcu.C(t-1) + bu )
Gf = Sig ( Wxf.X(t) + Wcf.C(t-1) + bf )

C~(t) = tanh (Wax.X(t) + Waa.C(t-1) + ba)
C(t) = Gu.C~(t) + Gf.C(t-1)   => Range Not Bounded 
a(t) = tanh (C(t)) => Bounded from [-1, +1]

Y(t) = g (Wcy.a(t) + by)
```

4. Output Gate (Go)
```
Gu = Sig ( Wxu.X(t) + Wcu.C(t-1) + bu )
Gf = Sig ( Wxf.X(t) + Wcf.C(t-1) + bf )
Go = Sig ( Wxo.X(t) + Wco.C(t-1) + bo )

C~(t) = tanh (Wax.X(t) + Waa.C(t-1) + ba)
C(t) = Gu.C~(t) + Gf.C(t-1)   => Range Not Bounded 
a(t) = Go( tanh (C(t)) ) => Bounded from [-1, +1]

Y(t) = g (Wcy.a(t) + by)
```
5. Input to Gates will be a(t-1) NOT C(t-1) As a is bounded
```
Gu = Sig ( Wxu.X(t) + Wau.a(t-1) + bu )
Gf = Sig ( Wxf.X(t) + Waf.a(t-1) + bf )
Go = Sig ( Wxo.X(t) + Wao.a(t-1) + bo )

C~(t) = tanh (Wax.X(t) + Waa.a(t-1) + ba)
C(t) = Gu.C~(t) + Gf.C(t-1)   => Range Not Bounded 
a(t) = Go( tanh (C(t)) ) => Bounded from [-1, +1]

Y(t) = g (Wcy.a(t) + by)
```

LSTM : 
- 3 Inputs 
    - 1. C(t-1)
    - 2. a(t-1)
    - 3. X(t)
- 3 Outputs
    - 1. C(t)
    - 2. a(t)
    - 3. y(t)

![image](https://github.com/user-attachments/assets/0412a582-44f5-4c49-97fa-beafb49fa610)

