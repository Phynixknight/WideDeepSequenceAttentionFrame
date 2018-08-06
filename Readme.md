# Wide And Deep Frame
[![license](https://img.shields.io/badge/License-Apache_2.0-brightgreen.svg)](https://github.com/Phynixknight/WideDeepSequenceAttentionFrame/LICENSE) [![dep1](https://img.shields.io/badge/Tensorflow-1.2+-blue.svg)](https://www.tensorflow.org/) [![dep2](https://img.shields.io/badge/Keras-2.0+-blue.svg)](https://keras.io/) 

Content:  
可以方便定制化的 Wide And Deep 框架， 各种特征编码方式正在完善。

Include:  
- [x] **Wide And Deep Model**
- [x] **Sequence Model**
- [x] **Attention Mechanism**

## Wide And Deep Model:  
**example for two deep branch**   
![](pic/model_wide_and_deep.png)  

## Attention Mechanism  

Attention can be happened befor or after rnn，there we use attention after lstm  

**simplest attention in Dense**  
![](pic/model_attention.png)  

**simplest multi attention in lstm**     
Attention defined per time series (each TS has its own attention)  
Theory Graph  
![Theory Graph](pic/graph_multi_attention.png)  
Code Graph  
![Code Graph](pic/model_attention.png)   
Application Graph  
![Application Graph](pic/model_wide_deep_sequence_nonmask_attention.png)  

**Attention shared across all the time series**  
Theory Graph  
![](pic/graph_single_attention.png)  

**disadvantage**   
Those method cannot support masking in embedding.     
So I use TimeDistribution instead, there's some diffrents accordingly.   
TimeDistributino ≈ Permute([2,1]) + Dense(1,softmax) + Permute([2,1])   
So it need Repeate to Reshape to the shape of lstm  

## Wide And Deep & Sequence & Attention
**example for two deep branch and three sequence branch**  
![](pic/model_wide_deep_sequence_attention.png)