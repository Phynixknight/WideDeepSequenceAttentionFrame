# Wide And Deep Frame
[![license](https://img.shields.io/badge/License-Apache_2.0-brightgreen.svg)](https://github.com/Phynixknight/WideDeepSequenceAttentionFrame/LICENSE) [![dep1](https://img.shields.io/badge/Tensorflow-1.2+-blue.svg)](https://www.tensorflow.org/) [![dep2](https://img.shields.io/badge/Keras-2.0+-blue.svg)](https://keras.io/) 

Include:  
- [x] **Wide And Deep Model**
- [x] **Sequence Model**
- [x] **Attention Mechanism**

## Wide And Deep Model:  
**example for two deep branch**   
![](model_wide_and_deep.png)  

## Attention Mechanism
**simplest multi attention**   
![](graph_multi_attention.png)   
**simplest multi attention in lstm**  
i.e. single attention vector is flase   
Attention defined per time series (each TS has its own attention)  
![](graph_multi_attention_lstm.png)  
**Attention shared across all the time series**  
![](graph_single_attention.png)  

## Wide And Deep & Sequence & Attention
**example for two deep branch and three sequence branch**  
![](model_wide_deep_sequence_attention.png)