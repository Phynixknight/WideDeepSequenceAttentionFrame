# Wide And Deep Frame

Include:
1. Wide And Deep Model
2. Sequence Model
3. Attention

## Wide And Deep Model:  
**example for two deep branch**   
![](model_wide_and_deep.png)  

## Attention
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