### ReadMe: PipeMaxVit
In this project, I adapted the MaxViT([Multi-Axis Vision Transformer](https://arxiv.org/abs/2204.01697)) model for parallel and distributed machine learning model training to enhance performance. I aimed to minimize throughput while maintaining statistical efficiency. Here are the key terms:
* Throughput - data samples processed per unit time
* Statistical Efficiency - empirical change in objective function per processed data sample
* Goodput - product of throughput and statistical efficiency, representing useful throughput

The three approaches were explored: Distributed Data Parallelism, Pipeline Parallelism, Distributed Data Parallelism + Pipeline Parallelism

1. Distributed Data Parallelism - This method divides data into smaller chunks and processes them concurrently across multiple computing nodes in a distributed system for faster, more efficient large dataset processing. In my case, this approach increased throughput by +85% and maintaining similar statistical efficiency.
2. Pipeline Parallelism - This technique involves dividing the MaxViT blocks among GPUs. For example, for 2 GPUs, each GPU processes 2 MaxViT blocks, and for 4 GPUs, each GPU processes 1 MaxViT block. This method performed better than single GPU training only for higher batch sizes. For 4 GPUs and a batch size of 128, Pipeline Parallelism improved goodput. Better results can be achieved by rewriting MaxViT blocks sequentially. 
3. Distributed Data Parallelism + Pipeline Parallelism - this approach combines the two previous techniques by splitting data into groups and further dividing the training task among GPUs within those groups. This method had the best overall goodput, improving it by ~+110%.

To run the project, simply run the script as follows:
```python train.py -e 5 -b 64 -m DP_PP -g 4```, for more options refer to train.py file

References: https://github.com/ChristophReich1996/MaxViT

