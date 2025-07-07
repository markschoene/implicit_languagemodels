<div align="center">

<h1> Implicit Language Models are RNNs</h1>  
<h3>Balancing Parallelization and Expressivity</h3>

Mark Schöne<sup>1,2 *</sup>, Babak Rahmani<sup>2 *</sup>, Heiner Kremer<sup>2</sup>, Fabian Falck<sup>2</sup>, Hitesh Ballani<sup>2</sup>, Jannes Gladrow<sup>2 $\dagger$</sup>

<sup>1</sup>  TU Dresden, Germany, <sup>2</sup> Microsoft Research, Cambridge, UK

(\*) Equal contribution. ($\dagger$) Corresponding author.

[![arXiv](https://img.shields.io/badge/ArXiv-2502.07827-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2502.07827)
[![Project](https://img.shields.io/badge/Homepage-AOC-orange.svg?logo=googlehome)](https://www.microsoft.com/en-us/research/project/aoc)

</div>

## ICML 2025 Spotlight

🎉🚀 We are delighted to announce that our paper was spotlighted at ICML 2025! 🚀🎉

# Abstract
State-space models (SSMs) and transformers dominate the language modeling landscape. However, they are constrained to a lower computational complexity than classical recurrent neural networks (RNNs), limiting their expressivity. In contrast, RNNs lack parallelization during training, raising fundamental questions about the trade off between parallelization and expressivity. We propose implicit SSMs, which iterate a transformation until convergence to a fixed point. Theoretically, we show that implicit SSMs implement the non-linear state-transitions of RNNs. Empirically, we find that only approximate fixed-point convergence suffices, enabling the design of a scalable training curriculum that largely retains parallelization, with full convergence required only for a small subset of tokens. Our approach demonstrates superior state-tracking capabilities on regular languages, surpassing transformers and SSMs. We further scale implicit SSMs to natural language reasoning tasks and pretraining of large-scale language models up to 1.3B parameters on 207B tokens - representing, to our knowledge, the largest implicit model trained to date. Notably, our implicit models outperform their explicit counterparts on standard benchmarks.

# Installation
Requirements:
- `mamba_ssm` and `causal_conv1d`

Install this package 
```
pip install .
```

# Usage
The code allows for integration with the HuggingFace Platform.
We provide local configuration files that can be loaded with `AutoConfig`
```python
from transformers import AutoConfig, AutoModel
import implicit_llm

cfg = AutoConfig.from_pretrained('hf_models/llama3-1.3b-implicit')
model = AutoModel.from_config(cfg)
```

# Examples

## State-Tracking
We provide a simple training script based on the huggingface Trainer. 
First, generate the dataset [following the instructions](state_tracking/README.md).
Then, train your models with
```python
python -m examples.state_tracking \ 
    --model_name hf_models/mamba2-state-tracking-implicit \
    --train_dataset /path/to/data/train_A5_L256_P090.bin \
    --eval_dataset /path/to/data/test_A5_L256_P050.bin \
    --test_dataset /path/to/data/test_A5_L256_P050.bin 
```
The script works for arbitrary models from the huggingface hub.
Feel free to train your favorite models!

To evaluate a trained model use the `--eval` flag and point `--model_name` to the trained model checkpoint. 
E.g. run evaluation on the test set with 1024 tokens
```python
python -m examples.state_tracking \ 
    --model_name path/to/trained/model/checkpoint \
    --train_dataset /path/to/data/train_A5_L256_P090.bin \
    --eval_dataset /path/to/data/test_A5_L256_P050.bin \
    --test_dataset /path/to/data/test_A5_L256_P050.bin
    --eval 
```

## Downstream Evaluation of Pretrained Models

## Duality of Simultaneous Fixed Point Iteration and Sequential Fixed Point Iteration
By default, training always used the simultaneous fixed point iteration, while generation always uses the sequential fixed point iteration.
We provide examples of evaluating a model in the sequential mode, e.g. to reproduce Figure 2C, in `tests/test_evaluation.py` and in `examples/state_tracking.py`.
The state tracking example code uses the simultaneous mode for validation during training.
A sequential pass is done at the end of training on the test set. 

# Common Issues
    ValueError: The checkpoint you are trying to load has model type `implicit_mamba2` but Transformers does not recognize this architecture.

--> Just `import implicit_llm` to register the implicit models with the HF library, or `
```
from implicit_llm import register_implicit_causal_lm
register_implicit_causal_lm()
```
# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

# Citation
```
@inproceedings{
schone2025implicit,
title={Implicit Language Models are {RNN}s: Balancing Parallelization and Expressivity},
author={Mark Sch{\"o}ne and Babak Rahmani and Heiner Kremer and Fabian Falck and Hitesh Ballani and Jannes Gladrow},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=5EbiopWH6e}
}
```