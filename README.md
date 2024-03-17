# MitigatePromptBias

This is the PyTorch implementation of the paper: "<b><i>Take Care of Your Prompt Bias: Investigating and Mitigating the Prompt Bias in Factual Knowledge Extraction</i></b>" in The International Conference on Computational Linguistics (<b>COLING 2024</b>).

## Environment Configuration

1. clone this code and create a new conda env named `PromptBias` using the `env.yaml` file by: `conda env create -f env.yaml`

2. download the dataset from the [link](https://gitee.com/FelliYang/factual-probing-dataset.git) and unzip it into the ./data directory.

3. Install the OpenPrompt library locally using my forked version in [OpenPrompt_Llama2](https://github.com/FelliYang/OpenPrompt), where I implement the inference code for LLama2 models. `cd OpenPrompt; pip install -e .`

Now the environment is OK!

## Run

To reproduce the experiment results, please run with `bash ./code/scripts/debias.sh`. You can test with different models and prompts by changing configurations in this file.


## TODO
- add more concrete examples
- add a concise explanation for the code construction and function.
- delete unnecessary source code files.

## Citation
If you find this work helpful, please consider citing as follows:  
```ruby
@inproceedings{xu2024promptbias,
  title={Take Care of Your Prompt Bias: Investigating and Mitigating the Prompt Bias in Factual Knowledge Extraction},
  author={Xu, Ziyang and Peng, Keqin and Ding, Liang and Tao, Dacheng and Lu, Xiliang},
  booktitle={Proc. of COLING},
  year={2024}
}
```
