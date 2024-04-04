# MitigatePromptBias

This is the PyTorch implementation of the paper: "<b><i>Take Care of Your Prompt Bias! Investigating and Mitigating the Prompt Bias in Factual Knowledge Extraction</i></b>" in The International Conference on Computational Linguistics (<b>COLING 2024</b>).

## Environment Configuration

1. Clone this code by: `git clone --recursive https://github.com/FelliYang/PromptBias.git`, it will recursively clone my forked version of the OpenPrompt library, where I implement the inference code for LLama2 models.

2. Clone datasets from the [link](https://gitee.com/FelliYang/factual-probing-dataset.git) and move them into the `./data` directory.

3. Create a new conda env named `PromptBias` using the `./env.yaml` file by: `conda env create -f env.yaml`

4. Install the OpenPrompt library locally by: `cd OpenPrompt; pip install -e .`

Now the environment is OK!

## Run

To reproduce the experiment results, please run with `bash ./code/scripts/debias.sh`. You can test with different models and prompts by changing configurations in the file.


## TODO
- add examples for different experiment settings
- add a concise explanation for the code construction and function.
- delete unnecessary source code files.

## Citation
If you find this work helpful, please consider citing as follows:  
```ruby
@inproceedings{xu2024promptbias,
  title={Take Care of Your Prompt Bias! Investigating and Mitigating the Prompt Bias in Factual Knowledge Extraction},
  author={Xu, Ziyang and Peng, Keqin and Ding, Liang and Tao, Dacheng and Lu, Xiliang},
  booktitle={Proc. of COLING},
  year={2024}
}
```
