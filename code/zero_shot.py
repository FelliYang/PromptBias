from datasets import load_from_disk
from tqdm import tqdm
from openprompt.data_utils.text_classification_dataset import AgnewsProcessor, DBpediaProcessor, ImdbProcessor, AmazonProcessor
from openprompt.data_utils.huggingface_dataset import YahooAnswersTopicsProcessor
import torch
from openprompt.data_utils.utils import InputExample
import argparse
import numpy as np
import wandb

from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer, KnowledgeableVerbalizer
from openprompt.prompts import ManualTemplate

from openprompt.pipeline_base import PromptForClassification


parser = argparse.ArgumentParser("")

parser.add_argument("--shot", type=int, default=0)
parser.add_argument("--seed", type=int, default=144)

parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument("--model", type=str, default='roberta')
parser.add_argument("--model_name_or_path", default='roberta-large')
parser.add_argument("--result_file", type=str, default="sfs_scripts/results_fewshot_manual_kpt.txt")
parser.add_argument("--openprompt_path", type=str, default="/workspace/data/users/xuziyang/code/PromptBias/TruelyKnow/OpenPrompt")

# parser.add_argument("--verbalizer", type=str)
parser.add_argument("--nocut", action="store_true")
parser.add_argument("--filter", default="none", type=str)
parser.add_argument("--template_id", type=int)
parser.add_argument("--max_token_split", default=-1, type=int)
parser.add_argument("--dataset",type=str)
parser.add_argument("--write_filter_record", action="store_true")

parser.add_argument("--sample_num",type=int, default=200)
# parser.add_argument("--calibration",type=str, default="none")
# parser.add_argument("--debias", type=str, default="none")
# parser.add_argument("--raw_evalute", type=bool, default=False)


args = parser.parse_args()

print(f"Exp Config: Dataset:{args.dataset} \t template_id: {args.template_id}")


from openprompt.utils.reproduciblity import set_seed
set_seed(args.seed)

from truely_know import Experiment
exp = Experiment()
exp.set_model(args.model, args.model_name_or_path)

dataset = {}


if args.dataset == "agnews":
    # raw_dataset = load_from_disk(f"{args.openprompt_path}/datasets/TextClassification/agnews")
    dataset['train'] = AgnewsProcessor().get_train_examples(f"{args.openprompt_path}/datasets/TextClassification/agnews/")
    dataset['test'] = AgnewsProcessor().get_test_examples(f"{args.openprompt_path}/datasets/TextClassification/agnews/")
    class_labels =AgnewsProcessor().get_labels()
    scriptsbase = "TextClassification/agnews"
    scriptformat = "txt"
    cutoff=0.5 if (not args.nocut) else 0.0
    max_seq_l = 128
    batch_s = 30
elif args.dataset == "dbpedia":
    # raw_dataset = load_from_disk(f"{args.openprompt_path}/datasets/TextClassification/dbpedia")
    dataset['train'] = DBpediaProcessor().get_train_examples(f"{args.openprompt_path}/datasets/TextClassification/dbpedia/")
    dataset['test'] = DBpediaProcessor().get_test_examples(f"{args.openprompt_path}/datasets/TextClassification/dbpedia/")
    class_labels =DBpediaProcessor().get_labels()
    scriptsbase = "TextClassification/dbpedia"
    scriptformat = "txt"
    cutoff=0.5 if (not args.nocut) else 0.0
    max_seq_l = 128
    batch_s = 30
elif args.dataset == "yahoo":
    raw_dataset = load_from_disk(f"{args.openprompt_path}/datasets/TextClassification/yahoo_answers_topics")
    dataset['train'] = list(map(YahooAnswersTopicsProcessor().transform, raw_dataset["train"]))
    dataset['test'] = list(map(YahooAnswersTopicsProcessor().transform, raw_dataset["test"]))
    class_labels =YahooAnswersTopicsProcessor().get_labels()
    scriptsbase = "TextClassification/yahoo_answers_topics"
    scriptformat = "json"
    cutoff=0.5 if (not args.nocut) else 0.0
    max_seq_l = 128
    batch_s = 30
elif args.dataset == "imdb":
    # raw_dataset = load_from_disk(f"{args.openprompt_path}/datasets/TextClassification/imdb")
    dataset['train'] = ImdbProcessor().get_train_examples(f"{args.openprompt_path}/datasets/TextClassification/imdb/")
    dataset['test'] = ImdbProcessor().get_test_examples(f"{args.openprompt_path}/datasets/TextClassification/imdb/")
    class_labels = ImdbProcessor().get_labels()
    scriptsbase = "TextClassification/imdb"
    scriptformat = "txt"
    cutoff=0
    max_seq_l = 512
    batch_s = 5
elif args.dataset == "amazon":
    # raw_dataset = load_from_disk(f"{args.openprompt_path}/datasets/TextClassification/amazon")
    dataset['train'] = AmazonProcessor().get_train_examples(f"{args.openprompt_path}/datasets/TextClassification/amazon/")
    dataset['test'] = AmazonProcessor().get_test_examples(f"{args.openprompt_path}/datasets/TextClassification/amazon/")
    class_labels = AmazonProcessor().get_labels()
    scriptsbase = "TextClassification/amazon"
    scriptformat = "txt"
    cutoff=0
    max_seq_l = 512
    batch_s = 5
else:
    raise NotImplementedError


mytemplate = ManualTemplate(tokenizer=exp.tokenizer).from_file(f"{args.openprompt_path}/scripts/{scriptsbase}/manual_template.txt", choice=args.template_id)
myverbalizer = ManualVerbalizer(exp.tokenizer, classes=class_labels).from_file(f"{args.openprompt_path}/scripts/{scriptsbase}/manual_verbalizer.{scriptformat}")


test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=exp.tokenizer,
            tokenizer_wrapper_class=exp.WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
            batch_size=batch_s,shuffle=False,teacher_forcing=False, predict_eos_token=False,
            truncate_method="tail")

# 计算mask的位置
# wrapped_example = mytemplate.wrap_one_example(dataset["test"][0])

# print(tokenized_example)

# 准备model
prompt_model = PromptForClassification(plm=exp.plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=True,plm_eval_mode=True)
prompt_model=  prompt_model.cuda()


def find_mask_id(content_free_template):
    t1 = content_free_template.replace("<input>", exp.tokenizer.unk_token)
    t2 = content_free_template.replace("<input>", exp.tokenizer.mask_token)

    promptTokenzier = exp.WrapperClass(max_seq_length=max_seq_l, decoder_max_length=3, tokenizer=exp.tokenizer,truncate_method="tail")
    wrapped_example_t1 = ManualTemplate(exp.tokenizer, text=t1).wrap_one_example(dataset["test"][0])
    tokenized_example_t1 = promptTokenzier.tokenize_one_example(wrapped_example_t1, teacher_forcing=False)
    wrapped_example_t2 = ManualTemplate(exp.tokenizer, text=t2).wrap_one_example(dataset["test"][0])
    tokenized_example_t2 = promptTokenzier.tokenize_one_example(wrapped_example_t2, teacher_forcing=False)
    assert len(tokenized_example_t1["input_ids"]) == len(tokenized_example_t2["input_ids"])
    # 找到unk中的mask位置
    mask_id = np.where(np.array(tokenized_example_t1["input_ids"])==exp.tokenizer.mask_token_id)
    
    return mask_id[0].item()



with open(f"{args.openprompt_path}/scripts/{scriptsbase}/manual_template.txt", "r") as f:
    raw_template = f.readlines()[args.template_id]
wandb.init(
    project="debais_vs_calibration",
    name=f"{args.dataset}_TemplateID_{args.template_id}",
    config={
        "dataset":args.dataset,
        "promptType": "manual",
        "templateID": args.template_id,
        "template": raw_template,
        "contentFreeType": "mask",
        "sampleNum": args.sample_num,    
    } 

)


# 不使用debias和calibration的原始inference
print("orinal inference...")
acc_raw, _ = exp.evaluate(prompt_model,test_dataloader)
print(acc_raw)
wandb.summary["raw_acc"] = acc_raw

# 传统debias, 使用三种输入得到的平均向量来估计prompt-only的特征
# prompt_only_template = ManualTemplate(tokenizer=exp.tokenizer).from_file(f"{args.openprompt_path}/scripts/{scriptsbase}/content_free_manual_template.txt", choice=args.template_id)
# 准备好模板
print("traditional debias...")
with open(f"{args.openprompt_path}/scripts/{scriptsbase}/content_free_manual_template.txt","r") as f:
    templates =  f.readlines()
content_free_template = templates[args.template_id]
mask_id = find_mask_id(content_free_template)
content_free_template = content_free_template.replace("<input>", exp.tokenizer.mask_token).replace('{"mask"}', exp.tokenizer.mask_token)
print("content_free_template: ***{}***".format(content_free_template))
assert '{' not in content_free_template

bias_logits, bias_vector = exp.get_manual_prompt_bias(exp.plm, exp.tokenizer,template=content_free_template, object_index=mask_id)
acc_debias_tra, _ = exp.evaluate(prompt_model,test_dataloader, bias_logits=bias_logits, bias_vector=bias_vector)
print(acc_debias_tra)
wandb.summary["debias_tra"] = acc_debias_tra


# sample-debias 通过采样200个样本的方式，来获得一个avg的向量用于估计prompt-only，重复三次随机实验
from openprompt.data_utils.data_sampler import FewShotSampler
acc_debias_sample = []
print("sample debias...")
for i in range(3):
    support_sampler = FewShotSampler(num_examples_total=args.sample_num, also_sample_dev=False)
    dataset['support'] = support_sampler(dataset['train'], seed=args.seed+i)
    for example in dataset['support']:
        example.label = -1 # remove the labels of support set for clarification
    support_dataloader = PromptDataLoader(dataset=dataset["support"], template=mytemplate, tokenizer=exp.tokenizer,
        tokenizer_wrapper_class=exp.WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
        batch_size=batch_s,shuffle=False, teacher_forcing=False, predict_eos_token=False,
        truncate_method="tail")
    bias_logits, bias_vector = exp.get_prompt_bias_by_sample(prompt_model, support_dataloader)
    acc, _ = exp.evaluate(prompt_model,test_dataloader, bias_logits=bias_logits, bias_vector=bias_vector)
    acc_debias_sample.append(acc)
print(acc_debias_sample)

for i in range(3):
    wandb.summary[f"acc_debias_sample_{i}"] = acc_debias_sample[i]


# traditional-calibration 
print("traditional calibration...")
with open(f"{args.openprompt_path}/scripts/{scriptsbase}/content_free_manual_template.txt","r") as f:
    templates =  f.readlines()
content_free_template = templates[args.template_id]
mask_id = find_mask_id(content_free_template)
content_free_template = content_free_template.replace("<input>", exp.tokenizer.mask_token).replace('{"mask"}', exp.tokenizer.mask_token)
print("content_free_template: ***{}***".format(content_free_template))
assert '{' not in content_free_template

bias_logits, bias_vector = exp.get_manual_prompt_bias(exp.plm, exp.tokenizer,template=content_free_template, object_index=mask_id)
cc_logits = torch.norm(bias_vector)*bias_logits
cc_logits = cc_logits.to(prompt_model.device)
acc_calibration_tra, _ = exp.evaluate(prompt_model,test_dataloader, calibration=True, calib_logits=cc_logits)
print(acc_calibration_tra)
wandb.summary["acc_calibration_tra"] = acc_calibration_tra

# 通过采样200个样本的方式，来获得一个avg的向量用于估计prompt-only 
from openprompt.data_utils.data_sampler import FewShotSampler
acc_calibration_sample = []
print("sample calibration...")
for i in range(3):
    support_sampler = FewShotSampler(num_examples_total=args.sample_num, also_sample_dev=False)
    dataset['support'] = support_sampler(dataset['train'], seed=args.seed+i) 
    for example in dataset['support']:
        example.label = -1 # remove the labels of support set for clarification
    support_dataloader = PromptDataLoader(dataset=dataset["support"], template=mytemplate, tokenizer=exp.tokenizer,
        tokenizer_wrapper_class=exp.WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
        batch_size=batch_s,shuffle=False, teacher_forcing=False, predict_eos_token=False,
        truncate_method="tail")
    cc_logits = exp.calibrate(prompt_model, support_dataloader).mean(dim=0)
    acc, _ = exp.evaluate(prompt_model,test_dataloader, calibration=True, calib_logits=cc_logits)
    acc_calibration_sample.append(acc)
print(acc_calibration_sample)
for i in range(3):
    wandb.summary[f"acc_calibration_sample_{i}"] = acc_calibration_sample[i]

wandb.finish()