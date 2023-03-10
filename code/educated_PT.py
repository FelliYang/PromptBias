"""
本代码用于探究debiasing能否在prompt tuning的过程中指导模型获得更好的性能
"""

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
from openprompt.prompts import ManualTemplate, MixedTemplate

from openprompt.pipeline_base import PromptForClassification
from openprompt.utils.reproduciblity import set_seed

from transformers import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

from openprompt.data_utils import FewShotSampler
from transformers import BertForMaskedLM, RobertaForMaskedLM

def find_mask_id(exp, content_free_template):
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


parser = argparse.ArgumentParser("")

parser.add_argument("--seed", type=int, default=144)
parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument("--model", type=str, default='roberta')
parser.add_argument("--model_name_or_path", default='/workspace/data/users/vega/checkpoints/huggingface_model/roberta-large')
parser.add_argument("--openprompt_path", type=str, default="/workspace/data/users/xuziyang/code/PromptBias/TruelyKnow/OpenPrompt")
parser.add_argument("--dataset",type=str)
parser.add_argument("--sample_num",type=int, default=200)


parser.add_argument("--num_epochs",type=int, default=10)
parser.add_argument("--learning_rate",type=float, default=0.03)
parser.add_argument("--warmup_proportion",type=float, default=0)
parser.add_argument("--shot", type=int, default=32)
parser.add_argument("--debias_educated", type=float, default=False)
parser.add_argument("--educated_by_sample", type=bool, default=False)

args = parser.parse_args()

set_seed(args.seed)

from truely_know import Experiment
exp = Experiment()
exp.init_model(args.model, args.model_name_or_path)

dataset = {}


import os

if args.dataset == "yahoo":
    raw_dataset = load_from_disk(f"{args.openprompt_path}/datasets/TextClassification/yahoo_answers_topics")
    
    if os.path.exists("debug_dataset.pt"):
        dataset = torch.load("debug_dataset.pt")
    else:
        if os.path.exists("train_val.pt"):
            dataset = torch.load("train_val.pt")
        else:
            train_dataset = list(map(YahooAnswersTopicsProcessor().transform,  raw_dataset["train"]))
            # few shot 采样
            sampler = FewShotSampler(num_examples_per_label=args.shot,also_sample_dev=True, num_examples_per_label_dev=args.shot)
            dataset["train"], dataset["valid"] = sampler(train_dataset, seed=args.seed)
            

        dataset['test'] = list(map(YahooAnswersTopicsProcessor().transform, raw_dataset["test"]))

        torch.save(dataset , "debug_dataset.pt")

    # dataset['test'] = list(map(YahooAnswersTopicsProcessor().transform, raw_dataset["test"]))
    # torch.save(dataset , "debug_dataset.pt")
    
    class_labels =YahooAnswersTopicsProcessor().get_labels()
    scriptsbase = "TextClassification/yahoo_answers_topics"
    scriptformat = "json"
    # cutoff=0.5 if (not args.nocut) else 0.0
    max_seq_l = 128
    batch_s = 30

    if os.path.exists("support.pt"):
        # 支撑集cache
        dataset["support"] = torch.load("support.pt")
    else:
        train_dataset = list(map(YahooAnswersTopicsProcessor().transform,  raw_dataset["train"]))
        support_sampler = FewShotSampler(num_examples_total=args.sample_num, also_sample_dev=False)
        dataset['support'] = support_sampler(train_dataset, seed=args.seed)
        for example in dataset['support']:
            example.label = -1 # remove the labels of support set for clarification
        torch.save(dataset["support"], "support.pt")
        
else:
    raise NotImplementedError


dataset["test"] = dataset["test"]

# 定义一个optiprompt的连续模板，暂时只用于yahoo
templateTxt = '{"placeholder": "text_a"} {"placeholder": "text_b"} {"soft":"This topic is about"} {"mask"} .'
templateOnly = '<input> {"soft":"This topic is about"} {"mask"} .'
# templateTxt = ' {"soft":None, "duplicate":10} {"placeholder": "text_a"} {"placeholder": "text_b"}  {"mask"}'
mytemplate = MixedTemplate(
    model=exp.plm, tokenizer=exp.tokenizer, text = templateTxt
)
# mytemplate.soft_embedding.weight.data.normal_(mean=0.0, std=exp.model_config.initializer_range)
myverbalizer = ManualVerbalizer(exp.tokenizer, classes=class_labels).from_file(f"{args.openprompt_path}/scripts/{scriptsbase}/manual_verbalizer.{scriptformat}")

train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=exp.tokenizer,
            tokenizer_wrapper_class=exp.WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
            batch_size=batch_s,shuffle=False,teacher_forcing=False, predict_eos_token=False,
            truncate_method="tail")

valid_dataloader = PromptDataLoader(dataset=dataset["valid"], template=mytemplate, tokenizer=exp.tokenizer,
            tokenizer_wrapper_class=exp.WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
            batch_size=batch_s,shuffle=False,teacher_forcing=False, predict_eos_token=False,
            truncate_method="tail")

test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=exp.tokenizer,
            tokenizer_wrapper_class=exp.WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
            batch_size=batch_s,shuffle=False,teacher_forcing=False, predict_eos_token=False,
            truncate_method="tail")


support_dataloader = PromptDataLoader(dataset=dataset["support"], template=mytemplate, tokenizer=exp.tokenizer,
    tokenizer_wrapper_class=exp.WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
    batch_size=batch_s,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")


prompt_model = PromptForClassification(plm=exp.plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=True)
prompt_model=  prompt_model.cuda()

# zero-shot
# acc, _ = exp.evaluate(prompt_model, test_dataloader)
# print(f"zero shot 测试集acc { acc}")

# 开始训练
print("数据集划分 train {} valid {} test {}".format(
    len(dataset["train"]),len(dataset["valid"]),len(dataset["test"])))


def get_debias_vector():
    if args.educated_by_sample:
        bias_logits, bias_vector = exp.get_template_bias_tensor_by_sample(prompt_model, support_dataloader, evaluteMode=False)
    else:
        # 使用mask直接debias
        mask_id = find_mask_id(exp, templateOnly)
        content_free_template = templateOnly.replace("<input>", exp.tokenizer.mask_token).replace('{"mask"}', exp.tokenizer.mask_token)
        print("content_free_template: ***{}***".format(content_free_template))
        assert '{' not in content_free_template
        bias_logits, bias_vector = exp.get_template_bias_tensor(exp.plm, exp.tokenizer,template=content_free_template, y_mask_index=mask_id)
        
    return bias_logits, bias_vector

import os
embedding_save_dir = "/workspace/data/users/xuziyang/code/PromptBias/TruelyKnow/outputs/promptbias"
best_ckpt = os.path.join(embedding_save_dir, f"weight_save/model_full-prompt_random_init.ckpt")

loss_func = torch.nn.CrossEntropyLoss()
no_decay = ['bias', 'LayerNorm.weight']

optimizer_grouped_parameters = [
    {'params': [p for n,p in prompt_model.template.named_parameters() if "raw_embedding" not in n]}
]


optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
t_total = len(train_dataloader) * args.num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, int(t_total * args.warmup_proportion), t_total)

pbar = tqdm(total=t_total)

from time import time
time_start = time()
best_acc = 0
best_epoch = -1  
exp.create_dir(best_ckpt)
global_step = 0
for epoch in range(args.num_epochs):
    tot_loss = 0
    prompt_model.train()
    # tot_loss = 0
    for step, inputs in enumerate(train_dataloader):
        global_step += 1
        pbar.update(1)
        inputs = inputs.cuda()
       
        # mask_logits = logits[torch.where(inputs['loss_ids']>0)]
        # 采用debias辅助优化
        if args.debias_educated:
            bias_logits, bias_vector =  get_debias_vector()

            model = prompt_model.plm
            if isinstance(model, BertForMaskedLM):
                transformer_blocks = model.bert
                tranform_layer  = model.cls.predictions.transform
                decoder = model.cls.predictions.decoder
            elif isinstance(model, RobertaForMaskedLM):
                transformer_blocks = model.roberta
                tranform_layer = torch.nn.Sequential(model.lm_head.dense, torch.nn.GELU(), model.lm_head.layer_norm)
                decoder = model.lm_head.decoder

            if isinstance(prompt_model, PromptForClassification):
                # prompt_model(inputs)
                # 以下逻辑是把openprompt的forward抄了过来
                batch = prompt_model.template.process_batch(inputs)
                input_batch = {key: batch[key] for key in batch if key in prompt_model.prompt_model.forward_keys}
                transformer_output = transformer_blocks(**input_batch)[0]
                mask_token_transformer_output = transformer_output[torch.where(inputs['loss_ids']>0)]
                mask_token_features = tranform_layer(mask_token_transformer_output)

                # linear层之后的特征向量, 论文理论框架中用于debias的向量
                prediction_vectors = mask_token_features

                mask_token_logits = None
                for i in range(mask_token_features.shape[0]):
                    norm2 = torch.norm(mask_token_features[i], p=2)
                    temp = decoder(mask_token_features[i]).unsqueeze(dim=0) / norm2
                    if mask_token_logits == None:
                        mask_token_logits = temp
                    else:
                        mask_token_logits = torch.cat((mask_token_logits,temp),dim=0)
                
                mask_logits = mask_token_logits

                # 从forward逻辑中中抄过来的
                # mask_logits =  prompt_model.verbalizer.process_outputs(mask_logits, batch=inputs)
                       
            # 使用verbalizer过滤
            mask_logits = prompt_model.verbalizer.process_outputs(mask_logits, batch=inputs)
            bias_logits = prompt_model.verbalizer.process_outputs(torch.unsqueeze(bias_logits,dim=0), batch=inputs)
            # 计算出两个向量的norm以及缩放因子
            with torch.no_grad():
                prompt_only_norm =  torch.norm(bias_vector).item()
                prompt_input_norms = torch.norm(prediction_vectors,dim=1)
                normalized_input_vector = (prediction_vectors.T / prompt_input_norms).T
                normalized_prompt_vector = bias_vector / prompt_only_norm 
                educated_vector = normalized_input_vector  - normalized_prompt_vector
                scale_factor = prompt_input_norms/torch.norm(educated_vector,dim=1)
            educated_logits = (scale_factor * ( (mask_logits.T / prompt_input_norms).T - bias_logits).T).T
            
            loss = loss_func(educated_logits, inputs["label"])
            loss.backward()

        else:
            mask_logits = prompt_model(inputs)
            loss = loss_func(mask_logits, inputs["label"])
            loss.backward()
        # torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        # if global_step%10==0:
        # print("global step {}: loss {:.6f}".format(global_step, loss.item()))
        
        tot_loss += loss.item()

        print(tot_loss/(step+1))
    # validate it
    if args.debias_educated:
        with torch.no_grad():
            bias_logits, bias_vector =  get_debias_vector() 
        acc,_ =  exp.evaluate(prompt_model, valid_dataloader, bias_logits=bias_logits, bias_vector=bias_vector)
    else:
        acc,_ =  exp.evaluate(prompt_model, valid_dataloader)
    if acc > best_acc:
        # 保存最好的epoch
        best_epoch = epoch
        best_acc = acc
        soft_embeds = prompt_model.template.state_dict()["soft_embedding.weight"].clone()
        torch.save(soft_embeds,best_ckpt)
        
pbar.close()
print("best epoch {} valid precision: {}".format(best_epoch,best_acc,))
if args.debias_educated:
    with torch.no_grad():
        bias_logits, bias_vector =  get_debias_vector() 
    acc,_ =  exp.evaluate(prompt_model, test_dataloader, bias_logits=bias_logits, bias_vector=bias_vector)
else:
    acc,_ =  exp.evaluate(prompt_model, test_dataloader)
print(f"测试集acc { acc}")


