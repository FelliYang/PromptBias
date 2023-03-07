"""
使用openprompt 来重构整个代码
"""
import collections
import json
import math
import os
from re import template
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn.parameter
from datasets import load_dataset, load_from_disk
from openprompt import (PromptDataLoader, PromptForClassification,
                        PromptForGeneration, PromptModel)
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompt_base import Template
from openprompt.prompts import ManualTemplate, MixedTemplate, SoftTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt.data_utils.huggingface_dataset import YahooAnswersTopicsProcessor
from openprompt.data_utils.data_sampler import FewShotSampler
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
from transformers import (AdamW, AutoModelForMaskedLM, AutoTokenizer,
                          BertConfig, BertModel,AutoConfig,
                          get_linear_schedule_with_warmup,
                          get_cosine_schedule_with_warmup,
                          BertForMaskedLM, RobertaForMaskedLM)
from transformers.tokenization_utils import PreTrainedTokenizer

from utils import load_json, load_jsonl, load_vocab
from torch import nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from collections import Counter,defaultdict
from math import log
from utils import set_seed
from prettytable import PrettyTable
import copy


set_seed(7)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
class Experiment():
    def __init__(self) -> None:
        # settting
        self.CONTINUE_PROMPT_NUM = 5
        self.num_epochs = 10
        self.learning_rate = 3e-3
        self.warmup_proportion = 0.1
        num_tokens = 5

        # output save
        self.output_result = {}
        
        self.work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # dataset
        self.lama_data_dir = self.work_dir + "/data/LAMA-TREx"
        self.auto_prompt_dir = self.work_dir + "/data/autoprompt_data"
        self.lama_whu_data_dir = self.work_dir + "/data/LAMA-TREx_UHN"
        self.wiki_uni_data_dir = self.work_dir + "/data/wiki_uni"
        self.common_vocab_path = self.work_dir + "/common_vocabs/common_vocab_cased.txt"
        self.lama_template_path = self.work_dir + "/relation_metainfo/LAMA_relations.jsonl"
        self.LPAQA_template_path = self.work_dir + "/relation_metainfo/LPAQA_relations.jsonl"
        self.AutoPrompt_template_path = self.work_dir + "/relation_metainfo/AutoPrompt_relations.jsonl"
        self.AutoPrompt_roberta_template_path = self.work_dir + "/relation_metainfo/AutoPrompt_relations_roberta.jsonl"
        # 导入LAMA manual prompt
        data = load_jsonl(self.lama_template_path)
        self.lama_template = dict([(item["relation"], item["template"]) for item in data])
        data = load_jsonl(self.LPAQA_template_path)
        self.LPAQA_template = dict([(item["relation"], item["template"]) for item in data])
        data = load_jsonl(self.AutoPrompt_template_path)
        self.AutoPrompt_template = dict([(item["relation"], item["template"]) for item in data])
        data = load_jsonl(self.AutoPrompt_roberta_template_path)
        self.AutoPrompt_tempalte_roberta = dict([(item["relation"], item["template"]) for item in data])
        self.AutoPrompt_tempalte_roberta = {k:v.strip() for k,v in self.AutoPrompt_tempalte_roberta.items()}

        # 导入LAMA所有关系
        data =  load_jsonl(self.work_dir + "/relation_metainfo/LAMA_relations.jsonl")
        self.relations = [ d["relation"] for d in data]
        # 过滤掉TRE之外的5个relation
        for relation in self.relations[:]:
            if not os.path.exists(os.path.join(self.lama_data_dir,f"{relation}.jsonl")):
                self.relations.remove(relation)

        # 获取预训练模型
        # self.init_model("bert","bert-base-cased")

        # 导入LAMA的公共词表
        self.init_common_vocab(self.common_vocab_path)
        # self.common_vocab_indices =  self.get_common_vocab_indices(self.tokenizer)

    def init_model(self,model_type, model_name):
        self.model_type = model_type
        self.model_name = model_name
        plm, tokenizer, model_config, WrapperClass = load_plm(model_type, model_name)
        self.plm = plm
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.WrapperClass = WrapperClass

    
    def init_common_vocab(self, common_vocab_path):
        self.lama_vocab_subset = load_vocab(common_vocab_path)


    def clear_output(self,):
        self.output_result = {}
    

    def save_output(self, path):
        self.create_dir(path)
        str = json.dumps(self.output_result,indent=4)
        with open(path, "w") as f:
            f.write(str)


    def load_output(self, path):
        with open(path, "r") as f:
            d = json.load(f)
            self.output_result = d
    

    def print_output(self,):
        for model in self.output_result.keys():
            table = PrettyTable()
            table.title = model
            table.field_names = ["Datasets",  "Prompts","P","P^d","KL","KL^d"]
            model_output = self.output_result[model]
            for dataset in model_output.keys():
                table.add_row([dataset,"","","","",""])
                dataset_output = model_output[dataset]
                for prompt in dataset_output.keys():
                    prompt_output = dataset_output[prompt]
                    P,P_d,KL,KL_d = prompt_output['P'], prompt_output["P_d"], prompt_output["KL"], prompt_output["KL_d"]
                    P = math.floor(P *100 * 10+0.5)/10
                    P_d = math.floor(P_d *100 * 10+0.5)/10
                    KL = math.floor(KL * 10+0.5)/10
                    KL_d = math.floor(KL_d * 10+0.5)/10
                    table.add_row(["",prompt,P,P_d,KL,KL_d])
            print(table)
                    

    def add_output_item(self,model, dataset, prompt, result):
        if model not in self.output_result.keys():
            self.output_result[model] = {}
        if dataset not in self.output_result[model].keys():
            self.output_result[model][dataset] = {}
        self.output_result[model][dataset][prompt] = {"P":result[0], "P_d":result[1], "KL":result[2], "KL_d":result[3]}


    def create_dir(self, path):
        """为任意路径创建其父目录"""
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    

    def load_data(self, data_path, vocab_subset=None, filter_repeat=True, return_resample_weights=False, ):
        # 从文件中load出list格式的数据集
        ext = os.path.splitext(data_path)[-1]
        assert ext  == ".jsonl" or ext == ".json"
        try:
            if ext == ".jsonl":
                data = []
                with open(data_path, "r") as f:
                    for line in f.readlines():
                        data.append(json.loads(line))
            elif ext == ".json":
                with open(data_path,"r") as f:
                    data =  json.load(f)
        except:
            print(f"打开文件{data_path}失败")
            exit(-1)

        # 如果data中不包括add_output_item
        # 如果vocab_subset存在，那么需要根据vocab_subset来过滤不符合要求的obj_label
        if vocab_subset:
            current_len = len(data)
            data = list(filter(lambda x: x["obj_label"] in vocab_subset, data))
            print("filter out {} items since the vocab_subset".format(current_len - len(data)))
        

        # 是否过滤重复的数据，默认为true
        if filter_repeat:
            current_len = len(data)
            sub_obj_set = set()
            for item in data:
                t = (item["sub_label"],item["obj_label"])
                sub_obj_set.add(t)
            sub_obj = list(sub_obj_set)
            sub_obj_dict = dict(zip(sub_obj,[1]*len(sub_obj)))

            for item in data[:]:
                k = (item["sub_label"],item["obj_label"])
                if sub_obj_dict[k] == 1:
                    sub_obj_dict[k] = 0
                else:
                    data.remove(item)

            print("filter out {} items since no repeat permit".format(current_len-len(data)))
            current_len = len(data)

        # 是否返回重采样所需要的weights
        weights = None
        if return_resample_weights:
            # count the num of each obj_label in data
            obj_count = {}
            # record type of sample in dataset
            type = []
            for e in data:
                cls = e["obj_label"]
                type.append(cls)
                if not obj_count.__contains__(cls):
                    obj_count[cls] = 1
                else:
                    obj_count[cls] += 1
            
            weights = [1 / obj_count[type[i]] for i in range(len(data))] 

        return data, weights 
    

    def get_vocab_indices(self, vocab_subset, tokenizer:PreTrainedTokenizer):
        """ 给定一个子词表，生成一个vocab_subset到tokenizer词表中的映射列表"""
        index_list = []
        for word in vocab_subset:
            # 这里加一个空格，是因为roberta分词的时候要求单独的单词前面有空格
            tokens = tokenizer.tokenize(' ' + str(word))
            if (len(tokens) == 1) and (tokens[0] != tokenizer.unk_token_id):
                index_list.append(tokenizer.convert_tokens_to_ids(tokens)[0])
            else:
                msg = "word {} from vocab_subset not in model vocabulary!".format(word)
                print(msg)
        index_list.sort()
        return index_list


    def get_common_vocab_indices(self, tokenizer:PreTrainedTokenizer):
        """需要对lama使用一个common vocab过滤，这样才能对其lama相关工作，但是我觉得这其实是有点虚提高了精度的"""
        index_list = self.get_vocab_indices(self.lama_vocab_subset,tokenizer)
        indices = torch.as_tensor(index_list)
        return indices


    def get_answer_entity_indices(self, relation,tokenizer:PreTrainedTokenizer, include_uni_labels=True, common_vocab_intersection=True):
        """这个函数用来得到lama对应的answer空间，例如P的answer空间是cities"""
        entity_path = self.work_dir + "/data/entity.json"
        entity_data = load_json(entity_path)[relation]
        entity_data = entity_data["objects"]
        # 需要对entity过滤，只留下单个词的entity

        #  # 获取预训练模型
        # plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")

        common_vocab = self.lama_vocab_subset
        def filter_common_vocab_subset(word):
            if word in common_vocab:
                return True
            else:
                return False

        # 加上relation的labels,做一个Union集合
        lama_path = os.path.join(self.lama_data_dir,f"{relation}.jsonl")
        lama_data = load_jsonl(lama_path)
        lama_labels = [ item["obj_label"] for item in lama_data]
        entity_data += lama_labels

        if include_uni_labels:
            uni_path = os.path.join(self.wiki_uni_data_dir,f"{relation}.json")
            uni_data = load_json(uni_path)
            uni_labels = [ item["obj_label"] for item in uni_data]
            entity_data += uni_labels

        # 对候选词表做一次common vocab的交叉
        if common_vocab_intersection:
            entity_data = list(filter(filter_common_vocab_subset,entity_data))
        
        
        # 去重复
        entity_data = list(set(entity_data))
        index_list = self.get_vocab_indices(entity_data,tokenizer)
        return torch.tensor(index_list)


    def check_answer_entity_subset_of_common_vocab(self,):
        """测试answer entity是否是common vocab的真子集"""
        c_idx = set(self.get_common_vocab_indices(self.tokenizer).tolist())
        
        t = 0
        for relation in self.relations:
            a_idx = set(self.get_answer_entity_indices(relation,self.tokenizer,common_vocab_intersection=False).tolist())
            if not a_idx.issubset(c_idx):
                print(f"关系{relation}的type tokens不是common vocab子集 元素个数{len(a_idx)}，common vocab之外个数{len(a_idx-c_idx)}")
                diff = a_idx-c_idx
                diff_tokens = self.tokenizer.convert_ids_to_tokens(diff)
                print("前10个不同的token为")
                print(diff_tokens[:10])
                t+=1
        
        if t==0:
            print("answer entity是common vocab的真子集")


    def generate_model_bias(self, model, tokenizer:AutoTokenizer, prompt_only, save_path=None, vocab_subset_indices =None, return_logits=False,):
        """
        该函数目的是针对某个manual prompt作用下的model，得到一个bias的分布
        prompt_only形式为 "[MASK] was born in [MASK] ." 这种形式
        映入vocab_subset_indices 来对齐lama在logits上的过滤
        """
        model_input = tokenizer(prompt_only,return_tensors="pt")
        mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        input_token_ids = model_input["input_ids"][0].tolist()
        # 从处理好的输入ids中找出来最后一个mask_id
        y_mask_index = 0
        for i,e in enumerate(reversed(input_token_ids)):
            if e == mask_id:
                y_mask_index = -(i+1) 
                break
        assert y_mask_index < 0
        with torch.no_grad():
            model_input = {k:v.to(model.device) for k,v in model_input.items()}
            model_output = model(**model_input).logits
            distribution = model_output[0][y_mask_index].cpu()
            if vocab_subset_indices != None:
                distribution = distribution.index_select(dim=0,index=vocab_subset_indices)
            if not return_logits:
                final_output = torch.softmax(distribution,0)
            else:
                final_output = distribution
        
        if save_path!=None:
            probs = final_output.tolist()
            index = list(range(len(probs)))
            tokens = tokenizer.convert_ids_to_tokens(index)
            output = dict(zip(index,zip(probs,tokens)))
            self.create_dir(save_path)
            with open(save_path, "w") as f:
                # f.write("data format: vocab_index:[prob, vocab]\n")
                json.dump(output,f,indent=4)
        

        return final_output
   
    
    def get_template_bias_tensor(self, model,tokenizer, template, y_mask_index='unk'):
        """
        返回某个manual template在model下的bais_logits，该logits已经归一化了, 将会用于debias
        template 的形式为 [MASK] xxx xxx xxx [MASK]
        """
        # 统一框架
        # 1. transformer block
        # 2. transform layer
        # 3. decoder
        if isinstance(model, BertForMaskedLM):
            transformer_blocks = model.bert
            tranform_layer  = model.cls.predictions.transform
            decoder = model.cls.predictions.decoder
        elif isinstance(model, RobertaForMaskedLM):
            transformer_blocks = model.roberta
            tranform_layer = nn.Sequential(model.lm_head.dense, nn.GELU(), model.lm_head.layer_norm)
            decoder = model.lm_head.decoder
        
        model_input = tokenizer(template,return_tensors="pt")
        model_input = {key:value.to(model.device) for key,value in model_input.items()}

        transformer_output = transformer_blocks(**model_input)[0]
        
        if y_mask_index=='unk':
            # 找到[MASK]的位置
            y_mask_index = 0
            mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
            input_token_ids = model_input["input_ids"][0].tolist()
            for i,e in enumerate(reversed(input_token_ids)):
                if e == mask_id:
                    y_mask_index = -(i+1) 
                    break
            assert y_mask_index < 0
        else:
            # 已经传入了index的位置
            y_mask_index = y_mask_index
            
        transformer_output = transformer_output[0][y_mask_index]

        # 这个transform层的意义我不清楚，但是很重要，这才算是得到了特征向量
        bias_features = tranform_layer(transformer_output)
        norm2 = torch.norm(bias_features)
        bias_logits = decoder(bias_features) / norm2

        bias_features = bias_features.detach()
        bias_logits = bias_logits.detach()

        return bias_logits, bias_features    


    def get_template_bias_tensor_by_sample(self, promptModel:PromptModel, support_dataloader, num_tokens=5):
        """
        根据采样得到的数据，估计出bias tensor 和 bias logits
        num_tokens 用于softtemplate
        """
        promptModel.eval()
        # 统一框架
        # 1. transformer block
        # 2. transform layer
        # 3. decoder
        model = promptModel.plm
        if isinstance(model, BertForMaskedLM):
            transformer_blocks = model.bert
            tranform_layer  = model.cls.predictions.transform
            decoder = model.cls.predictions.decoder
        elif isinstance(model, RobertaForMaskedLM):
            transformer_blocks = model.roberta
            tranform_layer = nn.Sequential(model.lm_head.dense, nn.GELU(), model.lm_head.layer_norm)
            decoder = model.lm_head.decoder
        
        # 暂时只支持prompt for classfication
        assert isinstance(promptModel, PromptForClassification)
        all_prediction_vectors = None
        for inputs in support_dataloader:
            inputs = inputs.to(promptModel.device)
            # 以下逻辑是把openprompt的forward抄了过来
            batch = promptModel.template.process_batch(inputs)
            input_batch = {key: batch[key] for key in batch if key in promptModel.prompt_model.forward_keys}
            transformer_output = transformer_blocks(**input_batch)[0]
            # 对于softtemplate，这里有一个后处理
            if isinstance(promptModel.template, SoftTemplate):
                # 去掉softembedding
                transformer_output = transformer_output[:,num_tokens:,:]
            mask_token_transformer_output = transformer_output[torch.where(inputs['loss_ids']>0)]
            mask_token_features = tranform_layer(mask_token_transformer_output)
            
            # linear层之后的特征向量, 论文理论框架中用于debias的向量
            if all_prediction_vectors==None:
                all_prediction_vectors = mask_token_features
            else:
                all_prediction_vectors = torch.cat((all_prediction_vectors, mask_token_features), dim=0)
            
        # 求平均vector以及它的模长
        avg_prediction_vector = torch.mean(all_prediction_vectors,dim=0)
        avg_norm = torch.norm(avg_prediction_vector)
        # avg_prediction_vector = avg_prediction_vector / avg_norm
        
        # 求平均logit值
        avg_logits = torch.mean(decoder(all_prediction_vectors), dim=0)

        # 对平均logit正则化
        normalized_avg_logits = avg_logits / avg_norm

        normalized_avg_logits = normalized_avg_logits.detach()
        avg_prediction_vector = avg_prediction_vector.detach()

        return normalized_avg_logits, avg_prediction_vector
        

    def generate_image(self,before_debias_preds, after_debias_preds, labels, model_bias, save_path="test.png"):
        """
        生成给定数据集在debias 前后的一个图像 均为typeQuery
        布局如下
        样本空间分布                 样本空间分布+debias前的preds分布

        样本空间分布+labels分布       样本空间分布+debias后的preds分布

        preds的散点大小分布           preds==labels的分布

        输入： debais前的preds和debias后的preds, label,model_bias
        """

        labels = labels
        preds = before_debias_preds
        count_labels  = Counter(labels)
        count_preds  = Counter(preds)
        # bound = 5*2

        raw_sorted_labels = sorted(count_labels.items())
        raw_sorted_preds = sorted(count_preds.items())

        fig, axs = plt.subplots(3,2,figsize=(14,15))
        axs[0,0].plot(model_bias.numpy(),"o")
        axs[0,1].plot(model_bias.numpy(),"o")
        axs[1,0].plot(model_bias.numpy(),"o")
        axs[1,1].plot(model_bias.numpy(),"o")
        raw_label_x = [i[0] for i in raw_sorted_labels]
        raw_label_y = [model_bias[i[0]].item() for i in raw_sorted_labels]
        raw_pred_x = [i[0] for i in raw_sorted_preds]
        raw_pred_y = [model_bias[i[0]].item() for i in raw_sorted_preds]
        axs[0,1].plot(raw_pred_x,raw_pred_y,"o")
        axs[1,0].plot(raw_label_x,raw_label_y,"o",color="pink")

        # 加入debias后的数据
        debias_preds = after_debias_preds
        count_debias_preds = Counter(debias_preds)
        raw_sorted_debais_preds = sorted(count_debias_preds.items())
        raw_debais_preds_x = [i[0] for i in raw_sorted_debais_preds]
        raw_debais_preds_y = [model_bias[i[0]].item() for i in raw_sorted_debais_preds]
        axs[1,1].plot(raw_debais_preds_x, raw_debais_preds_y,'o',color="orange")

        # axs1[0].scatter(list(range(len(model_bias))),model_bias,alpha=0.2)
        c = [i[1] for i in raw_sorted_debais_preds]
        axs[2,0].scatter(raw_debais_preds_x, raw_debais_preds_y,s=c,)

        acc_index = np.where(np.array(debias_preds)==np.array(labels))
        acc_labels = np.array(labels)[acc_index]
        count_acc_labels = Counter(acc_labels)
        count_acc_labels = sorted(count_acc_labels.items())
        acc_labels_x = [i[0] for i in count_acc_labels]
        acc_labels_y = [model_bias[i[0]] for i in count_acc_labels]
        acc_labels_num = np.array([i[1] for i in count_acc_labels])
        axs[2,1].scatter(list(range(len(model_bias))),model_bias, alpha=0.2)
        axs[2,1].scatter(acc_labels_x,acc_labels_y,cmap="Oranges",s=acc_labels_num*10)
        plt.savefig(save_path)


    def general_continue_prompt(self, dataset, embedding_save_dir,num_tokens=5, continue_prompt="optiprompt", random_init="none", evaluate_mode=False,):
        # 获取预训练模型
        plm, tokenizer, model_config, WrapperClass = self.plm,self.tokenizer,self.model_config,self.WrapperClass

        # 是否要对模型的weight随机初始化
        if random_init=="all":
            plm = AutoModelForMaskedLM.from_config(model_config)

        # 构造prompt

        current_template = '{"soft":None,"duplicate":' + str(num_tokens) + '}'
        current_template += ' sentence1: {"placeholder":"text_a"} . sentence2: {"placeholder":"text_b"} . word: {"meta":"word"} {"mask"} .'
        prompt_template = MixedTemplate(model=plm,tokenizer=tokenizer,text=current_template)
            # prompt_template.soft_embedding.weight.data.normal_(mean=0.0, std=model_config.initializer_range)
        
        # 加载数据集
        # max_tokens_len = self.compute_max_pad(dataset, WrapperClass, tokenizer, prompt_template)
        max_tokens_len = 128
        train_dataloader = PromptDataLoader(dataset=dataset["train"], template=prompt_template, tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass, max_seq_length=max_tokens_len,
                batch_size=16)
        
        valid_dataloader = PromptDataLoader(dataset=dataset["validation"], template=prompt_template, tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass, max_seq_length=max_tokens_len,
                batch_size=16)
        test_dataloader = PromptDataLoader(dataset=dataset["test"], template=prompt_template, tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass, max_seq_length=max_tokens_len,
                batch_size=16)
        
        # 定义verbalizer
        myverbalizer = ManualVerbalizer(tokenizer, num_classes=2,
                        label_words=[["False"], ["True"]])
        # 准备prompt Model
        promptModel = PromptForClassification(
            template = prompt_template,
            plm = plm,
            verbalizer=myverbalizer,
            freeze_plm=True,
            
        )

        promptModel.cuda()

        best_ckpt = os.path.join(embedding_save_dir, f"weight_save/model_full-prompt_random_init.ckpt")
        if evaluate_mode==False:
            # 开始训练
            loss_func = torch.nn.CrossEntropyLoss()
            no_decay = ['bias', 'LayerNorm.weight']

            optimizer_grouped_parameters = [
                {'params': [p for n,p in promptModel.template.named_parameters() if "raw_embedding" not in n]}
            ]

            optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
            t_total = len(dataset["train"]) * self.num_epochs
            scheduler = get_cosine_schedule_with_warmup(optimizer, int(t_total * self.warmup_proportion), t_total)

            time_start = time()
            best_acc = 0
            best_epoch = -1  
            self.create_dir(best_ckpt)
            for epoch in range(self.num_epochs):
                promptModel.train()
                tot_loss = 0
                for step, inputs in enumerate(train_dataloader):
                    inputs = inputs.cuda()
                    logits = promptModel(inputs)
                    # mask_logits = logits[torch.where(inputs['loss_ids']>0)]
                    loss = loss_func(logits, inputs["label"])
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    tot_loss += loss.item()
                    # print(tot_loss/(step+1))
                # validate it
            
                acc,_ =  self.evaluate(promptModel, valid_dataloader)
                if acc > best_acc:
                    # 保存最好的epoch
                    best_epoch = epoch
                    best_acc = acc
                    if continue_prompt=="prefix":
                        soft_embeds = promptModel.template.state_dict()["soft_embeds"].clone()
                    elif continue_prompt=='optiprompt':
                        soft_embeds = promptModel.template.state_dict()["soft_embedding.weight"].clone()
                    torch.save(soft_embeds,best_ckpt)
                    

            print("best epoch {} valid precision: {}".format(best_epoch,best_acc,))
            model_bias_output, bias_logits, bias_vector = self.get_continue_prompt_bias(promptModel,continue_prompt,num_tokens,best_ckpt,prompt_template)
            debias_output = self.evaluate(promptModel, test_dataloader[dataset], bias_logits=bias_logits, bias_vector=bias_vector, scaling_debiased_logit=True,soft_tempalte_ckpt=best_ckpt,continue_prompt=continue_prompt,num_tokens=num_tokens)
            origin_output = self.evaluate(promptModel, test_dataloader[dataset], vocab_subset_indices=subset_indices, soft_tempalte_ckpt=best_ckpt,continue_prompt=continue_prompt,num_tokens=num_tokens)
            return model_bias_output,(debias_output,origin_output)
    

    def test_wic(self):
        # 加载wic数据集e
        # wic = load_dataset("super_glue","wic",cache_dir="home/jiao/.cache/huggingface/datasets")
        # wic.save_to_disk("/home/jiao/datasets/super_glue/wic")
        wic = load_from_disk("/home/jiao/datasets/super_glue/wic")
        dataset = {}
        for split in ['train', 'validation', 'test']:
            dataset[split] = []
            for data in wic[split]:
                input_example = InputExample(text_a = data['sentence1'], text_b = data['sentence2'], meta={"word":data["word"]},label=int(data['label']), guid=data['idx'])
                dataset[split].append(input_example)
        wic_output_dir = "/home/jiao/code/prompt/PromptBias/TruelyKnow/outputs/superglue/wic"
        self.general_continue_prompt(dataset, wic_output_dir)


    def calibrate(self, prompt_model: PromptForClassification, dataloader: PromptDataLoader) -> torch.Tensor:
        r"""Calibrate. See `Paper <https://arxiv.org/abs/2108.02035>`_
        
        Args:
            prompt_model (:obj:`PromptForClassification`): the PromptForClassification model.
            dataloader (:obj:`List`): the dataloader to conduct the calibrate, could be a virtual one, i.e. contain an only-template example.
        
        Return:
            (:obj:`torch.Tensor`) A tensor of shape  (vocabsize) or (mask_num, vocabsize), the logits calculated for each word in the vocabulary
        """
        all_logits = []
        prompt_model.eval()
        for batch in tqdm(dataloader,desc='ContextCali'):
            batch = batch.to(prompt_model.device)
            logits = prompt_model.forward_without_verbalize(batch)
            all_logits.append(logits.detach())
        all_logits = torch.cat(all_logits, dim=0)
        return all_logits


    def prior_on_yahoo(self):
        # manual prompt  A {"mask"} news : {"placeholder": "text_a"} {"placeholder": "text_b"}
        # manual verblizer one to one
        """
        先导试验：对比vector debiasing 和 calibration 谁的debias效果好
        """
        # 准备好数据集
        dataset = {}
        raw_dataset = load_from_disk("/home/jiao/code/prompt/PromptBias/TruelyKnow/OpenPrompt/datasets/TextClassification/yahoo_answers_topics")
        dataset["train"] = list(map(YahooAnswersTopicsProcessor().transform,raw_dataset["train"]))
        dataset["test"] = list(map(YahooAnswersTopicsProcessor().transform,raw_dataset["test"]))[:100]
        class_labels = YahooAnswersTopicsProcessor().get_labels()

        # 假设数据已经梳理好了
        manual_tempalte = '[ Category : {"mask"} ] {"placeholder": "text_a"} {"placeholder": "text_b"}'
        prompt_template = ManualTemplate(tokenizer=self.tokenizer, text=manual_tempalte)

        # 定义prompt_pnly_tempalte用于计算bias
        # prompt_only_tempalte = 'A {"mask"} news : '+ '{} {}'.format(self.tokenizer.mask_token, self.tokenizer.mask_token)
        prompt_only_tempalte = '[ Category : {} ] {} {}'.format(self.tokenizer.mask_token, self.tokenizer.mask_token,self.tokenizer.mask_token)
        bias_logits, bias_vector = self.get_template_bias_tensor(self.plm,self.tokenizer,template=prompt_only_tempalte,y_mask_index=4)

        # 定义verbalizer
        myverbalizer = ManualVerbalizer(self.tokenizer, classes=class_labels).from_file("/home/jiao/code/prompt/PromptBias/TruelyKnow/OpenPrompt/scripts/TextClassification/yahoo_answers_topics/manual_verbalizer.json")
        # 计算得到正常精度和debias精度
        test_dataloader = PromptDataLoader(dataset=dataset["test"], template=prompt_template, tokenizer=self.tokenizer,
            tokenizer_wrapper_class=self.WrapperClass, max_seq_length=128, decoder_max_length=3,
            batch_size=32,shuffle=False,teacher_forcing=False, predict_eos_token=False,
            truncate_method="tail")
        
        # 准备model
        prompt_model = PromptForClassification(plm=self.plm,template=prompt_template, verbalizer=myverbalizer, freeze_plm=True,plm_eval_mode=True)
        prompt_model=  prompt_model.cuda()

         # 开始预测
        
        # 采用prompt-only input来估计先验分布
        cc_logits = torch.norm(bias_vector)*bias_logits
        cc_logits = cc_logits.to(prompt_model.device)
        acc_calibration_v1, _ = self.evaluate(prompt_model,test_dataloader,calibration=True, calib_logits=cc_logits)

        # 采用采样来估计先验分布
        from openprompt.data_utils.data_sampler import FewShotSampler
        support_sampler = FewShotSampler(num_examples_total=200, also_sample_dev=False)
        dataset['support'] = support_sampler(dataset['train'], seed=7)
        for example in dataset['support']:
            example.label = -1 # remove the labels of support set for clarification
        support_dataloader = PromptDataLoader(dataset=dataset["support"], template=prompt_template, tokenizer=self.tokenizer,
            tokenizer_wrapper_class=self.WrapperClass, max_seq_length=128, decoder_max_length=3,
            batch_size=32,shuffle=False, teacher_forcing=False, predict_eos_token=False,
            truncate_method="tail")
        cc_logits = self.calibrate(prompt_model, support_dataloader).mean(dim=0)
        acc_calibration_v2, _ = self.evaluate(prompt_model,test_dataloader,calibration=True, calib_logits=cc_logits)
        
        acc_raw, (probs, allpreds, alllabels) = self.evaluate(prompt_model,test_dataloader, vocab_subset_indices=None)
        acc_debias,(probs, allpreds, alllabels) = self.evaluate(prompt_model,test_dataloader, bias_logits=bias_logits, bias_vector=bias_vector,scaling_debiased_logit=True, vocab_subset_indices=None)

        print("原始精度 {} debias精度 {} 校验精度v1{}  校验精度v1{}".format(acc_raw, acc_debias, acc_calibration_v1, acc_calibration_v2))
            # acc_raw,(probs, allpreds, alllabels) = self.evaluate(promptModel,test_dataloader, vocab_subset_indices=subset_indices)
        

    def manual_prompt(self, relation, debias=False, vocab_subset_filter=True, vocab_subset="common_vocab",test_data_path=None, bias_tensor=None, embeddings_renormalize=False, prompt="LAMA" ):
        """
        vocab_subset_filter用于控制在evaluate的时候是否使用vocab_subset来过滤掉一些logits
        在数据集导入的时候, 是肯定要用vocab_subset来过滤的, 因为lama也这样子过滤了的
        vocab_subset用于控制过滤使用的词汇表，lama采用的是不同语言模型的共同词表，机默认参数的 `common_vocab`
            除此之外，还可以指定成 `answer_type_tokens` 用于表示特定类型的答案实体空间，例如对于P19，它的answer_type_tokens是323个城市
            可以使用self.get_answer_entity_indices来获取
        embeddigns_renormalize用来控制是否对plm的embedding做归一化
        不支持直接传入bias_tensor的用法
        """
        # 获取预训练模型
        # plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")
        plm = self.plm
        tokenizer = self.tokenizer
        model_config = self.model_config
        WrapperClass = self.WrapperClass
        model = plm
        

        if embeddings_renormalize:
            # 修改模型embeddings
            config = model_config
            renormal = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            renormal.weight.data = model.cls.predictions.decoder.weight.data.clone()
            renormal.weight.data = F.normalize(renormal.weight.data, p=2,dim=1)
            # renormal.bias = model.cls.predictions.decoder.bias
            output_embedding_backup = model.get_output_embeddings()
            model.set_output_embeddings(renormal)
            model.bert.embeddings.word_embeddings.weight = renormal.weight

        # 根据当前模型，构造出来subset indices
        if vocab_subset_filter:
            if vocab_subset=="common_vocab":
                subset_indices = self.get_common_vocab_indices(tokenizer)
            elif vocab_subset=="answer_type_tokens":
                subset_indices = self.get_answer_entity_indices(relation, tokenizer)
            else:
                raise ValueError(f"参数值{vocab_subset}不符合要求")
        else:
            subset_indices = None

        # 构造数据集        
        dataset = {}
        
        # 默认测试集
        if test_data_path is None:
            lama_data_path = os.path.join(self.lama_data_dir,f"{relation}.jsonl")
            test_data_path = lama_data_path
        # lama_data_path = os.path.join(self.auto_prompt_dir,relation,"test.jsonl")
        

        # 根据prompt参数选择合适的手工模板
        templates = None
        if prompt=="LAMA":
            templates = self.lama_template
        elif prompt=="LPAQA":
            templates = self.LPAQA_template
        elif prompt=="AutoPrompt":
            if self.model_name.startswith("roberta"):
                templates = self.AutoPrompt_tempalte_roberta
            elif self.model_name.startswith("bert"):
                templates = self.AutoPrompt_template
            else:
                raise(f"prompt参数出错: {prompt}")
        else:
            raise(f"prompt参数出错: {prompt}")

        # 定义手动模板
        raw_template = templates[relation]
        # 在[x] 和下一个token之间的那部分被认为是autoprompt的post_fix
        # 例如P37 [X]inen dialects resembled officially exclusively [Y].
        first_token = raw_template.split()[0]
        autoprompt_postfix = first_token[3:] if prompt=="AutoPrompt" else ''

        raw_test_data, _ = self.load_data(test_data_path, vocab_subset=self.lama_vocab_subset,)
        dataset["test"] = self.wrap_input_examples(raw_test_data, tokenizer,autoprompt_postfix=autoprompt_postfix)
        sub_labels = [d["sub_label"] for d in raw_test_data]


        # 将模板转换成openprompt
        template = raw_template.replace(first_token,'{"placeholder":"text_a"}') if prompt=="AutoPrompt" else \
                    raw_template.replace("[X]", '{"placeholder":"text_a"}')
        template = template.replace("[Y]", '{"mask"}')
        prompt_template = ManualTemplate(tokenizer=tokenizer, text=template)

        # 定义prompt_pnly_tempalte用于计算bias
        prompt_only_tempalte = raw_template.replace("[X]",self.tokenizer.mask_token).replace("[Y]", self.tokenizer.mask_token)
        bias_logits, bias_vector = self.get_template_bias_tensor(model,tokenizer,template=prompt_only_tempalte)
    


        # 加载数据集
        # 计算lama需要的最大pad数量
        max_tokens_len = self.compute_max_pad(dataset, WrapperClass, tokenizer, prompt_template)

        # Note:这里不能shuffle，否则导出的sub_label顺序将会出错
        test_dataloader = PromptDataLoader(dataset=dataset["test"], template=prompt_template, tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass, max_seq_length=max_tokens_len, 
            batch_size=16,shuffle=False)
        

        # 准备prompt Model
        promptModel = PromptModel(
            template = prompt_template,
            plm = plm,
            freeze_plm=True
        )
        promptModel.cuda()

        # 开始预测
        if debias:
            acc,(probs, allpreds, alllabels) = self.evaluate(promptModel,test_dataloader, bias_logits=bias_logits, bias_vector=bias_vector,scaling_debiased_logit=True, vocab_subset_indices=subset_indices)
        else:
            acc,(probs, allpreds, alllabels) = self.evaluate(promptModel,test_dataloader, vocab_subset_indices=subset_indices)

        # print("precision: {}".format(acc))

        if embeddings_renormalize==True:
            output_embedding_backup = output_embedding_backup.to(self.plm.device)
            self.plm.set_output_embeddings(output_embedding_backup)
            self.plm.bert.embeddings.word_embeddings.weight = output_embedding_backup.weight

        return round(acc,5), (probs, sub_labels, allpreds, alllabels)


    def run_manual_prompt_all_relation(self,vocab_subset="common_vocab", embeddings_renormalize=False):
        save_path = self.work_dir + f"/outputs/renormalize/lama_{vocab_subset}_renormalize.csv" if embeddings_renormalize else self.work_dir + f"/outputs/renormalize/lama_{vocab_subset}.csv"
        f = open(save_path,"w")
        f.write("relation,acc\n")
        pbar = tqdm(total=len(self.relations))
        for relation in self.relations:
            lama_data_path = os.path.join(self.lama_data_dir,f"{relation}.jsonl")
            uni_data_path = os.path.join(self.wiki_uni_data_dir,f"{relation}.json")
            data_path =lama_data_path
            acc,_ = self.manual_prompt(relation, vocab_subset=vocab_subset,embeddings_renormalize=embeddings_renormalize,test_data_path=lama_data_path)
            print(f"关系{relation} 精度为{acc}")
            pbar.update(1)

            f.write(f"{relation},{acc}\n")
        f.close
        pbar.close()
            

    def experiment_renormal_vector_debais_for_manual_prompt(self, vocab_subset_filter=True, vocab_subset="answer_type_tokens",embeddings_renormalize=False,ctrl_code=[1,1,1], manual_prompt="LAMA"):
        """
        ctrl_code 代表需要测试的数据集， 包扩[lama, wiki-uni, lama_whu]
        """
        if embeddings_renormalize==True:
             # 修改模型embeddings
            config = self.model_config
            renormal = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            renormal.weight.data = self.plm.cls.predictions.decoder.weight.data.clone()
            renormal.weight.data = F.normalize(renormal.weight.data, p=2,dim=1)
            # renormal.bias = model.cls.predictions.decoder.bias
            output_embedding_backup = self.plm.get_output_embeddings()
            self.plm.set_output_embeddings(renormal)
            self.plm.bert.embeddings.word_embeddings.weight = renormal.weight
            
        root_dir = self.work_dir + f"/outputs/openprompt/manual_prompt/{self.model_name}"

        save_dir = f"{manual_prompt}/debias_{vocab_subset}/"
        if embeddings_renormalize==True:
            save_dir += "renormalize_embedding"
        else:
            save_dir += "origin_embedding"
            
        dataset_names = ["lama","uni","lama_whu"]
        dataset_num = len(ctrl_code)
        files = [None]*dataset_num
        img_save_dir = [None]*dataset_num

        KL_save_path = os.path.join(root_dir, f"{save_dir}/KL.csv")
        self.create_dir(KL_save_path)
        KL_save_f = open(KL_save_path,"w")
        KL_save_f.write("Relation")
        preds_save_path = os.path.join(root_dir, f"{save_dir}/preds.pt")


        total_diff = [[],[],[]]
        # 保存所有精度和所有KL散度
        output_save = { 
                        "LAMA": {"P":[], "P_d":[],"KL":[], "KL_d":[]},
                        "WIKI-UNI":{"P":[], "P_d":[],"KL":[], "KL_d":[]},
                        "LAMA-WHU":{"P":[], "P_d":[],"KL":[], "KL_d":[]}
                      }
 
        # 保存所有预测结果
        preds_save = {
            "LAMA": {},
            "WIKI-UNI": {},
            "LAMA-WHU":{},
        }

        for i in range(dataset_num):
            save_path = os.path.join(root_dir, f"{save_dir}/{dataset_names[i]}_difference_debias.csv") 
            self.create_dir(save_path)
            img_save_path =  os.path.join(root_dir, f"{save_dir}/img/{dataset_names[i]}")
            img_save_dir[i] = img_save_path
            if ctrl_code[i]==1:
                f = open(save_path,"w")
                f.write("relation,diff,origin,debias\n")
                files[i] = f

                KL_save_f.write(f",{dataset_names[i]}_before,{dataset_names[i]}_after")



        pbar = tqdm(total=len(self.relations)*sum(ctrl_code))
        
        
        # 用来表示第一个要测试的数据集
        first_dataset_index = ctrl_code.index(1)

        # 根据prompt参数选择合适的手工模板
        templates = None
        if manual_prompt=="LAMA":
            templates = self.lama_template
        elif manual_prompt=="LPAQA":
            templates = self.LPAQA_template
        elif manual_prompt=="AutoPrompt":
            templates = self.AutoPrompt_template
        else:
            raise(f"prompt参数出错: {manual_prompt}")

        for relation in self.relations:
            lama_data_path = os.path.join(self.lama_data_dir,f"{relation}.jsonl")
            uni_data_path = os.path.join(self.wiki_uni_data_dir,f"{relation}.json")
            lama_whu_data_path = os.path.join(self.lama_whu_data_dir,f"{relation}.jsonl")
            data_paths = [lama_data_path,uni_data_path,lama_whu_data_path]

            raw_template =templates[relation]
            prompt_only_tempalte = raw_template.replace("[X]",self.tokenizer.mask_token).replace("[Y]", self.tokenizer.mask_token)
            
            subvocab_tokens_indices = self.get_answer_entity_indices(relation, self.tokenizer) \
                                                if vocab_subset == "answer_type_tokens" else \
                                    self.get_common_vocab_indices(self.tokenizer)
            subvocab_indices_list = subvocab_tokens_indices.tolist()
            bias_tensor = self.generate_model_bias(
                self.plm, self.tokenizer, prompt_only_tempalte, 
                vocab_subset_indices=subvocab_tokens_indices, 
                    return_logits=True)
            
            bias_probs = torch.softmax(bias_tensor,dim=-1).tolist()
            # 用于计算KL散度
            bias_dis = defaultdict(int, zip(subvocab_indices_list,bias_probs))

            for i in range(len(data_paths)):
                if ctrl_code[i]==0:
                    continue
                f = files[i]
                acc_origin,(probs_before, sub_labels, preds_before, labels) = self.manual_prompt(
                    relation, debias=False, 
                    vocab_subset_filter=vocab_subset_filter,
                    vocab_subset=vocab_subset,
                    test_data_path=data_paths[i],
                    prompt=manual_prompt)
                acc_debias,(probs_after,_, preds_after, _) = self.manual_prompt(
                    relation, debias=True, 
                    vocab_subset_filter=vocab_subset_filter, 
                    vocab_subset=vocab_subset,
                    test_data_path=data_paths[i],
                    prompt=manual_prompt)
                
                dataset = list(output_save.keys())[i]
                output_save[dataset]["P"].append(acc_origin)
                output_save[dataset]["P_d"].append(acc_debias)
                
                # 保存preds结果
                obj_labels = self.tokenizer.convert_ids_to_tokens(labels)
                raw_preds = self.tokenizer.convert_ids_to_tokens(preds_before)
                debiased_preds = self.tokenizer.convert_ids_to_tokens(preds_after)
                indices = list(range(len(sub_labels)))
                preds_save[dataset][relation] = {"data":{"index":indices,"sub_label":sub_labels,"obj_labels":obj_labels,"raw_preds":raw_preds,"debiased_preds":debiased_preds}}

                # 计算出来分布，用于绘图和计算KL散度
                num_data = len(preds_before)
                
                probs_dis_before = torch.softmax(probs_before,dim=-1).tolist()
                probs_dis_after = torch.softmax(probs_after,dim=-1).tolist()
                KL_before_list = []
                KL_after_list = []
                for index in range(num_data):
                    probs_dis_before_i = probs_dis_before[index]
                    probs_dis_after_i = probs_dis_after[index]
                    probs_dis_before_i = defaultdict(int, zip(subvocab_indices_list,probs_dis_before_i))
                    probs_dis_after_i = defaultdict(int, zip(subvocab_indices_list,probs_dis_after_i))
                    KL_before_list.append(self.kl_divergence(bias_dis, probs_dis_before_i))
                    KL_after_list.append(self.kl_divergence(bias_dis, probs_dis_after_i))

                KL_before = round(np.mean(KL_before_list),5)
                KL_after = round(np.mean(KL_after_list),5) 

                output_save[dataset]["KL"].append(KL_before)
                output_save[dataset]["KL_d"].append(KL_after)

                if i==first_dataset_index:
                    KL_save_f.write(f"\n{relation},{KL_before},{KL_after}")
                else:
                    KL_save_f.write(f",{KL_before},{KL_after}")

                # print(f"debais前KL散度为{KL_before} debias后KL散度为{KL_after}")


                # preds和labels均是原始词汇表里面的，绘图前需要转换成subset_vocab的索引
                preds_before = [subvocab_indices_list.index(i) for i in preds_before]
                preds_after = [subvocab_indices_list.index(i) for i in preds_after]

                labels = [subvocab_indices_list.index(i) for i in labels]

                img_save_path = os.path.join(img_save_dir[i], relation+".png")
                self.create_dir(img_save_path)

                self.generate_image(preds_before,preds_after,labels,model_bias=bias_tensor, save_path=img_save_path)


                diff = round(acc_debias - acc_origin,5)
                
                print("{} 原始精度{} debias精度{}".format(
                    dataset_names[i],
                    round(acc_origin*100,2),
                    round(acc_debias*100,2)))
                total_diff[i].append(diff)
                f.write(f"{relation},{diff},{acc_origin},{acc_debias}\n")
                f.flush()
                pbar.update(1)
                
                postfox = {
                    "lama_improve": 0 if ctrl_code[0]==0 or len(total_diff[0])==0 else round(sum(total_diff[0])/len(total_diff[0]), 3) * 100,
                    "uni_improve": 0 if ctrl_code[1]==0 or len(total_diff[1])==0 else round(sum(total_diff[1])/len(total_diff[1]), 3) * 100,
                    "lama_whu_improve": 0 if ctrl_code[2]==0 or len(total_diff[2])==0 else round(sum(total_diff[2])/len(total_diff[2]), 3) * 100,
                    }

                pbar.set_postfix(postfox)
        
        pbar.close()
        
        for f in files:
            if f != None:
                f.close()
        
        KL_save_f.close()

        # 处理输出
        for i,dataset in enumerate(output_save.keys()):
            if ctrl_code[i]==0:
                continue
            avg_p = np.mean(output_save[dataset]["P"])
            avg_p_d = np.mean(output_save[dataset]["P_d"])
            avg_KL = np.mean(output_save[dataset]["KL"])
            avg_KL_d = np.mean(output_save[dataset]["KL_d"])
            self.add_output_item(self.model_name, dataset,prompt=manual_prompt,result=[avg_p,avg_p_d,avg_KL,avg_KL_d])
        
        self.print_output()

        torch.save(preds_save,preds_save_path)
                


        if embeddings_renormalize==True:
            output_embedding_backup = output_embedding_backup.to(self.plm.device)
            self.plm.set_output_embeddings(output_embedding_backup)
            self.plm.bert.embeddings.word_embeddings.weight = output_embedding_backup.weight


    def experiment_renormal_vector_debias_for_continue_prompt(self, vocab_subset_filter=True, vocab_subset="answer_type_tokens",embeddings_renormalize=False,ctrl_code=[1,1,1], continue_prompt="optiprompt", num_tokens=5, repeat_times=3, evaluate_mode=False):
        """
        ctrl_code 代表需要测试的数据集， 包扩[lama, wiki-uni, lama_whu]
        continue_prompt 表示使用某个连续模板 支持prefix和optiprompt
        repeated_times用于表示重复次数
        evaluate_model 为true表示不训练，加载最好的epoch
        """

        pbar = tqdm(total=len(self.relations)*repeat_times)
        
        if embeddings_renormalize==True:
             # 修改模型embeddings
            config = self.model_config
            renormal = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            renormal.weight.data = self.plm.cls.predictions.decoder.weight.data.clone()
            renormal.weight.data = F.normalize(renormal.weight.data, p=2,dim=1)
            # renormal.bias = model.cls.predictions.decoder.bias
            output_embedding_backup = self.plm.get_output_embeddings()
            self.plm.set_output_embeddings(renormal)
            self.plm.bert.embeddings.word_embeddings.weight = renormal.weight

        # 保存self.output_result的当前值
        self_output_result_bk = copy.copy(self.output_result)

        output_results = []
        for index in range(1,repeat_times+1):
            self.clear_output()
            root_dir = self.work_dir + f"/outputs/openprompt/continue_prompt/{self.model_name}"
            
            save_dir = continue_prompt+f"_{num_tokens}" + f"/debias_{vocab_subset}/"
            if embeddings_renormalize==True:
                save_dir += "renormalize_embedding"
            else:
                save_dir += "origin_embedding"
            
            save_dir += f"/exp_{index}"
            
            dataset_names = ["lama","uni","lama_whu"]
            dataset_num = len(ctrl_code)
            files = [None]*dataset_num
            img_save_dir = [None]*dataset_num

            KL_save_path = os.path.join(root_dir,f"{save_dir}/KL.csv")
            self.create_dir(KL_save_path)
            KL_save_f = open(KL_save_path,"w")
            KL_save_f.write("Relation")
            preds_save_path = os.path.join(root_dir, f"{save_dir}/preds.pt")

            total_diff = [[],[],[]]
            # 保存所有精度和所有KL散度
            output_save = { 
                            "LAMA": {"P":[], "P_d":[],"KL":[], "KL_d":[]},
                            "WIKI-UNI":{"P":[], "P_d":[],"KL":[], "KL_d":[]},
                            "LAMA-WHU":{"P":[], "P_d":[],"KL":[], "KL_d":[]}
                        }

            # 保存所有预测结果
            preds_save = {
                "LAMA": {},
                "WIKI-UNI": {},
                "LAMA-WHU":{},
            }
            
            for i in range(dataset_num):
                save_path = os.path.join(root_dir, f"{save_dir}/{dataset_names[i]}_difference_debias.csv")
                self.create_dir(save_path)
                img_save_path = os.path.join(root_dir, f"{save_dir}/img/{dataset_names[i]}")
                img_save_dir[i] = img_save_path
                if ctrl_code[i]==1:
                    f = open(save_path,"w")
                    f.write("relation,diff,origin,debias\n")
                    files[i] = f

                    KL_save_f.write(f",{dataset_names[i]}_before,{dataset_names[i]}_after")

            first_dataset_index = ctrl_code.index(1)

            for relation in self.relations:
                
                model_bias, results = self.continue_prompt(relation, embedding_save_dir=os.path.join(root_dir,save_dir),vocab_subset_filter=vocab_subset_filter,
                                                                        vocab_subset=vocab_subset,continue_prompt=continue_prompt,num_tokens=num_tokens, evaluate_mode=evaluate_mode)
                
                subvocab_tokens_indices = self.get_answer_entity_indices(relation, self.tokenizer) \
                                                        if vocab_subset == "answer_type_tokens" else \
                                            self.get_common_vocab_indices(self.tokenizer)
                subvocab_indices_list = subvocab_tokens_indices.tolist()
                subvocab_tokens_indices = subvocab_tokens_indices.to(model_bias.device)
                bias_tensor = model_bias.index_select(index=subvocab_tokens_indices,dim=0).cpu().detach()
                bias_probs = torch.softmax(bias_tensor,dim=-1).tolist()
                # 用于计算KL散度
                bias_dis = defaultdict(int, zip(subvocab_indices_list,bias_probs))
                # 计算出来分布，用于绘图

                for i in range(len(dataset_names)):
                    f = files[i]
                    debias_res,origin_res = results[dataset_names[i]]
                    acc_origin,(probs_before, sub_labels, preds_before, labels) = origin_res
                    acc_debias,(probs_after,_,  preds_after, _) = debias_res
                    dataset = list(output_save.keys())[i]
                    output_save[dataset]["P"].append(acc_origin)
                    output_save[dataset]["P_d"].append(acc_debias)

                    # 保存preds结果
                    obj_labels = self.tokenizer.convert_ids_to_tokens(labels)
                    raw_preds = self.tokenizer.convert_ids_to_tokens(preds_before)
                    debiased_preds = self.tokenizer.convert_ids_to_tokens(preds_after)
                    indices = list(range(len(sub_labels)))
                    preds_save[dataset][relation] = {"data":{"index":indices,"sub_label":sub_labels,"obj_labels":obj_labels,"raw_preds":raw_preds,"debiased_preds":debiased_preds}}

                    # 计算出来分布，用于绘图和计算KL散度
                    num_data = len(preds_before)

                    probs_dis_before = torch.softmax(probs_before,dim=-1).tolist()
                    probs_dis_after = torch.softmax(probs_after,dim=-1).tolist()
                    KL_before_list = []
                    KL_after_list = []
                    for index in range(num_data):
                        probs_dis_before_i = probs_dis_before[index]
                        probs_dis_after_i = probs_dis_after[index]
                        probs_dis_before_i = defaultdict(int, zip(subvocab_indices_list,probs_dis_before_i))
                        probs_dis_after_i = defaultdict(int, zip(subvocab_indices_list,probs_dis_after_i))
                        KL_before_list.append(self.kl_divergence(bias_dis, probs_dis_before_i))
                        KL_after_list.append(self.kl_divergence(bias_dis, probs_dis_after_i))

                    KL_before = round(np.mean(KL_before_list),5)
                    KL_after = round(np.mean(KL_after_list),5) 

                    output_save[dataset]["KL"].append(KL_before)
                    output_save[dataset]["KL_d"].append(KL_after)

                    if i==first_dataset_index:
                        KL_save_f.write(f"\n{relation},{KL_before},{KL_after}")
                    else:
                        KL_save_f.write(f",{KL_before},{KL_after}")

                    # print(f"debais前KL散度为{KL_before} debias后KL散度为{KL_after}")


                    # preds和labels均是原始词汇表里面的，绘图前需要转换成subset_vocab的索引
                    preds_before = [subvocab_indices_list.index(i) for i in preds_before]
                    preds_after = [subvocab_indices_list.index(i) for i in preds_after]
                    

                    labels = [subvocab_indices_list.index(i) for i in labels]

                    img_save_path = os.path.join(img_save_dir[i], relation+".png")
                    self.create_dir(img_save_path)

                    self.generate_image(preds_before,preds_after,labels,model_bias=bias_tensor, save_path=img_save_path)


                    diff = round(acc_debias - acc_origin,5)
                    
                    print("{} 原始精度{} debias精度{}".format(
                        dataset_names[i],
                        round(acc_origin*100,2),
                        round(acc_debias*100,2)))
                    total_diff[i].append(diff)
                    f.write(f"{relation},{diff},{acc_origin},{acc_debias}\n")
                    f.flush()
                    
                    postfox = {
                        "run nums": index,
                        "lama_improve": 0 if ctrl_code[0]==0 or len(total_diff[0])==0 else round(sum(total_diff[0])/len(total_diff[0]), 3) * 100,
                        "uni_improve": 0 if ctrl_code[1]==0 or len(total_diff[1])==0 else round(sum(total_diff[1])/len(total_diff[1]), 3) * 100,
                        "lama_whu_improve": 0 if ctrl_code[2]==0 or len(total_diff[2])==0 else round(sum(total_diff[2])/len(total_diff[2]), 3) * 100,
                        }

                    pbar.set_postfix(postfox)

                pbar.update(1)
            for f in files:
                if f != None:
                    f.close()
            
            KL_save_f.close()

            # 处理输出
            for i,dataset in enumerate(output_save.keys()):
                if ctrl_code[i]==0:
                    continue
                avg_p = np.mean(output_save[dataset]["P"])
                avg_p_d = np.mean(output_save[dataset]["P_d"])
                avg_KL = np.mean(output_save[dataset]["KL"])
                avg_KL_d = np.mean(output_save[dataset]["KL_d"])
                self.add_output_item(self.model_name, dataset,prompt=continue_prompt+'_'+str(num_tokens),result=[avg_p,avg_p_d,avg_KL,avg_KL_d])
            
            self.save_output(os.path.join(root_dir, save_dir+"/result.json"))
            temp = copy.copy(self.output_result)
            output_results.append(temp)



        # 将多次实验的result融合起来
        self.clear_output()
        # 恢复之前的output
        self.output_result = self_output_result_bk

        for i,dataset in enumerate(output_save.keys()):
            if ctrl_code[i]==0:
                continue
            P = [result[self.model_name][dataset][continue_prompt+'_'+str(num_tokens)]["P"] for result in output_results]
            P_d = [result[self.model_name][dataset][continue_prompt+'_'+str(num_tokens)]["P_d"] for result in output_results]
            KL = [result[self.model_name][dataset][continue_prompt+'_'+str(num_tokens)]["KL"] for result in output_results]
            KL_d = [result[self.model_name][dataset][continue_prompt+'_'+str(num_tokens)]["KL_d"] for result in output_results]
            avg_p = np.mean(P)
            avg_p_d = np.mean(P_d)
            avg_KL = np.mean(KL)
            avg_KL_d = np.mean(KL_d)
            self.add_output_item(self.model_name, dataset,prompt=continue_prompt+'_'+str(num_tokens),result=[avg_p,avg_p_d,avg_KL,avg_KL_d])

        torch.save(preds_save,preds_save_path)

        if embeddings_renormalize==True:
            output_embedding_backup = output_embedding_backup.to(self.plm.device)
            self.plm.set_output_embeddings(output_embedding_backup)
            self.plm.bert.embeddings.word_embeddings.weight = output_embedding_backup.weight

        pbar.close()


    def wrap_input_examples(self, raw_dataset:list, tokenizer:PreTrainedTokenizer, autoprompt_postfix=""):
        # autoprompt_postfix: 针对autoprompt, 部分模板是在[x]后面加了一些字符。对于其他手工模板没有意义
        wrapped = []
        for data in raw_dataset:
            # 加入空格，使得roberta生效
            obj_token = tokenizer.tokenize(' ' + data["obj_label"])[0]
            input_example = InputExample(text_a=data["sub_label"]+autoprompt_postfix,label=tokenizer.convert_tokens_to_ids(obj_token))
            wrapped.append(input_example)

        return wrapped


    def compute_max_pad(self, dataset, WrapperClass, tokenizer, OpenPromptTemplate:Template):
        max_tokens_len = 0
        wrapped_tokenizer = WrapperClass(tokenizer=tokenizer,max_seq_length=100)
        splits = dataset.keys()
        if len(splits) == 0:
            print("error, dataset is empty ")
            exit(-1)
        else:
            for split in splits:
                for data in dataset[split]:
                    wrapped_example = OpenPromptTemplate.wrap_one_example(data)
                    token_list = wrapped_tokenizer.tokenize_one_example(wrapped_example,teacher_forcing=False)
                    current_len = sum(token_list["attention_mask"])
                    if current_len > max_tokens_len:
                        max_tokens_len =current_len
        
        
        assert max_tokens_len < 100
        
        return max_tokens_len


    def continue_prompt(self, relation, embedding_save_dir, continue_prompt="prefix", num_tokens=5, vocab_subset_filter=True, vocab_subset="answer_type_tokens", random_init="none", evaluate_mode=False):
        """
        对单个relation，训练出一个连续模板
        输出(model_bias, original_output, debiased_output)
        可以选择的连续模板包括： prefix prompt
        embedding_save_dir代表保存训练好的continue prompt的路径
        """

        # 获取预训练模型
        plm, tokenizer, model_config, WrapperClass = self.plm,self.tokenizer,self.model_config,self.WrapperClass
        model = plm

        # 是否要对模型的weight随机初始化
        if random_init=="all":
            plm = AutoModelForMaskedLM.from_config(model_config)

        # 构造数据集        
        dataset = {}
        lama_data_path = os.path.join(self.lama_data_dir,f"{relation}.jsonl")
        uni_data_path = os.path.join(self.wiki_uni_data_dir,f"{relation}.json")
        lama_whu_data_path = os.path.join(self.lama_whu_data_dir,f"{relation}.jsonl")
        data_paths = [lama_data_path,uni_data_path,lama_whu_data_path]

        autoprompt_train_data_path = os.path.join(self.auto_prompt_dir,relation,"train.jsonl")
        autoprompt_valid_data_path = os.path.join(self.auto_prompt_dir,relation, "dev.jsonl")

        raw_train_data, resample_weight = self.load_data(autoprompt_train_data_path,return_resample_weights=True)
        raw_valid_data, _ = self.load_data(autoprompt_valid_data_path)
        raw_test_data = []

        for data_path in data_paths:
            data,_ = self.load_data(data_path, vocab_subset=self.lama_vocab_subset,)
            raw_test_data.append(data)

        dataset["train"] = self.wrap_input_examples(raw_train_data, tokenizer)
        dataset["valid"] = self.wrap_input_examples(raw_valid_data, tokenizer)

        dataset["test"] = {}
        dataset["test"]["lama"] =  self.wrap_input_examples(raw_test_data[0], tokenizer)
        dataset["test"]["uni"] = self.wrap_input_examples(raw_test_data[1], tokenizer)
        dataset["test"]["lama_whu"] = self.wrap_input_examples(raw_test_data[2], tokenizer)
        
        sub_labels = {"lama":{},"uni":{},"lama_whu":{}}
        sub_labels["lama"] = [d["sub_label"] for d in raw_test_data[0]]
        sub_labels["uni"] = [d["sub_label"] for d in raw_test_data[1]]
        sub_labels["lama_whu"] = [d["sub_label"] for d in raw_test_data[2]]
        
        if continue_prompt=="prefix":
            current_template = '{"placeholder":"text_a"} '
            current_template += '{"mask"} .'

            prompt_template = SoftTemplate(
                model=plm, tokenizer=tokenizer, text=current_template,num_tokens=num_tokens)
            # 使用optiprompt中参考的初始化方式，将embedding初始化成正态分布
            prompt_template.soft_embeds.data.normal_(mean=0.0, std=model_config.initializer_range)
        elif continue_prompt=="optiprompt":
            current_template = '{"placeholder":"text_a"} '
            current_template += '{"soft":None,"duplicate":' + str(num_tokens) + '}'
            current_template += '{"mask"} .'
            prompt_template = MixedTemplate(
                model=plm, tokenizer=tokenizer, text=current_template)
            # FIXME 有点问题，之后再实现吧
            prompt_template.soft_embedding.weight.data.normal_(mean=0.0, std=model_config.initializer_range)

        # 加载数据集
        # 计算lama需要的最大pad数量
        
        max_tokens_len = 64
        
        # FIXME: 这里加了num_tokens，精度就没问题了，看起来应该是compute_max_pad里面有bug
        print(f"数据中最大token长度为{max_tokens_len}")

        sampler = WeightedRandomSampler(resample_weight, len(resample_weight))
        # train_dataloader = PromptDataLoader(dataset=dataset["train"], template=prompt_template, tokenizer=tokenizer,
        #         tokenizer_wrapper_class=WrapperClass, max_seq_length=max_tokens_len+5,
        #         batch_size=16,custom_sampler=sampler)
        train_dataloader = PromptDataLoader(dataset=dataset["train"], template=prompt_template, tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass, max_seq_length=max_tokens_len,
                batch_size=16)
        
        valid_dataloader = PromptDataLoader(dataset=dataset["valid"], template=prompt_template, tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass, max_seq_length=max_tokens_len,
                batch_size=16)
        test_dataloader = {}
        for dataset_name in ["lama","uni","lama_whu"]:
           test_dataloader[dataset_name] = PromptDataLoader(dataset=dataset["test"][dataset_name], template=prompt_template, tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass, max_seq_length=max_tokens_len,
                batch_size=16)
    
        if vocab_subset_filter:
            if vocab_subset=="common_vocab":
                subset_indices = self.get_common_vocab_indices(self.tokenizer)
            elif vocab_subset=="answer_type_tokens":
                subset_indices = self.get_answer_entity_indices(relation, tokenizer)
            else:
                raise ValueError(f"参数值{vocab_subset}不符合要求")
        else:       
            subset_indices = None
    
        # 准备prompt Model
        promptModel = PromptModel(
            template = prompt_template,
            plm = plm,
            freeze_plm=True
            
        )
        promptModel.cuda()
        best_ckpt = os.path.join(embedding_save_dir, f"weight_save/{relation}/model_full-prompt_random_init.ckpt")
        if evaluate_mode==False:
            # 开始训练
            loss_func = torch.nn.CrossEntropyLoss()
            no_decay = ['bias', 'LayerNorm.weight']

            optimizer_grouped_parameters = [
                {'params': [p for n,p in promptModel.template.named_parameters() if "raw_embedding" not in n]}
            ]

            optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
            t_total = len(dataset["train"]) * self.num_epochs
            scheduler = get_cosine_schedule_with_warmup(optimizer, int(t_total * self.warmup_proportion), t_total)

            time_start = time()
            best_acc = 0
            best_epoch = -1  
            self.create_dir(best_ckpt)
            for epoch in range(self.num_epochs):
                promptModel.train()
                tot_loss = 0
                for step, inputs in enumerate(train_dataloader):
                    inputs = inputs.cuda()
                    logits = promptModel(inputs).logits
                    mask_logits = logits[torch.where(inputs['loss_ids']>0)]
                    loss = loss_func(mask_logits, inputs["label"])
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    tot_loss += loss.item()
                    # print(tot_loss/(step+1))
                # validate it
            
                acc,_ =  self.evaluate(promptModel, valid_dataloader,vocab_subset_indices=subset_indices)
                if acc > best_acc:
                    # 保存最好的epoch
                    best_epoch = epoch
                    best_acc = acc
                    if continue_prompt=="prefix":
                        soft_embeds = promptModel.template.state_dict()["soft_embeds"].clone()
                    elif continue_prompt=='optiprompt':
                        soft_embeds = promptModel.template.state_dict()["soft_embedding.weight"].clone()
                    torch.save(soft_embeds,best_ckpt)
                    

            print("best epoch {} valid precision: {}".format(best_epoch,best_acc,))

        model_bias_output, bias_logits, bias_vector = self.get_continue_prompt_bias(promptModel,continue_prompt,num_tokens,best_ckpt,prompt_template)

        results  = {}
        for dataset in ["lama","uni","lama_whu"]:
            debias_output = self.evaluate(promptModel, test_dataloader[dataset], bias_logits=bias_logits, bias_vector=bias_vector, scaling_debiased_logit=True,vocab_subset_indices=subset_indices,soft_tempalte_ckpt=best_ckpt,continue_prompt=continue_prompt,num_tokens=num_tokens)
            origin_output = self.evaluate(promptModel, test_dataloader[dataset], vocab_subset_indices=subset_indices, soft_tempalte_ckpt=best_ckpt,continue_prompt=continue_prompt,num_tokens=num_tokens)
            acc_, (probs_, preds_, labels_) = debias_output
            debias_output = acc_, (probs_, sub_labels[dataset],preds_, labels_)
            acc_, (probs_, preds_, labels_) = origin_output
            origin_output = acc_, (probs_, sub_labels[dataset],preds_, labels_)
            results[dataset] = (debias_output,origin_output)

        return model_bias_output, results


    def get_continue_prompt_bias(self,promptModel, continue_prompt, num_tokens, ckpt_path, prompt_template):
        # 加载best epoch模型，得到model bias logits和normalize后的bias_tensor
         # 统一框架
        # 1. transformer block
        # 2. transform layer
        # 3. decoder
        model = promptModel.plm
        if isinstance(model, BertForMaskedLM):
            transformer_blocks = model.bert
            tranform_layer  = model.cls.predictions.transform
            decoder = model.cls.predictions.decoder
        elif isinstance(model, RobertaForMaskedLM):
            transformer_blocks = model.roberta
            tranform_layer = nn.Sequential(model.lm_head.dense, nn.GELU(), model.lm_head.layer_norm)
            decoder = model.lm_head.decoder
        
        soft_embeds = torch.load(ckpt_path)
        if continue_prompt=="prefix":
            promptModel.template.soft_embeds.data.copy_(soft_embeds)
        elif continue_prompt=="optiprompt":
            promptModel.template.soft_embedding.weight.data.copy_(soft_embeds)

        template_only = [{"sub_label":self.tokenizer.mask_token,"obj_label":"no"}]
        temp_dataloader = PromptDataLoader(dataset=self.wrap_input_examples(template_only,self.tokenizer), template=prompt_template, tokenizer=self.tokenizer,
            tokenizer_wrapper_class=self.WrapperClass, max_seq_length=256, batch_size=1)
        for inputs in temp_dataloader:
            inputs = inputs.cuda()
            batch = promptModel.template.process_batch(inputs)
            input_batch = {key: batch[key] for key in batch if key in promptModel.forward_keys}

            #FIXME: 不能只是用bert
            transformer_output = transformer_blocks(**input_batch)[0]
            # 后处理
            if continue_prompt=="prefix":
                # 丢弃掉输出中的soft embedding
                transformer_output = transformer_output[:,num_tokens:,:]
            mask_token_bert = transformer_output[torch.where(inputs['loss_ids']>0)]
            mask_token_features = tranform_layer(mask_token_bert)
            norm2 = torch.norm(mask_token_features, p=2)
            bias_logits = decoder(mask_token_features[0]) / norm2
        model_bias_output = bias_logits * norm2

        return model_bias_output, bias_logits, mask_token_features[0]


    def evaluate(self, promptModel:PromptModel, data_loader:PromptDataLoader,
                bias_logits=None, bias_vector=None, 
                scaling_debiased_logit=False, 
                vocab_subset_indices=None,
                soft_tempalte_ckpt=None, 
                continue_prompt="prefix",num_tokens=5,
                calibration=False, calib_logits=None):
        promptModel.eval()

        # 统一框架
        # 1. transformer block
        # 2. transform layer
        # 3. decoder
        model = promptModel.plm
        if isinstance(model, BertForMaskedLM):
            transformer_blocks = model.bert
            tranform_layer  = model.cls.predictions.transform
            decoder = model.cls.predictions.decoder
        elif isinstance(model, RobertaForMaskedLM):
            transformer_blocks = model.roberta
            tranform_layer = nn.Sequential(model.lm_head.dense, nn.GELU(), model.lm_head.layer_norm)
            decoder = model.lm_head.decoder
        
        if soft_tempalte_ckpt:
            soft_embeds = torch.load(soft_tempalte_ckpt)
            if continue_prompt=="prefix":
                promptModel.template.soft_embeds.data.copy_(soft_embeds)
            elif continue_prompt=="optiprompt":
                promptModel.template.soft_embedding.weight.data.copy_(soft_embeds)

        allpreds = []
        alllabels = []
        probs = None

        if bias_logits!=None:
            bias_logits = bias_logits.to(promptModel.plm.device)
            bias_vector = bias_vector.to(promptModel.plm.device)

        if vocab_subset_indices != None:
            vocab_subset_indices = vocab_subset_indices.to(promptModel.plm.device)
            if bias_logits!=None:
                bias_logits = bias_logits.index_select(index=vocab_subset_indices,dim=0)

        # calibration 需要对verbalizer操作一下
        if calibration:
            if isinstance(promptModel, PromptForClassification):
                promptModel.verbalizer.register_calibrate_logits(calib_logits)

        for step, inputs in enumerate(data_loader):
            inputs =  inputs.cuda()
            with torch.no_grad():
                if bias_logits!=None:
                    if isinstance(promptModel, PromptForClassification):
                        # promptModel(inputs)
                        # 以下逻辑是把openprompt的forward抄了过来
                        batch = promptModel.template.process_batch(inputs)
                        input_batch = {key: batch[key] for key in batch if key in promptModel.prompt_model.forward_keys}
                        transformer_output = transformer_blocks(**input_batch)[0]
                        # 对于softtemplate，这里有一个后处理
                        if isinstance(promptModel.template,SoftTemplate):
                            # 去掉softembedding
                            transformer_output = transformer_output[:,num_tokens:,:]
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

                         # 对分布做debias
                        mask_logits -= bias_logits
                        
                        # 从forward逻辑中中抄过来的
                        mask_logits =  promptModel.verbalizer.process_outputs(mask_logits, batch=inputs)
                       

                    elif isinstance(promptModel, PromptModel):
                        # 以下逻辑是把openprompt的forward抄了过来
                        batch = promptModel.template.process_batch(inputs)                                                                                                               
                        input_batch = {key: batch[key] for key in batch if key in promptModel.forward_keys}
                        transformer_output = transformer_blocks(**input_batch)[0]
                        # 对于softtemplate，这里有一个后处理
                        if isinstance(promptModel.template,SoftTemplate):
                            # 去掉softembedding
                            transformer_output = transformer_output[:,num_tokens:,:]
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
                        if vocab_subset_indices != None:
                            mask_logits = mask_logits.index_select(dim=1,index=vocab_subset_indices)
                        # 对分布做debias
                        mask_logits -= bias_logits

                    # 计算出来一个缩放参数，用于讲debias后的logits缩放到合理的范围
                    if scaling_debiased_logit:
                        norm_bias = torch.norm(bias_vector,p=2)
                        norm_preds = torch.norm(prediction_vectors,dim=-1)
                        normalized_pred_vecs = (prediction_vectors.T/norm_preds).T
                        normlaized_bias_vec = bias_vector/norm_bias
                        debais_vecs = normalized_pred_vecs - normlaized_bias_vec
                        norm_debias = torch.norm(debais_vecs,dim=-1)
                        batch_scaling_vec = norm_preds/norm_debias
                        mask_logits = (mask_logits.T*batch_scaling_vec).T

                else:
                    if isinstance(promptModel, PromptForClassification):
                        mask_logits = promptModel(inputs)
                        # # 执行verbalizer
                        # mask_logits =  promptModel.verbalizer.process_outputs(mask_logits, batch=inputs)

                    else:
                        logits = promptModel(inputs).logits
                        mask_logits = logits[torch.where(inputs['loss_ids']>0)]
                    if vocab_subset_indices != None:
                        mask_logits = mask_logits.index_select(dim=1,index=vocab_subset_indices)
                

                labels = inputs["label"]
                alllabels += labels.cpu().tolist()
                    

                predict_index = torch.argmax(mask_logits, dim=-1)

                if probs == None:
                    probs = mask_logits
                else:
                    probs = torch.cat((probs,mask_logits),dim=0)

                if vocab_subset_indices != None:
                    predict_index = vocab_subset_indices[predict_index]
                allpreds.extend(predict_index.cpu().tolist())
        
        acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
       

        # calibration 副作用需要手动
        if calibration:
            promptModel.verbalizer._calibrate_logits = None
            # uncalib_logits = torch.ones_like(calib_logits).to(calib_logits.device)
            # if isinstance(promptModel, PromptForClassification):
            #     promptModel.verbalizer.register_calibrate_logits(uncalib_logits)

         # 对于vocab_subset的场景，allpreds和alllabels均是原始词表中的索引
        return acc, (probs, allpreds, alllabels)

    """
    LAMA最基础的实现，效率低，但是简单易懂
    """
    def raw_manual_prompt(self, relation, PLM="bert-base-cased", prompt="LAMA"):
        tokenizer = AutoTokenizer.from_pretrained(PLM)
        model = AutoModelForMaskedLM.from_pretrained(PLM)
        if prompt=="LAMA":
            raw_template = self.lama_template[relation]
        elif prompt=="AutoPrompt":
            if isinstance(model,RobertaForMaskedLM):
                raw_template = self.AutoPrompt_tempalte_roberta[relation]
            elif isinstance(model, BertForMaskedLM):
                raw_template = self.AutoPrompt_template[relation]
            else:
                raise ValueError(f"参数值{PLM}不符合要求")
        else:
            raise ValueError(f"参数值{prompt}不符合要求")
        lama_data_path = os.path.join(self.lama_data_dir,f"{relation}.jsonl")
        # lama_data_path = os.path.join(self.auto_prompt_dir,relation,"test.jsonl")
        lama_vocab_subset = load_vocab(self.common_vocab_path)
        raw_test_data, _ = self.load_data(lama_data_path, vocab_subset=lama_vocab_subset,)
        # typed quering 候选词表
        answer_indices = self.get_answer_entity_indices(relation,tokenizer)


        correct = 0
        total = 0
        for data in raw_test_data:
            input_sentence = raw_template.replace("[X]",data["sub_label"])
            input_sentence = input_sentence.replace("[Y]",self.tokenizer.mask_token)
            model_input = tokenizer(input_sentence,return_tensors="pt")
            # model_input = {k: v.cuda() for k, v in model_input.items()}
            output = model(**model_input).logits

            # 找到[MASK]的位置
            y_mask_index = 0
            mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
            input_token_ids = model_input["input_ids"][0].tolist()
            for i,e in enumerate(reversed(input_token_ids)):
                if e == mask_id:
                    y_mask_index = -(i+1) 
                    break
            assert y_mask_index < 0
            
            mask_token = output[0][y_mask_index]
            mask_token = mask_token[answer_indices]
            index = torch.argmax(mask_token).item()
            
            if answer_indices[index] == tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' ' + data["obj_label"])[0]):
                correct += 1
            total += 1
        
        print(correct/total)
            

    def kl_divergence(self, dis_a:defaultdict, dis_b:defaultdict ):
        """
        计算两个分布之间的KL散度，以dis_a为分母
        假设dis_a和dis_b，都是概率分布
        """
        # 样本空间的所有点
        keys = dis_a.keys()
        KL = 0.0
        for key in keys:
            if dis_b[key]==0:
                continue
            KL += dis_b[key] * log(dis_b[key] / dis_a[key],2)
        
        return KL

if __name__ == '__main__':
    exp = Experiment()
    exp.init_model("bert","bert-large-cased")
    exp.prior_on_yahoo()
    

    # exp.init_model("bert","bert-large-cased")
    # exp.init_common_vocab(exp.work_dir + "/common_vocabs/common_vocab_cased.txt")
    # exp.experiment_renormal_vector_debais_for_manual_prompt(manual_prompt="LAMA")
    # exp.experiment_renormal_vector_debais_for_manual_prompt(manual_prompt="LPAQA")
    # exp.experiment_renormal_vector_debais_for_manual_prompt(manual_prompt="AutoPrompt")
    # # exp.experiment_renormal_vector_debias_for_continue_prompt(continue_prompt="optiprompt",num_tokens=5)
    # exp.save_output(output_dir+"/bert-large-cased/bert_vocab_intersection_result_new.json")
    # exp.print_output()
    # exp.clear_output()
    # # exp.clear_output()

    # exp.init_model("bert","nsadeq/InformBERT")
    # exp.init_common_vocab(self.work_dir + "/common_vocabs/common_vocab_cased.txt")
    # exp.experiment_renormal_vector_debais_for_manual_prompt(manual_prompt="LAMA")
    # exp.experiment_renormal_vector_debais_for_manual_prompt(manual_prompt="LPAQA")
    # exp.experiment_renormal_vector_debais_for_manual_prompt(manual_prompt="AutoPrompt")
    # exp.experiment_renormal_vector_debias_for_continue_prompt(continue_prompt="optiprompt",num_tokens=5)
    # exp.save_output(output_dir+"/InformBert/bert_vocab_intersection_result.json")
    # exp.clear_output()

    # exp.init_model("roberta","roberta-large")
    # exp.init_common_vocab(exp.work_dir + "/common_vocabs/common_vocab_cased_be_ro_al.txt")
    # exp.experiment_renormal_vector_debais_for_manual_prompt(manual_prompt="LAMA")
    # exp.experiment_renormal_vector_debais_for_manual_prompt(manual_prompt="LPAQA")
    # exp.experiment_renormal_vector_debais_for_manual_prompt(manual_prompt="AutoPrompt")
    # exp.save_output(output_dir+"/roberta-large/robert_vocab_intersection_result_new.json")
    # exp.print_output()
    # exp.clear_output()
    # exp.learning_rate = 6e-3
    # exp.num_epochs = 20
    # exp.experiment_renormal_vector_debias_for_continue_prompt(continue_prompt="optiprompt",num_tokens=5, repeat_times=1, )
    # # exp.save_output(output_dir+"/roberta-large/roberta_vocab_intersection_result.json")
    # exp.print_output()
    # exp.clear_output()

    # exp.init_model("roberta","roberta-large")
    # exp.init_common_vocab(self.work_dir + "/common_vocabs/common_vocab_cased_be_ro_al.txt")
    # exp.experiment_renormal_vector_debais_for_manual_prompt(manual_prompt="AutoPrompt",ctrl_code=[1,0,0])

    # exp.load_output(output_dir+"/bert-base-cased/bert_vocab_intersection_result.json")
    # exp.print_output()
    # exp.clear_output()
    # exp.load_output(output_dir+"/bert-large-cased/bert_vocab_intersection_result.json")
    # exp.print_output()
    # exp.clear_output()
    # exp.load_output(output_dir+"/roberta-large/roberta_vocab_intersection_result.json")
    # exp.print_output()
    # exp.clear_output()
