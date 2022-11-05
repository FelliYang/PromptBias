"""
使用openprompt 来重构整个代码
"""
import collections
import json
import math
import os
from re import template
from time import time
from black import out

import numpy as np
import pandas as pd
import torch
import torch.nn.parameter
from openprompt import (PromptDataLoader, PromptForClassification,
                        PromptForGeneration, PromptModel)
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompt_base import Template
from openprompt.prompts import ManualTemplate, MixedTemplate, SoftTemplate
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
from transformers import (AdamW, AutoModelForMaskedLM, AutoTokenizer,
                          BertConfig, BertModel,AutoConfig,
                          get_linear_schedule_with_warmup)
from transformers.tokenization_utils import PreTrainedTokenizer

from utils import load_json, load_jsonl, load_vocab
from torch import nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from collections import Counter,defaultdict
from math import log
from utils import set_seed

set_seed(7)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
class Experiment():
    def __init__(self) -> None:
        # settting
        self.CONTINUE_PROMPT_NUM = 5
        self.num_epochs = 10
        self.learning_rate = 3e-3
        self.warmup_proportion = 0.1
        self.prefix_token_num = 5
        
        # dataset
        self.lama_data_dir = "/home/jiao/code/prompt/OptiPrompt/data/LAMA-TREx"
        self.auto_prompt_dir = "/home/jiao/code/prompt/OptiPrompt/data/autoprompt_data"
        self.lama_whu_data_dir = "/home/jiao/code/prompt/OptiPrompt/data/LAMA-TREx_UHN"
        self.wiki_uni_data_dir = "/home/jiao/code/prompt/OptiPrompt/data/wiki_uni"
        self.common_vocab_path = "/home/jiao/code/prompt/OptiPrompt/common_vocabs/common_vocab_cased.txt"
        self.lama_template_path = "/home/jiao/code/prompt/OptiPrompt/relation_metainfo/LAMA_relations.jsonl"
        self.LPAQA_template_path = "/home/jiao/code/prompt/OptiPrompt/relation_metainfo/LPAQA_relations.jsonl"
        self.AutoPrompt_template_path = "/home/jiao/code/prompt/OptiPrompt/relation_metainfo/AutoPrompt_relations.jsonl"
        # 导入LAMA manual prompt
        data = load_jsonl(self.lama_template_path)
        self.lama_template = dict([(item["relation"], item["template"]) for item in data])
        data = load_jsonl(self.LPAQA_template_path)
        self.LPAQA_template = dict([(item["relation"], item["template"]) for item in data])
        data = load_jsonl(self.AutoPrompt_template_path)
        self.AutoPrompt_template = dict([(item["relation"], item["template"]) for item in data])

        # 导入LAMA所有关系
        data =  load_jsonl("/home/jiao/code/prompt/OptiPrompt/relation_metainfo/LAMA_relations.jsonl")
        self.relations = [ d["relation"] for d in data]
        # 过滤掉TRE之外的5个relation
        for relation in self.relations[:]:
            if not os.path.exists(os.path.join(self.lama_data_dir,f"{relation}.jsonl")):
                self.relations.remove(relation)
        
        # 提供一个公用的plm，token 
        # TODO 使用的模型完全由配置决定, 修改其他文件中重复导入的配置（在后续支持其他模型的时候做吧）
        # 获取预训练模型
        plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")
        self.plm = plm
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.WrapperClass = WrapperClass

        # 导入LAMA的公共词表
        self.lama_vocab_subset = load_vocab(self.common_vocab_path)
        self.common_vocab_indices =  self.get_common_vocab_indices(self.lama_vocab_subset,self.tokenizer)

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

        # 如果data中不包括predicate_id信息，则手动加入
        relation = os.path.basename(os.path.dirname(data_path))
        if not data[0].__contains__("predicate_id"):
            relation = os.path.basename(os.path.dirname(data_path))
            for item in data:
                item.update({"predicate_id":relation})
        
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


    def get_common_vocab_indices(self, vocab_subset, tokenizer:PreTrainedTokenizer):
        """需要对lama使用一个common vocab过滤，这样才能对其lama相关工作，但是我觉得这其实是有点虚提高了精度的"""
        index_list = []
        for word in vocab_subset:
            tokens = tokenizer.tokenize(str(word))
            if (len(tokens) == 1) and (tokens[0] != tokenizer.unk_token_id):
                index_list.append(tokenizer.convert_tokens_to_ids(tokens)[0])
            else:
                msg = "word {} from vocab_subset not in model vocabulary!".format(word)
                print(msg)
        index_list.sort()
        indices = torch.as_tensor(index_list)
        return indices


    def get_answer_entity_indices(self, relation,tokenizer:PreTrainedTokenizer, include_uni_labels=True, common_vocab_intersection=True):
        """这个函数用来得到lama对应的answer空间，例如P的answer空间是cities"""
        entity_path = "/home/jiao/code/prompt/OptiPrompt/data/entity.json"
        entity_data = load_json(entity_path)[relation]
        entity_data = entity_data["objects"]
        # 需要对entity过滤，只留下单个词的entity

        #  # 获取预训练模型
        # plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")

        def filter_out_multi_token(entity):
            out = tokenizer.tokenize(entity)
            if len(out) == 1:
                return True
            else:
                return False
        
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
            if common_vocab_intersection:
                uni_labels = list(filter(filter_common_vocab_subset,uni_labels))
            entity_data += uni_labels
        

        single_entity_data =  list(filter(filter_out_multi_token,entity_data))

        single_entity_ids = tokenizer.convert_tokens_to_ids(single_entity_data)
        # 去重复
        single_entity_ids = sorted(list(set(single_entity_ids)))
        
        return torch.tensor(single_entity_ids)


    def check_answer_entity_subset_of_common_vocab(self,):
        """测试answer entity是否是common vocab的真子集"""
        c_idx = set(self.get_common_vocab_indices(self.lama_vocab_subset, self.tokenizer).tolist())
        
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
        该函数目的是针对某个prompt作用下的model，得到一个bias的分布表
        prompt_only形式为 [MASK] was born in [MASK] . 这种形式
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


    def load_Y_AVG(self, path, resample_weight=None):
        if not os.path.exists(path):
            print(f"文件不存在{path}")
            exit(-1)
        df = pd.read_csv(path)
        if resample_weight==None:
            bias = torch.from_numpy(df.mean().values)
        else:
            raw_data = df.values
            bias = np.average(raw_data,axis=0,weights=resample_weight)
            bias = torch.from_numpy(bias)
        return bias
    
    def get_template_bias_tensor(self, model,tokenizer, template):
        """
        返回某个template在model下的bais_logits，该logits已经归一化了
        template 的形式为 [MASK] xxx xxx xxx [MASK]
        """
        model_bert = model.bert
        model_cls  = model.cls
        
        model_input = tokenizer(template,return_tensors="pt")
        model_input = {key:value.to(model.device) for key,value in model_input.items()}

        bias_bert = model_bert(**model_input)[0]
         # 找到[MASK]的位置
        y_mask_index = 0
        mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        input_token_ids = model_input["input_ids"][0].tolist()
        for i,e in enumerate(reversed(input_token_ids)):
            if e == mask_id:
                y_mask_index = -(i+1) 
                break
        assert y_mask_index < 0
        bias_bert = bias_bert[0][y_mask_index]

        # 这个transform层的意义我不清楚，但是很重要，这才算是得到了特征向量
        bias_features = model.cls.predictions.transform(bias_bert)
        # 两种不同模式下的处理
        norm2 = torch.norm(bias_features)
        bias_logits = model_cls.predictions.decoder(bias_features) / norm2

        return bias_logits    


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
        model_bert = model.bert
        model_cls = model.cls
        

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
                subset_indices = self.get_common_vocab_indices(self.lama_vocab_subset,tokenizer)
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
        raw_test_data, _ = self.load_data(test_data_path, vocab_subset=self.lama_vocab_subset,)
        dataset["test"] = self.wrap_input_examples(raw_test_data, tokenizer)
        

        # 根据prompt参数选择合适的手工模板
        templates = None
        if prompt=="LAMA":
            templates = self.lama_template
        elif prompt=="LPAQA":
            templates = self.LPAQA_template
        elif prompt=="AutoPrompt":
            templates = self.AutoPrompt_template
        else:
            raise(f"prompt参数出错: {prompt}")

        # 定义手动模板
        raw_template = templates[relation]
        # 将模板转换成openprompt
        template = raw_template.replace("[X]",'{"placeholder":"text_a"}')
        template = template.replace("[Y]", '{"mask"}')
        prompt_template = ManualTemplate(tokenizer=tokenizer, text=template)

        # 定义prompt_pnly_tempalte用于计算bias
        prompt_only_tempalte = raw_template.replace("[X]",'[MASK]').replace("[Y]", '[MASK]')
        bias_logits = self.get_template_bias_tensor(model,tokenizer,template=prompt_only_tempalte)
        

        # # 默认路径
        # if bias_tensor is None:
        #     bias_Y_path = "model_bias_data/bert/manual_prompt/{}/{}/uni_data.csv".format(
        #         relation,"subvocab_filter" if vocab_subset_filter else "full")
        #     bias_tensor = self.load_Y_AVG(bias_Y_path)



        # 加载数据集
        # 计算lama需要的最大pad数量
        max_tokens_len = self.compute_max_pad(dataset, WrapperClass, tokenizer, prompt_template)

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
            acc,(probs, allpreds, alllabels) = self.evaluate(promptModel,test_dataloader, bias_logits=bias_logits, vocab_subset_indices=subset_indices)
        else:
            acc,(probs, allpreds, alllabels) = self.evaluate(promptModel,test_dataloader, vocab_subset_indices=subset_indices)

        # print("precision: {}".format(acc))

        if embeddings_renormalize==True:
            output_embedding_backup = output_embedding_backup.to(self.plm.device)
            self.plm.set_output_embeddings(output_embedding_backup)
            self.plm.bert.embeddings.word_embeddings.weight = output_embedding_backup.weight

        return round(acc,5), (probs, allpreds, alllabels)

    def run_manual_prompt_all_relation(self,vocab_subset="common_vocab", embeddings_renormalize=False):
        save_path = f"/home/jiao/code/prompt/OptiPrompt/outputs/renormalize/lama_{vocab_subset}_renormalize.csv" if embeddings_renormalize else f"/home/jiao/code/prompt/OptiPrompt/outputs/renormalize/lama_{vocab_subset}.csv"
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
            

        save_dir = f"{manual_prompt}/debias_{vocab_subset}/"
        if embeddings_renormalize==True:
            save_dir += "renormalize_embedding"
        else:
            save_dir += "origin_embedding"
            
        dataset_names = ["lama","uni","lama_whu"]
        dataset_num = len(ctrl_code)
        files = [None]*dataset_num
        img_save_dir = [None]*dataset_num

        KL_save_path = f"/home/jiao/code/prompt/OptiPrompt/outputs/openprompt/{save_dir}/KL.csv"
        self.create_dir(KL_save_path)
        KL_save_f = open(KL_save_path,"w")
        KL_save_f.write("Relation")

        total_diff = [[],[],[]]
        
        for i in range(dataset_num):
            save_path = f"/home/jiao/code/prompt/OptiPrompt/outputs/openprompt/{save_dir}/{dataset_names[i]}_difference_debias.csv"
            self.create_dir(save_path)
            img_save_path = f"/home/jiao/code/prompt/OptiPrompt/outputs/openprompt/{save_dir}/img/{dataset_names[i]}"
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
            prompt_only_tempalte = raw_template.replace("[X]",'[MASK]').replace("[Y]", '[MASK]')
            
            subvocab_tokens_indices = self.get_answer_entity_indices(relation, self.tokenizer) \
                                                if vocab_subset == "answer_type_tokens" else \
                                    self.get_common_vocab_indices(self.lama_vocab_subset, self.tokenizer)
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
                acc_origin,(_, preds_before, labels) = self.manual_prompt(
                    relation, debias=False, 
                    vocab_subset_filter=vocab_subset_filter,
                    vocab_subset=vocab_subset,
                    test_data_path=data_paths[i],
                    prompt=manual_prompt)
                acc_debias,(_, preds_after, _) = self.manual_prompt(
                    relation, debias=True, 
                    vocab_subset_filter=vocab_subset_filter, 
                    vocab_subset=vocab_subset,
                    test_data_path=data_paths[i],
                    prompt=manual_prompt)


                # 计算出来分布，用于绘图
                num_data = len(preds_before)
                preds_before_count =  Counter(preds_before)
                preds_before_dis = defaultdict(int,
                                                 {key: preds_before_count[key]/num_data
                                                 for key in preds_before_count.keys()}
                                                 )
                preds_after_count = Counter(preds_after)
                preds_after_dis = defaultdict(int,
                                                 {key: preds_after_count[key]/num_data 
                                                    for key in preds_after_count.keys()}
                                                 )
                
                KL_before = round(self.kl_divergence(bias_dis, preds_before_dis),5)
                KL_after = round(self.kl_divergence(bias_dis, preds_after_dis),5)
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

        if embeddings_renormalize==True:
            output_embedding_backup = output_embedding_backup.to(self.plm.device)
            self.plm.set_output_embeddings(output_embedding_backup)
            self.plm.bert.embeddings.word_embeddings.weight = output_embedding_backup.weight

    def experiment_renormal_vector_debias_for_continue_prompt(self, vocab_subset_filter=True, vocab_subset="answer_type_tokens",embeddings_renormalize=False,ctrl_code=[1,1,1], continue_prompt="prefix"):
        #TODO 实现一下
        """
        ctrl_code 代表需要测试的数据集， 包扩[lama, wiki-uni, lama_whu]
        """
        root_dir = "/home/jiao/code/prompt/OptiPrompt/outputs/openprompt/continue_prompt"
        
        save_dir = continue_prompt+f"_{self.prefix_token_num}" + f"/debias_{vocab_subset}/"
        if embeddings_renormalize==True:
            save_dir += "renormalize_embedding"
        else:
            save_dir += "origin_embedding"
        
        dataset_names = ["lama","uni","lama_whu"]
        dataset_num = len(ctrl_code)
        files = [None]*dataset_num
        img_save_dir = [None]*dataset_num

        KL_save_path = os.path.join(root_dir,f"{save_dir}/KL.csv")
        self.create_dir(KL_save_path)
        KL_save_f = open(KL_save_path,"w")
        KL_save_f.write("Relation")

        total_diff = [[],[],[]]

        
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



        pbar = tqdm(total=len(self.relations)*sum(ctrl_code))

        first_dataset_index = ctrl_code.index(1)

        for relation in self.relations:
            lama_data_path = os.path.join(self.lama_data_dir,f"{relation}.jsonl")
            uni_data_path = os.path.join(self.wiki_uni_data_dir,f"{relation}.json")
            lama_whu_data_path = os.path.join(self.lama_whu_data_dir,f"{relation}.jsonl")
            data_paths = [lama_data_path,uni_data_path,lama_whu_data_path]
            for i in range(len(data_paths)):
                if ctrl_code[i]==0:
                    continue
                f = files[i]
                (model_bias,debias_res,origin_res) = self.random_prompt(relation,vocab_subset_filter=vocab_subset_filter,vocab_subset=vocab_subset,)
                subvocab_tokens_indices = self.get_answer_entity_indices(relation, self.tokenizer) \
                                                        if vocab_subset == "answer_type_tokens" else \
                                            self.common_vocab_indices
                subvocab_indices_list = subvocab_tokens_indices.tolist()
                subvocab_tokens_indices = subvocab_tokens_indices.to(model_bias.device)
                bias_tensor = model_bias.index_select(index=subvocab_tokens_indices,dim=0).cpu().detach()
                bias_probs = torch.softmax(bias_tensor,dim=-1).tolist()
                # 用于计算KL散度
                bias_dis = defaultdict(int, zip(subvocab_indices_list,bias_probs))
                # 计算出来分布，用于绘图
                acc_origin,(_, preds_before, labels) = origin_res
                acc_debias,(_, preds_after, _) = debias_res

                num_data = len(preds_before)
                preds_before_count =  Counter(preds_before)
                preds_before_dis = defaultdict(int,
                                                 {key: preds_before_count[key]/num_data
                                                 for key in preds_before_count.keys()}
                                                 )
                preds_after_count = Counter(preds_after)
                preds_after_dis = defaultdict(int,
                                                 {key: preds_after_count[key]/num_data 
                                                    for key in preds_after_count.keys()}
                                                 )
                
                KL_before = round(self.kl_divergence(bias_dis, preds_before_dis),5)
                KL_after = round(self.kl_divergence(bias_dis, preds_after_dis),5)
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

    def wrap_input_examples(self, raw_dataset:list, tokenizer:PreTrainedTokenizer):
        wrapped = []
        for data in raw_dataset:
            input_example = InputExample(text_a=data["sub_label"],label=tokenizer.convert_tokens_to_ids(data["obj_label"]))
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

     #TODO 在手工模板搞定后,重构random的代码,
    def random_prompt(self, relation, debias=False, vocab_subset_filter=True, vocab_subset="answer_type_tokens", random_init="none", ):
        # 获取预训练模型
        plm, tokenizer, model_config, WrapperClass = self.plm,self.tokenizer,self.model_config,self.WrapperClass

        if random_init=="all":
            plm = AutoModelForMaskedLM.from_config(model_config)

        # 构造数据集        
        dataset = {}
        lama_data_path = os.path.join(self.lama_data_dir,f"{relation}.jsonl")
        autoprompt_train_data_path = os.path.join(self.auto_prompt_dir,relation,"train.jsonl")
        autoprompt_valid_data_path = os.path.join(self.auto_prompt_dir,relation, "dev.jsonl")

        raw_test_data, _ = self.load_data(lama_data_path, vocab_subset=self.lama_vocab_subset,)
        raw_train_data, resample_weight = self.load_data(autoprompt_train_data_path,return_resample_weights=True)
        raw_valid_data, _ = self.load_data(autoprompt_valid_data_path)

        dataset["test"] = self.wrap_input_examples(raw_test_data, tokenizer)
        dataset["train"] = self.wrap_input_examples(raw_train_data, tokenizer)
        dataset["valid"] = self.wrap_input_examples(raw_valid_data, tokenizer)
        
        current_template = '{"placeholder":"text_a"} '
        current_template += '{"soft"} ' * 5
        current_template += '{"mask"} .'
        template = "soft"
        if template=="soft":
            prompt_template = SoftTemplate(
                model=plm, tokenizer=tokenizer, text=current_template,num_tokens=self.prefix_token_num)
            
            # 使用opti中参考的初始化方式，将embedding初始化成正态分布
            prompt_template.soft_embeds.data.normal_(mean=0.0, std=model_config.initializer_range)
        elif template=="mixed":
            prompt_template = MixedTemplate(
                model=plm, tokenizer=tokenizer, text=current_template)
            # FIXME 有点问题，之后再实现吧
            prompt_template.soft_embedding.weight.normal_(mean=0.0, std=model_config.initializer_range)

        # 加载数据集
        # 计算lama需要的最大pad数量
        
        # +20 是因为compute_max_pad不会考虑到soft token
        max_tokens_len =  self.compute_max_pad(dataset,WrapperClass,tokenizer,prompt_template)+self.prefix_token_num
        
        # FIXME: 这里加了self.prefix_token_num，精度就没问题了，看起来应该是compute_max_pad里面有bug
        print(f"数据中最大token长度为{max_tokens_len}")

        sampler = WeightedRandomSampler(resample_weight, len(resample_weight))
        # train_dataloader = PromptDataLoader(dataset=dataset["train"], template=prompt_template, tokenizer=tokenizer,
        #         tokenizer_wrapper_class=WrapperClass, max_seq_length=max_tokens_len+5,
        #         batch_size=16,custom_sampler=sampler)
        train_dataloader = PromptDataLoader(dataset=dataset["train"], template=prompt_template, tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass, max_seq_length=max_tokens_len+5,
                batch_size=16)
        
        valid_dataloader = PromptDataLoader(dataset=dataset["valid"], template=prompt_template, tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass, max_seq_length=max_tokens_len+5,
                batch_size=16)
        test_dataloader = PromptDataLoader(dataset=dataset["test"], template=prompt_template, tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass, max_seq_length=max_tokens_len+5,
                batch_size=16)
    
        if vocab_subset_filter:
            if vocab_subset=="common_vocab":
                subset_indices = self.common_vocab_indices
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

        # 开始训练
        loss_func = torch.nn.CrossEntropyLoss()
        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for n,p in promptModel.template.named_parameters() if "raw_embedding" not in n]}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        t_total = len(dataset["train"]) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, int(t_total * self.warmup_proportion), t_total)

        time_start = time()
        best_acc = 0
        best_epoch = -1
        save_name = f"/home/jiao/code/prompt/OptiPrompt/outputs/openprompt/continue_prompt/weight_save/{relation}/model_full-prompt_random_init.ckpt"
        self.create_dir(save_name)
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
                if template=="soft":
                    soft_embeds = promptModel.template.state_dict()["soft_embeds"].clone()
                elif template=='mixed':
                    soft_embeds = promptModel.template.state_dict()["soft_embedding"].clone()
                torch.save(soft_embeds,save_name)
                

        print("best epoch {} valid precision: {}".format(best_epoch,best_acc,))
    
        
        # if debias:
        # 加载best epoch模型，得到bias logits
        soft_embeds = torch.load(save_name)
        promptModel.template.soft_embeds.data.copy_(soft_embeds)
        # TODO:如何使用promptModel来得到一个bias logits?
        template_only = [{"sub_label":"[MASK]","obj_label":"no"}]
        temp_dataloader = PromptDataLoader(dataset=self.wrap_input_examples(template_only,tokenizer), template=prompt_template, tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass, max_seq_length=max_tokens_len+5,
            batch_size=1)
        for inputs in temp_dataloader:
            inputs = inputs.cuda()
            batch = promptModel.template.process_batch(inputs)
            input_batch = {key: batch[key] for key in batch if key in promptModel.forward_keys}
            output_bert = promptModel.plm.bert(**input_batch)[0]
            # 后处理
            if template=="soft":
                # 丢弃掉输出中的soft embedding
                output_bert = output_bert[:,self.prefix_token_num:,:]
            mask_token_bert = output_bert[torch.where(inputs['loss_ids']>0)]
            mask_token_features = promptModel.plm.cls.predictions.transform(mask_token_bert)
            norm2 = torch.norm(mask_token_features, p=2)
            bias_logits = promptModel.plm.cls.predictions.decoder(mask_token_features[0]) / norm2
        model_bias_output = bias_logits * norm2
        debias_output = self.evaluate(promptModel, test_dataloader,bias_logits=bias_logits,vocab_subset_indices=subset_indices)
        # else:
            # self.evaluate(promptModel, save_name=save_name,test_dataloader,vocab_subset_indices=subset_indices)
        origin_output = self.evaluate(promptModel, test_dataloader, save_name, vocab_subset_indices=subset_indices)
        # print("best test precision: {}".format(acc,))

        # print("use time {}".format(time()-time_start))

        return model_bias_output,debias_output,origin_output
        
    def evaluate(self, promptModel:PromptModel, data_loader:PromptDataLoader, soft_tempalte_ckpt=None, bias_logits=None, vocab_subset_indices=None):
        promptModel.eval()
        
        if soft_tempalte_ckpt:
            soft_embeds = torch.load(soft_tempalte_ckpt)
            promptModel.template.soft_embeds.data.copy_(soft_embeds)

        allpreds = []
        alllabels = []
        probs = None

        if vocab_subset_indices != None:
            vocab_subset_indices = vocab_subset_indices.to(promptModel.plm.device)
            if bias_logits!=None:
                bias_logits = bias_logits.to(promptModel.plm.device)
                bias_logits = bias_logits.index_select(index=vocab_subset_indices,dim=0)

        for step, inputs in enumerate(data_loader):
            inputs =  inputs.cuda()
            with torch.no_grad():
                if bias_logits!=None:
                    
                    # 以下逻辑是把openprompt的forward抄了过来
                    batch = promptModel.template.process_batch(inputs)
                    input_batch = {key: batch[key] for key in batch if key in promptModel.forward_keys}
                    output_bert = promptModel.plm.bert(**input_batch)[0]
                    # FIXME:这里好像有一个后处理
                    if isinstance(promptModel.template,SoftTemplate):
                        # 去掉softembedding
                        output_bert = output_bert[:,self.prefix_token_num:,:]
                    mask_token_bert = output_bert[torch.where(inputs['loss_ids']>0)]
                    mask_token_features = promptModel.plm.cls.predictions.transform(mask_token_bert)
                    mask_token_logits = None
                    for i in range(mask_token_features.shape[0]):
                        norm2 = torch.norm(mask_token_features[i], p=2)
                        temp = promptModel.plm.cls.predictions.decoder(mask_token_features[i]).unsqueeze(dim=0) / norm2
                        if mask_token_logits == None:
                            mask_token_logits = temp
                        else:
                            mask_token_logits = torch.cat((mask_token_logits,temp),dim=0)
                    
                    mask_logits = mask_token_logits
                else:
                    logits = promptModel(inputs).logits
                    mask_logits = logits[torch.where(inputs['loss_ids']>0)]

                if vocab_subset_indices != None:
                    mask_logits = mask_logits.index_select(dim=1,index=vocab_subset_indices)
                
                # 对分布做debias
                if bias_logits != None:
                    mask_logits -= bias_logits

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
        
        print(acc)
        # 对于vocab_subset的场景，allpreds和alllabels均是原始词表中的索引
        return acc, (probs, allpreds, alllabels)

    def raw_manual_prompt(self, relation):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        model = AutoModelForMaskedLM.from_pretrained("bert-base-cased")
        raw_template = self.lama_template[relation]
        lama_data_path = os.path.join(self.lama_data_dir,f"{relation}.jsonl")
        # lama_data_path = os.path.join(self.auto_prompt_dir,relation,"test.jsonl")
        lama_vocab_subset = load_vocab(self.common_vocab_path)
        raw_test_data, _ = self.load_data(lama_data_path, vocab_subset=lama_vocab_subset,)

        correct = 0
        total = 0
        for data in raw_test_data:
            input_sentence = raw_template.replace("[X]",data["sub_label"])
            input_sentence = input_sentence.replace("[Y]",'[MASK]')
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
            index = torch.argmax(mask_token).item()
            if index == tokenizer.convert_tokens_to_ids(data["obj_label"]):
                correct += 1
            total += 1
        
        print(correct/total)
            
    
    def experiment_renormal_raw_manual_prompt(self, relation):
        """
        测试关于是否可以把decoder的weight renormal
        """
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        model = AutoModelForMaskedLM.from_pretrained("bert-base-cased")
        config = AutoConfig.from_pretrained("bert-base-cased")

        renormal = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        renormal.weight.data = model.cls.predictions.decoder.weight.data.clone()
        renormal.weight.data = F.normalize(renormal.weight.data, p=2,dim=1)
        # renormal.bias = model.cls.predictions.decoder.bias
        model.set_output_embeddings(renormal)
        model.bert.embeddings.word_embeddings.weight = renormal.weight


        raw_template = self.lama_template[relation]
        lama_data_path = os.path.join(self.lama_data_dir,f"{relation}.jsonl")
        # lama_data_path = os.path.join(self.auto_prompt_dir,relation,"test.jsonl")
        lama_vocab_subset = load_vocab(self.common_vocab_path)
        raw_test_data, _ = self.load_data(lama_data_path, vocab_subset=lama_vocab_subset,)

        correct = 0
        total = 0
        for data in raw_test_data:
            input_sentence = raw_template.replace("[X]",data["sub_label"])
            input_sentence = input_sentence.replace("[Y]",'[MASK]')
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
            index = torch.argmax(mask_token).item()
            if index == tokenizer.convert_tokens_to_ids(data["obj_label"]):
                correct += 1
            total += 1
        
        print(correct/total)


    def experiment_renormal_vector_difference_bias(self, relation):
        """
        实验在向量空间里，对输出特征归一化后，消除数量级不同的影响做减法是否有效
        """
        # 关键的一点在分开bert和cls的操作
        tokenizer = self.tokenizer
        model = self.plm
        config = self.model_config
        model_bert = model.bert
        model_cls = model.cls
        raw_bias = model.cls.predictions.decoder.bias.data.clone()
        answer_type_indices = self.get_answer_entity_indices("P19",tokenizer=tokenizer)

        # renormal = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # renormal.weight.data = model.cls.predictions.decoder.weight.data.clone()
        # renormal.weight.data = F.normalize(renormal.weight.data, p=2,dim=1)
        # # renormal.bias = model.cls.predictions.decoder.bias
        # model.set_output_embeddings(renormal)
        # model.bert.embeddings.word_embeddings.weight = renormal.weight


        raw_template = self.lama_template[relation]
        lama_data_path = os.path.join(self.lama_data_dir,f"{relation}.jsonl")
        uni_data_path = os.path.join(self.wiki_uni_data_dir,f"{relation}.json")
        lama_vocab_subset = load_vocab(self.common_vocab_path)
        raw_test_data, _ = self.load_data(uni_data_path, vocab_subset=lama_vocab_subset,)

        correct = 0
        total = 0

        # 关键: 构建出来一个renormalized bias logits
        prompt_only_template = raw_template.replace("[X]",'[MASK]').replace("[Y]",'[MASK]')
        model_input = tokenizer(prompt_only_template,return_tensors="pt")
        bias_bert = model_bert(**model_input)[0]
         # 找到[MASK]的位置
        y_mask_index = 0
        mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        input_token_ids = model_input["input_ids"][0].tolist()
        for i,e in enumerate(reversed(input_token_ids)):
            if e == mask_id:
                y_mask_index = -(i+1) 
                break
        assert y_mask_index < 0

        bias_bert = bias_bert[0][y_mask_index]

        # 这个transform层的意义我不清楚，但是很重要，这才算是得到了特征向量
        bias_features = model.cls.predictions.transform(bias_bert)
        norm2 = torch.norm(bias_features)
        model.cls.predictions.decoder.bias.data = raw_bias / norm2
        bias_features_renormal = F.normalize(bias_features,p=2,dim=-1)
        bias_logits = model_cls.predictions.decoder(bias_features_renormal)           


        f = open("debias/P19_bias.csv","w")
        f.write("label,pred\n")     

        for data in raw_test_data:
            input_sentence = raw_template.replace("[X]",data["sub_label"])
            input_sentence = input_sentence.replace("[Y]",'[MASK]')
            model_input = tokenizer(input_sentence,return_tensors="pt")
            # model_input = {k: v.cuda() for k, v in model_input.items()}

            # 关键: 归一化
            output_bert = model_bert(**model_input)[0]
            # 找到[MASK]的位置
            y_mask_index = 0
            mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
            input_token_ids = model_input["input_ids"][0].tolist()
            for i,e in enumerate(reversed(input_token_ids)):
                if e == mask_id:
                    y_mask_index = -(i+1) 
                    break
            assert y_mask_index < 0
            mask_token_bert = output_bert[0][y_mask_index]
            mask_token_features = model.cls.predictions.transform(mask_token_bert)
            # mask_token_origin = model_cls(mask_token_bert)\

            norm2 = torch.norm(mask_token_features, p=2)
            model.cls.predictions.decoder.bias.data = raw_bias / norm2
            mask_token_features_renormal = F.normalize(mask_token_features,p=2,dim=-1)
            mask_token_logits = model_cls.predictions.decoder(mask_token_features_renormal)

            # 关键: debias
            mask_token_logits -= bias_logits
            
            # 加上过滤
            filtered_mask_token_logits = mask_token_logits[answer_type_indices]

    
            answer_type_index = torch.argmax(filtered_mask_token_logits).item()
            pred = answer_type_index
            vocab_label = tokenizer.convert_tokens_to_ids(data["obj_label"])
            label = answer_type_indices.tolist().index(vocab_label)
            if pred == label:
                correct += 1
            total += 1

            # 导出来一个分布-在answer空间下的序列
            f.write(f"{label},{pred}\n")
        
        print(correct/total)
        f.close()


    def experiment_renormal_vector_difference_bias_with_renormal_embedding(self, relation):
        """
        实验在renormal后的向量空间里，对输出特征归一化后，消除数量级不同的影响做减法是否有效
        """
        # 关键的一点在分开bert和cls的操作
        tokenizer = self.tokenizer
        model = self.plm
        config = self.model_config
        model_bert = model.bert
        model_cls = model.cls
        raw_bias = model.cls.predictions.decoder.bias.data.clone()
        answer_type_indices = self.get_answer_entity_indices("P19",tokenizer=tokenizer)

        renormal = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        renormal.weight.data = model.cls.predictions.decoder.weight.data.clone()
        renormal.weight.data = F.normalize(renormal.weight.data, p=2,dim=1)
        # renormal.bias = model.cls.predictions.decoder.bias
        model.set_output_embeddings(renormal)
        model.bert.embeddings.word_embeddings.weight = renormal.weight


        raw_template = self.lama_template[relation]
        lama_data_path = os.path.join(self.lama_data_dir,f"{relation}.jsonl")
        uni_data_path = os.path.join(self.wiki_uni_data_dir,f"{relation}.json")
        lama_vocab_subset = load_vocab(self.common_vocab_path)
        raw_test_data, _ = self.load_data(uni_data_path, vocab_subset=lama_vocab_subset,)

        correct = 0
        total = 0

        # 关键: 构建出来一个renormalized bias logits
        prompt_only_template = raw_template.replace("[X]",'[MASK]').replace("[Y]",'[MASK]')
        model_input = tokenizer(prompt_only_template,return_tensors="pt")
        bias_bert = model_bert(**model_input)[0]
         # 找到[MASK]的位置
        y_mask_index = 0
        mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        input_token_ids = model_input["input_ids"][0].tolist()
        for i,e in enumerate(reversed(input_token_ids)):
            if e == mask_id:
                y_mask_index = -(i+1) 
                break
        assert y_mask_index < 0

        bias_bert = bias_bert[0][y_mask_index]

        # 这个transform层的意义我不清楚，但是很重要，这才算是得到了特征向量
        bias_features = model.cls.predictions.transform(bias_bert)
        # norm2 = torch.norm(bias_features)
        # model.cls.predictions.decoder.bias.data = raw_bias / norm2
        bias_features_renormal = F.normalize(bias_features,p=2,dim=-1)
        bias_logits = model_cls.predictions.decoder(bias_features_renormal)           


        f = open("debias/P19_bias.csv","w")
        f.write("label,pred\n")     

        for data in raw_test_data:
            input_sentence = raw_template.replace("[X]",data["sub_label"])
            input_sentence = input_sentence.replace("[Y]",'[MASK]')
            model_input = tokenizer(input_sentence,return_tensors="pt")
            # model_input = {k: v.cuda() for k, v in model_input.items()}

            # 关键: 归一化
            output_bert = model_bert(**model_input)[0]
            # 找到[MASK]的位置
            y_mask_index = 0
            mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
            input_token_ids = model_input["input_ids"][0].tolist()
            for i,e in enumerate(reversed(input_token_ids)):
                if e == mask_id:
                    y_mask_index = -(i+1) 
                    break
            assert y_mask_index < 0
            mask_token_bert = output_bert[0][y_mask_index]
            mask_token_features = model.cls.predictions.transform(mask_token_bert)
            # mask_token_origin = model_cls(mask_token_bert)\

            # norm2 = torch.norm(mask_token_features, p=2)
            # model.cls.predictions.decoder.bias.data = raw_bias / norm2
            mask_token_features_renormal = F.normalize(mask_token_features,p=2,dim=-1)
            mask_token_logits = model_cls.predictions.decoder(mask_token_features_renormal)

            # 关键: debias
            # mask_token_logits = torch.acos(mask_token_logits)
            # bias_logits = torch.acos(bias_logits)
            mask_token_logits -= bias_logits
            # mask_token_logits *= -1
            
            # 加上过滤
            filtered_mask_token_logits = mask_token_logits[answer_type_indices]

    
            answer_type_index = torch.argmax(filtered_mask_token_logits).item()
            pred = answer_type_index
            vocab_label = tokenizer.convert_tokens_to_ids(data["obj_label"])
            label = answer_type_indices.tolist().index(vocab_label)
            if pred == label:
                correct += 1
            total += 1

            # 导出来一个分布-在answer空间下的序列
            f.write(f"{label},{pred}\n")
        
        print(correct/total)
        f.close()

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

    def experiment_renormlize_accord_for_bias(self, relation):
        """该函数用来测试bias是否来与embeddings不均衡有关系"""
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        model = AutoModelForMaskedLM.from_pretrained("bert-base-cased")
        config = AutoConfig.from_pretrained("bert-base-cased")
        raw_template = self.lama_template[relation]
        prompt_only_template = raw_template.replace("[X]",'[MASK]').replace("[Y]",'[MASK]')
        answer_type_tokens = self.get_answer_entity_indices(relation,tokenizer)
        origin_bias = self.generate_model_bias(
            model, tokenizer, prompt_only=prompt_only_template, vocab_subset_indices=answer_type_tokens, return_logits=True)
        


        # 修改模型embeddings
        renormal = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        renormal.weight.data = model.cls.predictions.decoder.weight.data.clone()
        renormal.weight.data = F.normalize(renormal.weight.data, p=2,dim=1)
        # renormal.bias = model.cls.predictions.decoder.bias
        model.set_output_embeddings(renormal)
        model.bert.embeddings.word_embeddings.weight = renormal.weight

        renormal_bias = self.generate_model_bias(
            model, tokenizer, prompt_only=prompt_only_template, vocab_subset_indices=answer_type_tokens, return_logits=True)

        self.create_dir("debias/origin_bias.pt")
        torch.save(origin_bias,"debias/origin_bias.pt")
        torch.save(renormal_bias,"debias/renormal_bias.pt")
    
    """废弃函数"""
    def experiment_difference_bias(self,vocab_subset_filter=True):
        save_path = "/home/jiao/code/prompt/OptiPrompt/outputs/openprompt/debias/difference_debias.csv"
        self.create_dir(save_path)
        f = open(save_path, "w")
        f.write("relation,diff,acc_origin,acc_debias\n")
        pbar = tqdm(total=len(self.relations))
        total_diff = []
        for relation in self.relations:
            acc_origin,_ = self.manual_prompt(relation,debias=False,vocab_subset_filter=vocab_subset_filter)
            acc_debias,_ = self.manual_prompt(relation, debias=True,vocab_subset_filter=vocab_subset_filter)
            print("原始精度{} debias精度{}".format(round(acc_origin*100,2),round(acc_debias*100,2)))
            diff = round(acc_debias - acc_origin,5)

            total_diff.append(diff)
            f.write(f"{relation},{diff},{acc_origin},{acc_debias}\n")
            f.flush()
            pbar.update(1)
            pbar.set_postfix_str("avg improve {}".format(
                round(sum(total_diff)/len(total_diff), 3)
                ))
        
        pbar.close()

        f.close()

    """废弃函数，这个函数本来的目的是使用M(T(X))的probs做平均来debias但是没什么用"""
    def experiment_difference_bias_new(self, vocab_subset_filter=True, ctrl_code=[1,1]):
        """
        ctrl_code 代表需要测试的数据集， 包扩[lama, wiki-uni]
        """
        lama_save_path = "/home/jiao/code/prompt/OptiPrompt/outputs/openprompt/debias/lama_difference_debias.csv"
        uni_save_path = "/home/jiao/code/prompt/OptiPrompt/outputs/openprompt/debias/uni_difference_debias.csv"
        save_paths = [lama_save_path,uni_save_path]


        pbar = tqdm(total=len(self.relations)*sum(ctrl_code))
        total_diff = [[],[]]
        
        for relation in self.relations:
            lama_data_path = os.path.join(self.lama_data_dir,f"{relation}.jsonl")
            uni_data_path = os.path.join(self.wiki_uni_data_dir,f"{relation}.json")
            data_paths = [lama_data_path,uni_data_path]
            # 构建两个bias tensor
            uni_bias_Y_path = "model_bias_data/bert/manual_prompt/{}/{}/uni_data.csv".format(
                relation,"subvocab_filter" if vocab_subset_filter else "full")
            uni_bias_tensor = self.load_Y_AVG(uni_bias_Y_path)
           
            lama_bias_Y_path = "model_bias_data/bert/manual_prompt/{}/{}/lama_data.csv".format(
                relation,"subvocab_filter" if vocab_subset_filter else "full")
            lama_bias_tensor = self.load_Y_AVG(lama_bias_Y_path)
            bias_Y_paths = [lama_bias_Y_path, uni_bias_Y_path]

            bias_tensors = [lama_bias_tensor, uni_bias_tensor]
            # 根据label加上resample
            lama_label_path = "model_bias_data/bert/manual_prompt/{}/{}/lama_label.csv".format(
                relation,"subvocab_filter" if vocab_subset_filter else "full")
            uni_label_path = "model_bias_data/bert/manual_prompt/{}/{}/uni_label.csv".format(
                relation,"subvocab_filter" if vocab_subset_filter else "full")
            lama_label = pd.read_csv(lama_label_path)["label"].values
            uni_label = pd.read_csv(uni_label_path)["label"].values
            labels = [lama_label, uni_label]
            for i,label in enumerate(labels):
                label_count = collections.Counter(label)
                type_num = len(label_count.keys())
                label_num = len(label)
                resample_weight = [ 1/label_count[type] for type in label]
                # resample_weight = torch.tensor(resample_weight)
                bias_tensors[i] = self.load_Y_AVG(bias_Y_paths[i], resample_weight=resample_weight)

            for i in range(len(data_paths)):
                if ctrl_code[i]==0:
                    continue

                f = open(save_paths[i], "w")
                acc_origin,_ = self.manual_prompt(
                    relation, debias=False, 
                    vocab_subset_filter=vocab_subset_filter, 
                    test_data_path=data_paths[i])
                acc_debias,_ = self.manual_prompt(
                    relation, debias=True, 
                    vocab_subset_filter=vocab_subset_filter, 
                    test_data_path=data_paths[i],
                    bias_tensor=bias_tensors[i])
                diff = acc_debias - acc_origin
                print("{} 原始精度{} debias精度{}".format(
                    "lama" if i==0 else "uni",
                    round(acc_origin*100,2),
                    round(acc_debias*100,2)))
                total_diff[i].append(diff)
                f.write(f"{relation},{diff},{acc_origin},{acc_debias}\n")
                f.flush()
                pbar.update(1)
                
                postfox = {
                    "lama_improve": 0 if ctrl_code[0]==0 or len(total_diff[0])==0 else round(sum(total_diff[0])/len(total_diff[0]), 3) * 100,
                    "uni_improve": 0 if ctrl_code[1]==0 or len(total_diff[1])==0 else round(sum(total_diff[1])/len(total_diff[1]), 3) * 100,
                    }

                pbar.set_postfix(postfox)
        
        pbar.close()

        """废弃函数"""
    
    """废弃函数"""
    def experiment_answer_subset_bias(self, vocab_subset_filter=True, vocab_subset="answer_type_tokens", ctrl_code=[1,1]):
        """
        测试answer子空间上能否debias成功
        """
        lama_save_path = "/home/jiao/code/prompt/OptiPrompt/outputs/openprompt/debias/lama_answer_entities_debias.csv"
        uni_save_path = "/home/jiao/code/prompt/OptiPrompt/outputs/openprompt/debias/uni_answer_entities_debias.csv"
        save_paths = [lama_save_path,uni_save_path]
        # self.relations = ["P19"]
        pbar = tqdm(total=len(self.relations)*sum(ctrl_code))
        total_diff = [[],[]]

        for relation in self.relations:
            lama_data_path = os.path.join(self.lama_data_dir,f"{relation}.jsonl")
            uni_data_path = os.path.join(self.wiki_uni_data_dir,f"{relation}.json")
            data_paths = [lama_data_path,uni_data_path]
            # 构建两个bias tensor, 在本实验中，bias均选取的是M(T)因此，是一样的
            bias_tensor_path = f"/home/jiao/code/prompt/OptiPrompt/model_bias_data/bert/manual_prompt/{relation}/full/model_bias.csv"
            full_bias_tensor = torch.from_numpy(pd.read_csv(bias_tensor_path).values).squeeze()
            answer_entity_indices = self.get_answer_entity_indices(relation, self.tokenizer)

            # 做一个softmax
            bias_tensor = torch.softmax(full_bias_tensor[answer_entity_indices],dim=0)
            bias_tensors = [bias_tensor, bias_tensor]

            for i in range(len(data_paths)):
                if ctrl_code[i]==0:
                    continue

                f = open(save_paths[i], "w")
                acc_origin,_ = self.manual_prompt(
                    relation, debias=False, 
                    vocab_subset_filter=vocab_subset_filter, 
                    vocab_subset=vocab_subset,
                    test_data_path=data_paths[i])
                acc_debias,_ = self.manual_prompt(
                    relation, debias=True, 
                    vocab_subset_filter=vocab_subset_filter, 
                    vocab_subset=vocab_subset,
                    test_data_path=data_paths[i],
                    bias_tensor=bias_tensors[i])
                diff = acc_debias - acc_origin
                print("{} 原始精度{} debias精度{}".format(
                    "lama" if i==0 else "uni",
                    round(acc_origin*100,2),
                    round(acc_debias*100,2)))
                total_diff[i].append(diff)
                f.write(f"{relation},{diff},{acc_origin},{acc_debias}\n")
                f.flush()
                pbar.update(1)
                
                postfox = {
                    "lama_improve": 0 if ctrl_code[0]==0 or len(total_diff[0])==0 else round(sum(total_diff[0])/len(total_diff[0]), 3) * 100,
                    "uni_improve": 0 if ctrl_code[1]==0 or len(total_diff[1])==0 else round(sum(total_diff[1])/len(total_diff[1]), 3) * 100,
                    }

                pbar.set_postfix(postfox)
        pbar.close()
    
    """废弃函数"""
    def generate_bias_dataset(self, relation, vocab_subset_filter, vocab_subset="common_vocab", round_num=-1, sparse=False):
        """
        生成特定relation分析prompt需要的数据
        目前仅支持手工模板
        round_num 表示是否开启舍入，用来节省空间 
        sparseb表示是否开启稀疏存储，用来节省空间
        save_logits_tensor表示不再使用json文本格式来稀疏存储了，而是转为使用

        #TODO:支持连续模板
        """
        if sparse and round_num <=0:
            print("没有开启舍入，稀疏存储不起作用")
            exit(-1)

        # 获取预训练模型
        plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")
        # 根据当前模型，构造出来lama subset indices
        if vocab_subset_filter:
            if vocab_subset=="common_vocab":
                subset_indices = self.get_common_vocab_indices(self.lama_vocab_subset,tokenizer)
            elif vocab_subset == "answer_type_tokens":
                subset_indices = self.get_answer_entity_indices(relation,tokenizer)
            else:
                raise ValueError(f"参数值{vocab_subset}不符合要求")
            subset_indices_list = subset_indices.tolist()

        else:
            subset_indices = None
            subset_indices_list = None
        
        raw_template = self.lama_template[relation]
        
        prompt_only_template = raw_template.replace("[X]","[MASK]").replace("[Y]","[MASK]")
        bias = self.generate_model_bias(plm,tokenizer,
            prompt_only_template,
            vocab_subset_indices =subset_indices)
        if vocab_subset_filter:
            words = tokenizer.convert_ids_to_tokens(subset_indices)
        
        subvocab_name = "common_vocab" if vocab_subset=="common_vocab" else "answer_type_tokens"
        
        model_bias_save_path = "model_bias_data/bert/manual_prompt/{}/{}/model_bias.jsonl".format(
            relation,subvocab_name if vocab_subset_filter else "full")

        model_bias_csv_save_path = "model_bias_data/bert/manual_prompt/{}/{}/model_bias.csv".format(
            relation,subvocab_name if vocab_subset_filter else "full")

        self.create_dir(model_bias_save_path)
        self.create_dir(model_bias_csv_save_path)

        f = open(model_bias_save_path,"w")
        f_csv = open(model_bias_csv_save_path, "w")

        csv_title = [f"{i}" for i in range(bias.shape[0])]
        f_csv.write(",".join(csv_title))
        f_csv.write("\n")

        for i in range(bias.shape[0]):
            item = {"index": i,
                    "ori_index": subset_indices[i].item() if vocab_subset_filter else i,
                    "word": words[i] if vocab_subset_filter else tokenizer.convert_ids_to_tokens(i),
                    "prob": bias[i].item()}
            item_str = json.dumps(item)
            f.write(item_str+'\n')

            if i>0:
                f_csv.write(",{}".format(bias[i].item()))
            else:
                f_csv.write("{}".format(bias[i].item()))
        
        f_csv.write("\n")


        f.close()
        f_csv.close()


        
        # 将模板转换成openprompt
        template = raw_template.replace("[X]",'{"placeholder":"text_a"}')
        template = template.replace("[Y]", '{"mask"}')
        prompt_template = ManualTemplate(tokenizer=tokenizer, text=template)
        # 准备prompt Model
        promptModel = PromptModel(
            template = prompt_template,
            plm = plm,
            freeze_plm=True
        )
        promptModel.cuda()
        

        # 加载lama数据集
        lama_data_path = os.path.join(self.lama_data_dir,f"{relation}.jsonl")
        raw_lama_data, _ = self.load_data(lama_data_path, vocab_subset=self.lama_vocab_subset,)
        lama_dataset = self.wrap_input_examples(raw_lama_data, tokenizer)
        
        max_tokens_len = self.compute_max_pad({"test":lama_dataset}, WrapperClass, tokenizer, prompt_template)
        lama_dataloader = PromptDataLoader(dataset=lama_dataset, template=prompt_template, tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass, max_seq_length=max_tokens_len, 
            batch_size=16,shuffle=False)
        # 加载wiki-wni数据集
        wiki_uni_data_path = os.path.join(self.wiki_uni_data_dir,f"{relation}.json")
        raw_uni_data, _ = self.load_data(wiki_uni_data_path, vocab_subset=self.lama_vocab_subset,)
        uni_dataset = self.wrap_input_examples(raw_uni_data, tokenizer)
        
        max_tokens_len = self.compute_max_pad({"test":uni_dataset}, WrapperClass, tokenizer, prompt_template)
        uni_dataloader = PromptDataLoader(dataset=uni_dataset, template=prompt_template, tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass, max_seq_length=max_tokens_len, 
            batch_size=16,shuffle=False)
        
        lama_data_save_path = "model_bias_data/bert/manual_prompt/{}/{}/lama_data.jsonl".format(
            relation,subvocab_name if vocab_subset_filter else "full")
        uni_data_save_path = "model_bias_data/bert/manual_prompt/{}/{}/uni_data.jsonl".format(
            relation,subvocab_name if vocab_subset_filter else "full")
        lama_data_csv_save_path = "model_bias_data/bert/manual_prompt/{}/{}/lama_data.csv".format(
            relation,subvocab_name if vocab_subset_filter else "full")
        uni_data_csv_save_path = "model_bias_data/bert/manual_prompt/{}/{}/uni_data.csv".format(
            relation,subvocab_name if vocab_subset_filter else "full")
        lama_label_csv_path = "model_bias_data/bert/manual_prompt/{}/{}/lama_label.csv".format(
            relation,subvocab_name if vocab_subset_filter else "full")
        uni_label_csv_path = "model_bias_data/bert/manual_prompt/{}/{}/uni_label.csv".format(
            relation,subvocab_name if vocab_subset_filter else "full")
        


        dataloaders = [lama_dataloader, uni_dataloader]
        save_paths = [lama_data_save_path,uni_data_save_path]
        save_csv_paths = [lama_data_csv_save_path, uni_data_csv_save_path]
        raw_datasets  = [raw_lama_data, raw_uni_data]
        label_save_paths = [lama_label_csv_path, uni_label_csv_path]

        for dataloader, save_path, save_csv_path,label_csv_path,raw_dataset in zip(dataloaders,save_paths,save_csv_paths,label_save_paths, raw_datasets):
            self.create_dir(save_path)
            f = open(save_path,"w")
            f_csv = open(save_csv_path, "w")
            label_csv = open(label_csv_path,"w")

            if vocab_subset_filter:
                acc1,(probs, preds, labels ) = self.evaluate(promptModel,dataloader,vocab_subset_indices=subset_indices)
                
            else:
                acc1,(probs, preds, labels ) = self.evaluate(promptModel,dataloader)
            
            
            # 数据csv title
            probs_num = probs.shape[-1]
            csv_title = [f"{i}" for i in range(probs_num)]
            f_csv.write(",".join(csv_title))
            f_csv.write("\n")

            # label csv title
            label_csv.write("label,pred\n")
            

            for i in range(len(preds)):
                item = {}
                item["sub_label"] = raw_dataset[i]["sub_label"]
                item["obj_label"] = raw_dataset[i]["obj_label"]
                # pred和label为原始词表，需要转成vocab_subset的index
                if vocab_subset_filter:
                    item["pred"] = subset_indices_list.index(preds[i])
                    item["label"] = subset_indices_list.index(labels[i])
                else:
                    item["pred"] = preds[i]
                    item["label"] = labels[i]
                item["probs"] = probs[i].cpu().tolist()

                # 写入label csv中
                label_csv.write(f'{item["label"]},{item["pred"]}\n')

                if round_num>0:
                    round_res = torch.floor(probs[i] * math.pow(10,round_num)) / math.pow(10,round_num)
                    # round 可能会造成某些概率非常近似的两个logit无法区分，如果正好是pred这个logit那么就会带来错误
                    # 因此这里对这种特殊情况做一个处理
                    pred_index = item["pred"]
                    round_res[pred_index] = item["probs"][pred_index]
                    
                    item["probs"] = round_res.tolist()
                    values = [round(i, round_num) for i in item["probs"]]
                    values[pred_index] = item["probs"][pred_index]
                    item["probs"] = values
                    csv_item = [str(i) for i in item["probs"]]
                    csv_content = ",".join(csv_item)
                    f_csv.write(csv_content)
                    f_csv.write("\n")

                    item["probs"] = torch.tensor(values).cuda()

                if sparse:
                    non_zero = torch.nonzero(item["probs"]).squeeze(dim=1)
                    prob_output = {"len":len(non_zero),
                                    "nonzero_index":non_zero.tolist(),
                                    }
                    values = item["probs"][non_zero].tolist()
                    prob_output["values"] = values
                    item["probs"] = prob_output
                else:
                    item["probs"] = item["probs"].tolist()
                
                f.write(json.dumps(item))
                f.write('\n')
            
            f.close()
            f_csv.close()

    """废弃函数"""
    def check_bias_dataset(self, model_bias, dataset):
        """该函数用来校验生成的bias数据集是否存在问题"""
        model_bias = load_jsonl(model_bias)
        dataset = load_jsonl(dataset)

        # 构建vocab_subset的索引字典
        subset_index = dict([(item["index"],item["word"]) for item in model_bias])

        correct = 0
        for data in dataset:
            # 检查1 probs最大值索引是否是预测值
            pred = data["pred"]
            prob_max_index = data["probs"]["values"].index(max(data["probs"]["values"]))
            prob_max = data["probs"]["nonzero_index"][prob_max_index]
            assert pred == prob_max
            if pred == data["label"]:
                correct += 1
            # 检查2 检查当前数据的Lbael对应的字符串是否是obj_label
            assert data["obj_label"] == subset_index[data["label"]]

        print("检查通过-数据集的精度为{}".format(correct/len(dataset)))

    """废弃函数"""
    def generate_bias_datasets(self, round_num=-1, sparse=False, ):
        #TODO 增加不同的子词过滤
        self.relations = ["P19"]
        pbar = tqdm(total=len(self.relations)*2)
        for relation in self.relations:
            for vocab_subset_filter in [True, False]:
                self.generate_bias_dataset(relation, vocab_subset_filter,round_num,sparse)
                model_bias_save_path = "model_bias_data/bert/manual_prompt/{}/{}/model_bias.jsonl".format(
                    relation, "subvocab_filter" if vocab_subset_filter else "full")
                lama_data_save_path = "model_bias_data/bert/manual_prompt/{}/{}/lama_data.jsonl".format(
                    relation, "subvocab_filter" if vocab_subset_filter else "full")
                uni_data_save_path = "model_bias_data/bert/manual_prompt/{}/{}/uni_data.jsonl".format(
                    relation, "subvocab_filter" if vocab_subset_filter else "full")
                self.check_bias_dataset(model_bias_save_path,lama_data_save_path)
                self.check_bias_dataset(model_bias_save_path,uni_data_save_path)
                pbar.update(1)
        pbar.close()



exp = Experiment()
# exp.random_prompt("P19",debias=False)
# _,(acc,_),_ = exp.random_prompt("P19",debias=True,vocab_subset_filter="answer_type_tokens")



# exp.experiment_answer_subset_bias()
# exp.experiment_renormal_raw_manual_prompt("P19")
# exp.run_manual_prompt_all_relation(vocab_subset="answer_type_tokens",embeddings_renormalize=True)
# exp.run_manual_prompt_all_relation(vocab_subset="answer_type_tokens",embeddings_renormalize=False)
# exp.run_manual_prompt_all_relation(vocab_subset="common_vocab", embeddings_renormalize=True)

# exp.experiment_renormal_vector_difference_bias(relation='P19')
# exp.experiment_renormal_vector_difference_bias_with_renormal_embedding(relation='P19')

# exp.manual_prompt("P19",vocab_subset="answer_type_tokens",debias=True)

# exp.relations = ["P19","P20"]
exp.experiment_renormal_vector_debias_for_continue_prompt(ctrl_code=[1,0,0])
# exp.experiment_renormal_vector_debais_for_manual_prompt(ctrl_code=[1,1,1], embeddings_renormalize=False)
# exp.experiment_renormal_vector_debais(ctrl_code=[1,0,0],vocab_subset="common_vocab", embeddings_renormalize=False, prompt="AutoPrompt")
# exp.experiment_renormal_vector_debais(ctrl_code=[1,1,1], embeddings_renormalize=True, prompt="LPAQA")
# exp.experiment_renormal_vector_debais(ctrl_code=[1,1,1], embeddings_renormalize=True, prompt="AutoPrompt")
# exp.experiment_renormal_vector_debais(ctrl_code=[1,1,1],embeddings_renormalize=True)
# exp.experiment_renormal_vector_debais(ctrl_code=[0,1,1])

# exp.check_answer_entity_subset_of_common_vocab()

# exp.experiment_renormlize_accord_for_bias("P19")
# exp.run_manual_prompt_all_relation(vocab_subset="common_vocab", embeddings_renormalize=False)
# t = exp.get_answer_entity_indices("P19",exp.tokenizer)
# print(t.shape)

# exp.generate_bias_dataset("P19",vocab_subset_filter=True,vocab_subset="answer_type_tokens",round_num=5)
# exp.experiment_renormlize_accord_for_bias("P19")
# exp.raw_manual_prompt("P138")
# exp.generate_bias_datasets(round_num=5,sparse=True)
# exp.manual_prompt("P19",debias=True)
# exp.random_prompt("P19", random_init="all",)
# exp.experiment_difference_bias(vocab_subset_filter=True)
# exp.generate_bias_dataset(relation="P19",vocab_subset_filter=True, round_num=5, sparse=True)
# exp.check_bias_dataset("/home/jiao/code/prompt/OptiPrompt/model_bias_data/bert/manual_prompt/P19/subvocab_filter/model_bias.jsonl",
#         "/home/jiao/code/prompt/OptiPrompt/model_bias_data/bert/manual_prompt/P19/subvocab_filter/lama_data.jsonl")

# exp.check_bias_dataset("/home/jiao/code/prompt/OptiPrompt/model_bias_data/bert/manual_prompt/P19/subvocab_filter/model_bias.jsonl",
#         "/home/jiao/code/prompt/OptiPrompt/model_bias_data/bert/manual_prompt/P19/subvocab_filter/uni_data.jsonl")
# exp.generate_bias_dataset(relation="P19",vocab_subset_filter=False, round_num=5, sparse=True)
# exp.check_bias_dataset("/home/jiao/code/prompt/OptiPrompt/model_bias_data/bert/manual_prompt/P19/full/model_bias.jsonl",
#         "/home/jiao/code/prompt/OptiPrompt/model_bias_data/bert/manual_prompt/P19/full/lama_data.jsonl")

# exp.check_bias_dataset("/home/jiao/code/prompt/OptiPrompt/model_bias_data/bert/manual_prompt/P19/full/model_bias.jsonl",
#         "/home/jiao/code/prompt/OptiPrompt/model_bias_data/bert/manual_prompt/P19/full/uni_data.jsonl")


# exp.manual_prompt("P19",debias=True)
# acc = exp.manual_prompt("P19",vocab_subset_filter=True,debias=True)
# print("精度为{}".format(round(acc*100,2)))
# exp.experiment_difference_bias()
# exp.experiment_difference_bias_new()
# exp.get_answer_entity_indices(relation="P19")
# acc = exp.manual_prompt("P19",vocab_subset_filter=True,vocab_subset="answer_type_tokens")
# print("精度为{}".format(round(acc*100,2)))
# exp.generate_bias_datasets(sparse=True,round_num=5)
# b, _ = exp.load_data(os.path.join(exp.lama_data_dir,"P19.jsonl"))
# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# model = AutoModelForMaskedLM.from_pretrained("bert-base-cased")
# prompt_only = "[MASK] was born in [MASK] ."
# exp.generate_model_bias(model,tacc = self.evaluate(promptModel,test_dataloader)
