import collections
import json
import math
import os
from re import template
from time import time
import warnings
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
                          BertForMaskedLM, RobertaForMaskedLM,PreTrainedModel)
from transformers.tokenization_utils import PreTrainedTokenizer

from utils import load_json, load_jsonl, load_vocab, set_seed, create_dir
from torch import nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from collections import Counter,defaultdict
from math import log
from prettytable import PrettyTable
import copy

class Experiment():
    def __init__(self, seed=7, support_dataset=200) -> None:
        # settting
        self.CONTINUE_PROMPT_NUM = 5
        self.num_epochs = 10
        self.learning_rate = 3e-3
        self.warmup_proportion = 0.1
        self.seed=seed
        self.support_dataset = support_dataset
        set_seed(self.seed)

        # output save
        self.output_result = {}
        self.work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # dataset
        self.lama_data_dir = self.work_dir + "/data/LAMA-TREx"
        self.auto_prompt_dir = self.work_dir + "/data/autoprompt_data"
        self.lama_whu_data_dir = self.work_dir + "/data/LAMA-TREx_UHN"
        self.wiki_uni_data_dir = self.work_dir + "/data/wiki_uni"
        self.templama_data_dir = self.work_dir + "/data/TEMPLAMA"
        self.common_vocab_path = self.work_dir + "/common_vocabs/common_vocab_cased.txt"
        self.lama_template_path = self.work_dir + "/relation_metainfo/LAMA_relations.jsonl"
        self.LPAQA_template_path = self.work_dir + "/relation_metainfo/LPAQA_relations.jsonl"
        self.AutoPrompt_bert_template_path = self.work_dir + "/relation_metainfo/AutoPrompt_relations.jsonl"
        self.AutoPrompt_roberta_template_path = self.work_dir + "/relation_metainfo/AutoPrompt_relations_roberta.jsonl"
        
        # load different prompts (including LAMA manual, LPAQA, AutoPrompt)
        # for AutoPrompt, there are two groups of prompt, designed for Bert and Roberta respectively.
        data = load_jsonl(self.lama_template_path)
        self.lama_template = dict([(item["relation"], item["template"]) for item in data])
        data = load_jsonl(self.LPAQA_template_path)
        self.LPAQA_template = dict([(item["relation"], item["template"]) for item in data])
        data = load_jsonl(self.AutoPrompt_bert_template_path)
        self.AutoPrompt_template_bert = dict([(item["relation"], item["template"]) for item in data])
        data = load_jsonl(self.AutoPrompt_roberta_template_path)
        self.AutoPrompt_tempalte_roberta = dict([(item["relation"], item["template"]) for item in data])
        self.AutoPrompt_tempalte_roberta = {k:v.strip() for k,v in self.AutoPrompt_tempalte_roberta.items()}

        # load all 46 relations in the LAMA benchmark
        data =  load_jsonl(self.work_dir + "/relation_metainfo/LAMA_relations.jsonl")
        self.relations = [ d["relation"] for d in data]
        # filter out the 5 relations out of the T-REx split
        for relation in self.relations[:]:
            if not os.path.exists(os.path.join(self.lama_data_dir,f"{relation}.jsonl")):
                self.relations.remove(relation)

        # load the common vocab of different language models. In default setting, we use the common vocabulary constructed in
        # "Language Model as Knowledge Bases?", containing ~21 K case-sensetive tokens.
        self.set_common_vocab(self.common_vocab_path)


    def set_model(self,model_type, model_name):
        self.model_type = model_type
        self.model_name = model_name
        plm, tokenizer, model_config, WrapperClass = load_plm(model_type, model_name)
        self.plm = plm
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.WrapperClass = WrapperClass


    def set_common_vocab(self, common_vocab_path):
        self.common_vocab = load_vocab(common_vocab_path)


    def clear_output(self,):
        self.output_result = {}
    

    def save_output(self, path):
        create_dir(path)
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
            table.field_names = ["Datasets",  "Prompts","P","P^d","KL","KL^d","TT_CE","TT_CE^d","FF_E","FF_E^d"]
            model_output = self.output_result[model]
            for dataset in model_output.keys():
                table.add_row([dataset,"","","","","","","","",""])
                dataset_output = model_output[dataset]
                for prompt in dataset_output.keys():
                    prompt_output = dataset_output[prompt]
                    P,P_d,KL,KL_d = prompt_output['P'], prompt_output["P_d"], prompt_output["KL"], prompt_output["KL_d"]
                    TT_CE, TT_CE_d, FF_E, FF_E_d = prompt_output["TT_CE"], prompt_output["TT_CE_d"], prompt_output["FF_E"], prompt_output["FF_E_d"]
                    P = math.floor(P *100 * 10+0.5)/10
                    P_d = math.floor(P_d *100 * 10+0.5)/10
                    KL = math.floor(KL * 10+0.5)/10
                    KL_d = math.floor(KL_d * 10+0.5)/10
                    TT_CE= math.floor(TT_CE * 1000+0.5)/1000
                    TT_CE_d= math.floor(TT_CE_d * 1000+0.5)/1000
                    FF_E = math.floor(FF_E * 1000+0.5)/1000
                    FF_E_d= math.floor(FF_E_d * 1000+0.5)/1000
                    table.add_row(["",prompt,P,P_d,KL,KL_d, TT_CE, TT_CE_d, FF_E,FF_E_d])
            print(table)
                    

    def add_output_item(self, model, dataset, prompt, result):
        if model not in self.output_result.keys():
            self.output_result[model] = {}
        if dataset not in self.output_result[model].keys():
            self.output_result[model][dataset] = {}
        self.output_result[model][dataset][prompt] = {"P":result[0], "P_d":result[1], "KL":result[2], "KL_d":result[3], 
                                                    "TT_CE":result[4], "TT_CE_d": result[5], "FF_E": result[6], "FF_E_d": result[7]}
    

    def load_data(self, data_path, common_vocab=None, filter_repeat=True, filter_out_tokens=[], return_resample_weights=False, close_filter=False):
        """load and preprocess a dataset. 
        
        The raw dataset is assumed to be a .jsonl or .json file, with data items <sub_label, relation, obj_label>.

        Args:
            data_path (str): path of the dataset that will be loaded in.
            common_vocab (list, optional): . Defaults to None.
            filter_repeat (bool, optional): whether to filter out repeat data items. Defaults to True.
            filter_out_tokens (list, optional): Forcefully filter out data with labels present in this list.
            return_resample_weights (bool, optional): whether to return a resample weights, which can be used to avoid a extreme label distribution in training. Defaults to False.
            close_filter (bool, optional): whether to close all filters

        Returns:
            (list, None): the dataset loaded.
            (list, list): the dataset loaded as well as a resample weights.
        """
        
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

        # filter out obj_label that not in the common_vocab
        if not close_filter and common_vocab:
            current_len = len(data)
            data = list(filter(lambda x: x["obj_label"] in common_vocab, data))
            print("filter out {} items not in the vocab_subset".format(current_len - len(data)))
        
        # filter out obj_label that in the filter_out_tokens list
        def _token_to_id(x):
            # 加入空格，使得roberta生效
            obj_token = self.tokenizer.tokenize(' ' + x)[0]
            id = self.tokenizer.convert_tokens_to_ids(obj_token)
            return id

        if not close_filter and (filter_out_tokens != []):
            current_len = len(data)
            data = list(filter(lambda x: _token_to_id(x["obj_label"]) not in filter_out_tokens, data))
            print("filter out {} items belong to the top {} prompt biased token(s)".format(current_len - len(data), len(filter_out_tokens)))

        # filter out repeat data items
        if not close_filter and filter_repeat:
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

        # calcuate the resample weights according to the label ratio in the dataset
        weights = None
        if return_resample_weights:
            # count the num of each obj_label in data
            obj_count = {}
            # record type of sample in dataset
            types = []
            for e in data:
                cls = e["obj_label"]
                types.append(cls)
                # if the cls had not been added to types
                if not obj_count.__contains__(cls):
                    obj_count[cls] = 1
                else:
                    obj_count[cls] += 1
            
            weights = [1 / obj_count[types[i]] for i in range(len(data))] 

        return data, weights 
    

    def get_vocab_subset_indices(self, vocab_subset, tokenizer:PreTrainedTokenizer):
        """Map the vocab subset to the model vocab. 
        
        For each word in the vocab subset, get the corresponding index in the model vocab.

        Args:
            vocab_subset (list): a subset of the model vocabulary, for example, a common vocab or a answer entity vocab.
            tokenizer (PreTrainedTokenizer): the tokenizer of the pretrained language model.

        Returns:
            list:  A list of indices, sorted from smallest to largest.
        """
        
        
        index_list = []
        for word in vocab_subset:
            # This space is added since RoBERTa use BPE tokenizer. 
            # A prefix space is important for BPE to differ (ĠHello) from (Hello).
            tokens = tokenizer.tokenize(' ' + str(word))
            if (len(tokens) == 1) and (tokens[0] != tokenizer.unk_token_id):
                index_list.append(tokenizer.convert_tokens_to_ids(tokens)[0])
            else:
                msg = f"word {word} in the common vocab is not in the model vocabulary or its length is larger than 1! \
                This might cause unexpected errors. Please ensure this warning has been eliminated!"
                warnings.warn(msg, Warning)
        index_list.sort()
        return index_list


    def get_common_vocab_indices(self, tokenizer:PreTrainedTokenizer):
        """A simple wrap of get_vocab_subset_indices when the vocab_subset is self.common_vocab.

        Args:
            tokenizer (PreTrainedTokenizer): the tokenizer of the pretrained language model.

        Returns:
            tensor: model vocab's indices for the common vocab, sorted from smallest to largest.
        """
        index_list = self.get_vocab_subset_indices(self.common_vocab,tokenizer)
        indices = torch.as_tensor(index_list)
        return indices


    def get_answer_entity_indices(self, relation, 
                tokenizer:PreTrainedTokenizer, 
                include_multi_lama=True,
                include_lama_labels=True,
                include_uni_labels=True, 
                include_templama_labels=False,
                common_vocab_intersection=True):
        """Get the answer space vocab indices.

        When use the Typed Query, we would consturct a answer space for each relation. For example,
        the answer space of relation P19 (birthplace) should consists of cities. This func can return
        single world cities in the model vocabulary.

        Args:
            relation (str): the target relation
            tokenizer (PreTrainedTokenizer): tokenizer of the pretrained languaage model
            include_multi_lama (bool, optional): whether to consider the entity.json into the answer space. Defaults to True. 
            include_lama_labels (bool, optional): whether to consider the labels of the same `relation` in LAMA dataset into the answer space. 
                In default, we had consideres the labels of LAMA dataset. Defaults to True. 
            include_uni_labels (bool, optional): whether to consider the labels of the same `relation` in UNI dataset into the answer space. 
                In default, we had consideres the labels of UNI dataset. Defaults to True. 
            include_templama_labels (bool, optional): whether to consider the labels of the same `relation` in TEMPLAMA dataset into the answer space.
                Defaults to False.
            common_vocab_intersection (bool, optional): whether to intersect the answer space with common vocab to ensure 
                all candidate words are in the common vocabulary. Defaults to True.

        Returns:
            tensor: model vocab's indices for the answer space, sorted from smallest to largest.
        """

        # take the entity.json as the base answer space, which is constructed by 
        # "multilingual lama: investigating knowledge in multilingual pretrained language models"
        entity_data = []

        if include_multi_lama:
            entity_path = self.work_dir + "/data/entity.json"
            entity_data = load_json(entity_path)[relation]
            entity_data = entity_data["objects"]

        # add labels of the same relation in LAMA dataset to the answer space
        if include_lama_labels:
            lama_path = os.path.join(self.lama_data_dir,f"{relation}.jsonl")
            lama_data = load_jsonl(lama_path)
            lama_labels = [ item["obj_label"] for item in lama_data]
            entity_data += lama_labels

        if include_uni_labels:
            # add labels of the same relation in UNI dataset to the answer space
            uni_path = os.path.join(self.wiki_uni_data_dir,f"{relation}.json")
            uni_data = load_json(uni_path)
            uni_labels = [ item["obj_label"] for item in uni_data]
            entity_data += uni_labels
        
        if include_templama_labels:
            # add labels of the same relation in tempLAMA dataset to the answer space
            templama_path = os.path.join(self.templama_data_dir,f"{relation}.jsonl")
            templama_data = load_jsonl(templama_path)
            templama_labels = [item["answer"][0]["name"] for item in templama_data]
            entity_data += templama_labels

        def filter_common_vocab_subset(word):
            if word in self.common_vocab:
                return True
            else:
                return False
        if common_vocab_intersection:
            # intersect answer space with common vocab
            entity_data = list(filter(filter_common_vocab_subset,entity_data))
        
        # remove repeat tokens in the answer space
        entity_data = list(set(entity_data))
        index_list = self.get_vocab_subset_indices(entity_data,tokenizer)
        return torch.tensor(index_list)


    def generate_model_bias(self, model:PreTrainedModel, tokenizer:AutoTokenizer, prompt_only, save_path=None, vocab_subset_indices=None, return_logits=False,):
        """Get the model bias distribution on the vocab (or vocab subset) for the given prompt.
        
        This function is designed for manual prompts with the typically format `Obama was born in [MASK].`
        The model bias refers to the logits when masking the Obama with a meaningless token like `[MASK]`.

        Args:
            model (PreTrainedModel): the underlying pretrained model.
            tokenizer (AutoTokenizer): tokenier of the model.
            prompt_only (str): a manual prompt with subject masked by a meaningless token. For instance
                `[MASK] was born in [MASK].`, the first [MASK] usually is a subject like `Obama`.
            save_path (str, optional): save the biased probabilies of all tokens, if this is not None, 
                return_logits must be false and vocab_subset_indices must be None. Defaults to None.
            vocab_subset_indices (tensor, optional): filter the model bias distribution on the specified vocab subset. 
                Defaults to None.
            return_logits (bool, optional): return either the logits form or the probs form of the bias distribution . 
                Defaults to False, that is returning the probs form.

        Returns:
           tensor : a logits tensor or a probs tensor representing the model bias for the given prompt.
        """
        
        if not isinstance(prompt_only,list):
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
            assert return_logits==False
            assert vocab_subset_indices==None
            probs = final_output.tolist()
            index = list(range(len(probs)))
            tokens = tokenizer.convert_ids_to_tokens(index)
            output = dict(zip(index,zip(probs,tokens)))
            create_dir(save_path)
            with open(save_path, "w") as f:
                json.dump(output,f,indent=4)
        

        return final_output
   

    def get_manual_prompt_bias(self, model,tokenizer, template, template_embeddings=None, object_index='unk'):
        """Generate the model bias tensor used to debias. Used only for manual prompts or its variants.

        Args:
            model (PreTrainedModel): the underlying pretrained model.
            tokenizer (AutoTokenizer): tokenier of the model.
            template (str): a manual prompt with subject masked by a meaningless token. For instance
                `[MASK] was born in [MASK].`, the first [MASK] usually is a subject like `Obama`.
                A special case is that if this param equal to `soft mask`, then we ignore this param and 
                use `template_embeddings` instead.
            template_embeddings (tensor, optional): the embedding tensor of the template. Defaults to None.
                This param is used when there are soft tokens in the template.
            y_mask_index (str, optional): the object token index in the template. Defaults to 'unk', which means 
                we will take the index of first `[MASK]` token found reversely as the y_mask_index. 

        Returns:
            tuple : a tuple of (bias_logits, bias_vector). `bias_logits` refers to the normalized bias logits
                `bias_vector` refers to the raw bias vector.
        """

        # The following is an unified framework splitting the process of language model into 3 parts
        # 1. transformer block
        # 2. transform layer
        # 3. decoder, not used in this function
        if isinstance(model, BertForMaskedLM):
            transformer_blocks = model.bert
            tranform_layer  = model.cls.predictions.transform
            decoder = model.cls.predictions.decoder
        elif isinstance(model, RobertaForMaskedLM):
            transformer_blocks = model.roberta
            tranform_layer = nn.Sequential(model.lm_head.dense, nn.GELU(), model.lm_head.layer_norm)
            decoder = model.lm_head.decoder
        
        # when template=="soft mask", template_embeddings!=None
        if template=="soft mask":
            assert template_embeddings!=None, "when template==\"soft mask\", template_embeddings!=None"
            model_input = template_embeddings
            model_input = {key:value.to(model.device) for key,value in model_input.items()}
        else:
            model_input = tokenizer(template,return_tensors="pt")
            model_input = {key:value.to(model.device) for key,value in model_input.items()}
        
        # step 1
        transformer_output = transformer_blocks(**model_input)[0]
        
        # find the the object token index in the template
        if object_index=='unk':
            object_index = 0
            mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
            input_token_ids = model_input["input_ids"][0].tolist()
            # from back to front
            for i,e in enumerate(reversed(input_token_ids)):
                if e == mask_id:
                    object_index = -(i+1) 
                    break
            assert object_index < 0
        else:
            object_index = int(object_index)

        # step 2
        transformer_output = transformer_output[0][object_index]
    
        bias_vector = tranform_layer(transformer_output)
        norm2 = torch.norm(bias_vector)
        bias_logits = decoder(bias_vector) / norm2
        bias_vector = bias_vector.detach()
        bias_logits = bias_logits.detach()

        return bias_logits, bias_vector    


    def get_continue_prompt_bias(self, promptModel, prompt_template, continue_prompt:str, ckpt_path=None, num_tokens=0):
        """Generate the model bias tensor used to debias. Used only for Openprompt continue prompts.

        Args:
            promptModel (PromptModel): an OpenPrompt wraped pretrained model.
            prompt_template (Template): an OpenPrompt Template object.
            continue_prompt (str): type of the continue prompt, can be 'prefix' or 'optiprompt'.
                `prefix` adds soft tokens before the input. `optiprompt` adds soft tokens inside the input.
            num_tokens (int, optional): the soft tokens number. Used only when continue_prompt=='prefix'. Defaults to 0.
            ckpt_path (str, optional): the loaded checkpoint of the continue prompt. Defaults to None.

        Returns:
            tuple: a tuple of (model_bias_output, bias_logits, bias_vector). `model_bias_output` refers to 
                the biased logits on the vocabulary. `bias_logits` refers to the normalized bias vector
                `bias_vector` refers to the raw bias vector.
        """
        # The following is an unified framework splitting the process of language model into 3 parts
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
        
        if ckpt_path!=None:
            soft_embeds = torch.load(ckpt_path)
            if continue_prompt=="prefix":
                promptModel.template.soft_embeds.data.copy_(soft_embeds)
            elif continue_prompt.lower()=="optiprompt":
                promptModel.template.soft_embedding.weight.data.copy_(soft_embeds)

        template_only = [{"sub_label":self.tokenizer.mask_token,"obj_label":"no"}]
        temp_dataloader = PromptDataLoader(dataset=self.wrap_input_examples(template_only,self.tokenizer), template=prompt_template, tokenizer=self.tokenizer,
            tokenizer_wrapper_class=self.WrapperClass, max_seq_length=256, batch_size=1)
        for inputs in temp_dataloader:
            inputs = inputs.cuda()
            batch = promptModel.template.process_batch(inputs)
            input_batch = {key: batch[key] for key in batch if key in promptModel.forward_keys}
            transformer_output = transformer_blocks(**input_batch)[0]
            # post process for `prefix` continue prompt
            if continue_prompt=="prefix":
                # drop the soft embedding in embedding space
                transformer_output = transformer_output[:,num_tokens:,:]
            mask_token_bert = transformer_output[torch.where(inputs['loss_ids']>0)]
            mask_token_features = tranform_layer(mask_token_bert)
            norm2 = torch.norm(mask_token_features, p=2)
            bias_logits = decoder(mask_token_features[0]) / norm2
        bias_vector = mask_token_features[0]
        model_bias_output = bias_logits * norm2

        return model_bias_output, bias_logits, bias_vector


    def get_prompt_bias_by_sample(self, promptModel:PromptModel, support_dataloader, evaluateMode=True, num_tokens=5):
        """Generate the model bias tensor used to debias based on sampling strategy. Suitable for both manual and continue prompts.


        Args:
            promptModel (PromptModel): an OpenPrompt wrapeed model.
            support_dataloader (_type_): the sample dataset used to esitimate the bias tensor.
                200 samples will be a good number in practice.
            evaluateMode (bool, optional): Whether this func is running in evaluate model. Defaults to True.
            num_tokens (int, optional): the soft tokens number. Used only when continue_prompt=='prefix'. Defaults to 5.

        Returns:
            tuple : a tuple of (bias_logits, bias_vector). `bias_logits` refers to the normalized bias vector
                `bias_vector` refers to the raw bias vector.
        """
        if evaluateMode==True:
            promptModel.eval()

        # The following is an unified framework splitting the process of language model into 3 parts
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
        
        # When this code is runing in training model, it may be hard to process the whole 200 support sample 
        # in 1 bacth, so the DDP in pytorch can help reduce the memory problems.

        # if not isinstance(transformer_blocks, torch.nn.DataParallel):
        #     transformer_blocks  = torch.nn.DataParallel(transformer_blocks)
        
        # assert isinstance(promptModel, PromptForClassification), "Current this func only support PromptForClassification"
        
        all_prediction_vectors = None
        # 如果输入的模型是分类模型，在前向过程中会稍有不同
        if isinstance(promptModel, PromptForClassification):
            for inputs in support_dataloader:
                inputs = inputs.to(promptModel.device)
                # copy the following code from OpenPrompt library
                batch = promptModel.template.process_batch(inputs)
                input_batch = {key: batch[key] for key in batch if key in promptModel.prompt_model.forward_keys}
                transformer_output = transformer_blocks(**input_batch)[0]
                # post process only for soft template
                if isinstance(promptModel.template, SoftTemplate):
                    transformer_output = transformer_output[:,num_tokens:,:]
                mask_token_transformer_output = transformer_output[torch.where(inputs['loss_ids']>0)]
                mask_token_features = tranform_layer(mask_token_transformer_output)
                
                # accumlate the debias vector
                if all_prediction_vectors==None:
                    all_prediction_vectors = mask_token_features
                else:
                    all_prediction_vectors = torch.cat((all_prediction_vectors, mask_token_features), dim=0)
        else:
            for inputs in support_dataloader:
                inputs = inputs.cuda()
                # copy the following code from OpenPrompt library
                batch = promptModel.template.process_batch(inputs)
                input_batch = {key: batch[key] for key in batch if key in promptModel.forward_keys}
                transformer_output = transformer_blocks(**input_batch)[0]
                # post process only for soft template
                if isinstance(promptModel.template, SoftTemplate):
                    transformer_output = transformer_output[:,num_tokens:,:]
                mask_token_transformer_output = transformer_output[torch.where(inputs['loss_ids']>0)]
                mask_token_features = tranform_layer(mask_token_transformer_output)
                
                # accumlate the debias vector
                if all_prediction_vectors==None:
                    all_prediction_vectors = mask_token_features
                else:
                    all_prediction_vectors = torch.cat((all_prediction_vectors, mask_token_features), dim=0)
             
        # avg_prediction_vector = torch.mean(all_prediction_vectors,dim=0)
        # avg_norm = torch.norm(avg_prediction_vector).item()
        # avg_logits = torch.mean(decoder(all_prediction_vectors), dim=0)
        # normalized_avg_logits = avg_logits / avg_norm

        # 方案1 直接对vector做一个平均
        avg_prediction_vector = torch.mean(all_prediction_vectors,dim=0)
        avg_norm = torch.norm(avg_prediction_vector).item()
        avg_logits = torch.mean(decoder(all_prediction_vectors), dim=0)
        normalized_avg_logits = avg_logits / avg_norm

        # 方案2 对vector的方向做一个平均
        # prediction_vector_norm = torch.norm(all_prediction_vectors,dim=1)
        # normalied_prediction_vector = (all_prediction_vectors.T / prediction_vector_norm).T
        # avg_normalized_prediction_vector =torch.mean(normalied_prediction_vector,dim=0)
        # avg_prediction_vector = avg_normalized_prediction_vector

        
        if evaluateMode==True:
            normalized_avg_logits = normalized_avg_logits.detach()
            avg_prediction_vector = avg_prediction_vector.detach()

        bias_logits = normalized_avg_logits
        bias_vector = avg_prediction_vector

        return bias_logits, bias_vector


    def get_probs_by_sample(self, promptModel:PromptModel, support_dataloader, subvocab_indices_list, bias_vector, num_tokens=5):
        """Generate the avg probs on the support_dataloader based on sampling strategy. Suitable for both manual and continue prompts.


        Args:
            promptModel (PromptModel): an OpenPrompt wrapeed model.
            support_dataloader (Dataset): the sample dataset used to esitimate the bias tensor.
                200 samples will be a good number in practice.
            bias_vector (tensor): the prompt bias representation vector.

        Returns:
            probs : a list of the probabilities.
        """
        promptModel.eval()

        # The following is an unified framework splitting the process of language model into 3 parts
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
        
        all_prediction_vectors = None

        for inputs in support_dataloader:
            inputs = inputs.cuda()
            # copy the following code from OpenPrompt library
            batch = promptModel.template.process_batch(inputs)
            input_batch = {key: batch[key] for key in batch if key in promptModel.forward_keys}
            transformer_output = transformer_blocks(**input_batch)[0]
            # post process only for soft template
            if isinstance(promptModel.template, SoftTemplate):
                transformer_output = transformer_output[:,num_tokens:,:]
            mask_token_transformer_output = transformer_output[torch.where(inputs['loss_ids']>0)]
            mask_token_features = tranform_layer(mask_token_transformer_output)
            
            # accumlate the debias vector
            if all_prediction_vectors==None:
                all_prediction_vectors = mask_token_features
            else:
                all_prediction_vectors = torch.cat((all_prediction_vectors, mask_token_features), dim=0)

        # 方案1 直接对vector做一个平均
        # avg_prediction_vector = torch.mean(all_prediction_vectors,dim=0)
        # avg_norm = torch.norm(avg_prediction_vector).item()
        logits = decoder(all_prediction_vectors)
        logits = logits[:, subvocab_indices_list]
        avg_logit = torch.mean(logits, dim=0)
        normalize_logits = avg_logit - torch.mean(avg_logit)
        norm_probs = torch.softmax(normalize_logits,dim=0)
        

        # 方案2 对vector的方向做一个平均
        # prediction_vector_norm = torch.norm(all_prediction_vectors,dim=1)
        # normalied_prediction_vector = (all_prediction_vectors.T / prediction_vector_norm).T
        # avg_normalized_prediction_vector =torch.mean(normalied_prediction_vector,dim=0)
        # avg_prediction_vector = avg_normalized_prediction_vector

        # 使用debias过后的vector来估计prompt bias
        bias_vector = bias_vector.cuda()
        batched_debiased_logits = None
        D_T = bias_vector / torch.norm(bias_vector)
        for i in range(len(all_prediction_vectors)):
            D_Tx = all_prediction_vectors[i] / torch.norm(all_prediction_vectors[i])
            D_debias = (D_Tx - D_T) / torch.norm(D_Tx - D_T)
            V_debias = D_debias * torch.norm(all_prediction_vectors[i])
            debiased_logits = decoder(V_debias).unsqueeze(dim=0)
            if batched_debiased_logits == None:
                batched_debiased_logits = debiased_logits
            else:
                batched_debiased_logits = torch.cat((batched_debiased_logits,debiased_logits),dim=0)

        batched_debiased_logits = batched_debiased_logits[:, subvocab_indices_list]
        avg_debias_logit = torch.mean(batched_debiased_logits, dim=0)
        normalize_debias_logits = avg_debias_logit - torch.mean(avg_debias_logit)
        norm_debias_probs = torch.softmax(normalize_debias_logits,dim=0)


        # 测试精度
        # allpreds = torch.argmax(batched_debiased_logits[:,subvocab_indices_list],dim=1)
        # allpreds = torch.tensor(subvocab_indices_list)[allpreds].tolist()
        # alllabels = []
        # for inputs in support_dataloader:
        #     labels = inputs["label"]
        #     alllabels += labels.cpu().tolist()
        
        # acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
        # print(acc)
        # exit()

        return norm_probs, norm_debias_probs

    
    def save_KL_results(self, results, path):
        """Save the KL results into the specific path.

        Args:
            results (tuple): the output of the func `quantify_prompt_bias`
            path (str): the target saving path
        """
        reformat_results = {
            "raw_KL": results[0],
            "debiased_KL": results[1],
            "details": results[2]
        }
        str = json.dumps(reformat_results, indent=4)
        with open(path, "w") as f:
            f.write(str)


    def generate_image(self, before_debias_preds, after_debias_preds, labels, model_bias, save_path="test.png"):
        """Draw some images before and after debiasing on a given dataset.

        The image layout are as follows:
        1.sample space                 2.sample space + preds dis before debiasing
        3.sample space + labels dis    4.sample space + preds dis after debiasing
        5.preds scatter dis            6.preds==labels dis (true preds dis)

        Args:
            before_debias_preds (list): preds before debiasing.
            after_debias_preds (_type_): preds after debiasing.
            labels (list): labels
            model_bias (tensor):  a logits tensor representing the model bias for the given prompt.
            save_path (str, optional): the save path of the image. Defaults to "test.png".
        """
        labels = labels
        preds = before_debias_preds
        count_labels  = Counter(labels)
        count_preds  = Counter(preds)

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

        debiased_preds = after_debias_preds
        count_debias_preds = Counter(debiased_preds)
        raw_sorted_debais_preds = sorted(count_debias_preds.items())
        raw_debais_preds_x = [i[0] for i in raw_sorted_debais_preds]
        raw_debais_preds_y = [model_bias[i[0]].item() for i in raw_sorted_debais_preds]
        axs[1,1].plot(raw_debais_preds_x, raw_debais_preds_y,'o',color="orange")

        c = [i[1] for i in raw_sorted_debais_preds]
        axs[2,0].scatter(raw_debais_preds_x, raw_debais_preds_y,s=c,)

        acc_index = np.where(np.array(debiased_preds)==np.array(labels))
        acc_labels = np.array(labels)[acc_index]
        count_acc_labels = Counter(acc_labels)
        count_acc_labels = sorted(count_acc_labels.items())
        acc_labels_x = [i[0] for i in count_acc_labels]
        acc_labels_y = [model_bias[i[0]] for i in count_acc_labels]
        acc_labels_num = np.array([i[1] for i in count_acc_labels])
        axs[2,1].scatter(list(range(len(model_bias))),model_bias, alpha=0.2)
        axs[2,1].scatter(acc_labels_x,acc_labels_y,cmap="Oranges",s=acc_labels_num*10)
        plt.savefig(save_path)


    def calibrate(self, prompt_model: PromptForClassification, dataloader: PromptDataLoader) -> torch.Tensor:
        """Calibrate. See `Paper <https://arxiv.org/abs/2108.02035>`
        
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


    def manual_prompt(self, relation, debias=False, vocab_subset_filter=True, vocab_subset="common_vocab",test_data_path=None,
        embeddings_renormalize=False, prompt="LAMA", sampling_debias=False, calibrate=False, filter_biased_token_nums=0,
        ablation_no_normalization=False,
        ablation_no_rescale=False,):
        """Probe factual knowledge in the model with a manual prompt.

        Args:
            relation (str): the type of probed facts.  
            debias (bool, optional): Whether to appling our debiasing algorithm when probing facts in the model. Defaults to False.
            vocab_subset_filter (bool, optional): whether to limit the output vocabulary space of the model. Defaults to True.
            vocab_subset (str, optional): A smaller vocab used to probe facts. Defaults to "common_vocab".
            test_data_path (str, optional): A dataset used to evaluate how many facts contained in the model. Defaults to None.
            embeddings_renormalize (bool, optional): Whether to renormalize the embeddings of the model before probing. Defaults to False.
            prompt (str, optional): The prompt series used in probing. Defaults to "LAMA". Could also be "LPAQA", "AutoPrompt".
            sampling_debias (str, optional): Whether to use sampling debiasing strategy to mitigate the prompt bias.
            filter_biased_token_nums (int, optional): A filtered applied to the dataset to filter out the current prompt biased data. 
                Default to 0 that do not filter out any data. Set it to 1 means filtering out data prompt biased mostly from the dataset.

        Raises:
            ValueError: when `vocab_subset` is set to an invalid value.
            ValueError: When `self.model_name` is not start with roberta or bert.
            ValueError: When `prompt` is set to an invalid value.

        Returns:
            tuple : a tuple of (acc, (probs, sub_labels, allpreds, alllabels)).
        """
        plm = self.plm
        tokenizer = self.tokenizer
        model_config = self.model_config
        WrapperClass = self.WrapperClass
        model = plm
        

        if embeddings_renormalize:
            config = model_config
            renormal = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            renormal.weight.data = model.cls.predictions.decoder.weight.data.clone()
            renormal.weight.data = F.normalize(renormal.weight.data, p=2,dim=1)
            output_embedding_backup = model.get_output_embeddings()
            model.set_output_embeddings(renormal)
            model.bert.embeddings.word_embeddings.weight = renormal.weight

        if vocab_subset_filter:
            if vocab_subset=="common_vocab":
                subset_indices = self.get_common_vocab_indices(tokenizer)
            elif vocab_subset=="answer_type_tokens":
                subset_indices = self.get_answer_entity_indices(relation, tokenizer)
            else:
                raise ValueError(f"an invalid param: {vocab_subset}")
        else:
            subset_indices = None
        
        dataset = {}
        
        # default test dataset
        if test_data_path is None:
            lama_data_path = os.path.join(self.lama_data_dir,f"{relation}.jsonl")
            test_data_path = lama_data_path
        

        # choose a suitable manual prompt
        templates = None
        if prompt=="LAMA":
            templates = self.lama_template
        elif prompt=="LPAQA":
            templates = self.LPAQA_template
        elif prompt=="AutoPrompt":
            if self.model_name.startswith("roberta"):
                templates = self.AutoPrompt_tempalte_roberta
            elif self.model_name.startswith("bert"):
                templates = self.AutoPrompt_template_bert
            else:
                raise ValueError(f"an invalid param: {self.model_name}")
        else:
            raise ValueError(f"an invalid param: {prompt}")
        raw_template = templates[relation]

        # A special process for AutoPrompt
        # In some template of AutoPrompt, for example, P37 "[X]inen dialects resembled officially exclusively [Y].", 
        # there is a postfix after [X]. 
        first_token = raw_template.split()[0]
        autoprompt_postfix = first_token[3:] if prompt=="AutoPrompt" else ''

        # transform to the template with OpenPrompt's format.
        template = raw_template.replace(first_token,'{"placeholder":"text_a"}') if prompt=="AutoPrompt" else \
                    raw_template.replace("[X]", '{"placeholder":"text_a"}')
        template = template.replace("[Y]", '{"mask"}')
        prompt_template = ManualTemplate(tokenizer=tokenizer, text=template)

        # constrct a prompt_pnly_template used to calculate the prompt bias vector
        prompt_only_tempalte = raw_template.replace("[X]",self.tokenizer.mask_token).replace("[Y]", self.tokenizer.mask_token)
        bias_logits, bias_vector = self.get_manual_prompt_bias(model,tokenizer, template=prompt_only_tempalte)


        # 导入数据集的时候，添加过滤功能，过滤掉prompt bias最偏向的前n个tokens
        subset_indices = subset_indices.to(bias_logits.device)

        filtered_logits = bias_logits.index_select(dim=0, index=subset_indices)
        if filter_biased_token_nums<=filtered_logits.shape[0]:
            _, prompt_biased_indices = torch.topk(filtered_logits, k=filter_biased_token_nums)
            prompt_biased_vocab_indices = subset_indices[prompt_biased_indices].tolist()
        else:
            # 超出了数据集大小，直接丢弃掉这个数据集
            prompt_biased_vocab_indices = subset_indices.tolist()
        

        raw_test_data, _ = self.load_data(test_data_path, common_vocab=self.common_vocab, filter_out_tokens=prompt_biased_vocab_indices)

        # 处理空数据集的特别情况
        # print("数据集长度为{}".format(len(raw_test_data)))
        if len(raw_test_data)==0:
            acc, sub_labels, probs, allpreds, alllabels = 0, [], None, [], [] 
        else:
            dataset["test"] = self.wrap_input_examples(raw_test_data, tokenizer,autoprompt_postfix=autoprompt_postfix)
            sub_labels = [d["sub_label"] for d in raw_test_data]

            # get the max pad number in dataset
            max_tokens_len = self.compute_max_pad(dataset, WrapperClass, tokenizer, prompt_template)
            # Note: Do not shuffle!
            test_dataloader = PromptDataLoader(dataset=dataset["test"], template=prompt_template, tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass, max_seq_length=max_tokens_len, 
                batch_size=16,shuffle=False)
            
            promptModel = PromptModel(
                template = prompt_template,
                plm = plm,
                freeze_plm=True
            )
            promptModel.cuda()
            
            if sampling_debias==True:
                # support dataset for sampling
                support_sampler = FewShotSampler(num_examples_total=200, also_sample_dev=False)
                dataset['support'] = support_sampler(dataset['test'], seed=self.seed)
                support_dataloader = PromptDataLoader(dataset=dataset["support"], template=prompt_template, tokenizer=tokenizer,
                    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_tokens_len,
                    batch_size=16, shuffle=False)
                bias_logits, bias_vector = self.get_prompt_bias_by_sample(promptModel, support_dataloader)
            
            cc_logits = torch.norm(bias_vector)*bias_logits
            if debias:
                acc,(probs, allpreds, alllabels) = self.evaluate(promptModel,test_dataloader, bias_logits=bias_logits, bias_vector=bias_vector,
                    vocab_subset_indices=subset_indices, calibrate=calibrate, calib_logits=cc_logits,
                    ablation_no_normalization=ablation_no_normalization,
                    ablation_no_rescale=ablation_no_rescale,)
            else:
                acc,(probs, allpreds, alllabels) = self.evaluate(promptModel,test_dataloader, vocab_subset_indices=subset_indices)

        # recover the embeddings
        if embeddings_renormalize==True:
            output_embedding_backup = output_embedding_backup.to(self.plm.device)
            self.plm.set_output_embeddings(output_embedding_backup)
            self.plm.bert.embeddings.word_embeddings.weight = output_embedding_backup.weight

        return round(acc,5), (probs, sub_labels, allpreds, alllabels)


    def manual_prompt_for_templama(self, relation,debias=False, vocab_subset_filter=True, vocab_subset="common_vocab"):
        """use templama to evaluate the temporary factual knowledge contained in the model

        Args:
            relation (str): the type of probed facts.  
            debias (bool, optional): Whether to appling our debiasing algorithm when probing facts in the model. Defaults to False.
            vocab_subset_filter (bool, optional): whether to limit the output vocabulary space of the model. Defaults to True.
            vocab_subset (str, optional): A smaller vocab used to probe facts. Defaults to "common_vocab".

        Raises:
            ValueError: when `vocab_subset` is set to an invalid value.

        Returns:
            tuple : a tuple of (acc, (probs, sub_labels, allpreds, alllabels)).
        """
        plm = self.plm
        tokenizer = self.tokenizer
        model = plm
        model.eval()

        if vocab_subset_filter:
            if vocab_subset=="common_vocab":
                vocab_subset_indices = self.get_common_vocab_indices(tokenizer)
            elif vocab_subset=="answer_type_tokens":
                vocab_subset_indices = self.get_answer_entity_indices(relation, tokenizer, include_templama_labels=True)
            else:
                raise ValueError(f"an invalid param:{vocab_subset}")
        else:
            vocab_subset_indices = None
      
        dataset = load_jsonl(os.path.join(self.templama_data_dir,f"{relation}.jsonl"))
        
        # The following is an unified framework splitting the process of language model into 3 parts
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
        
        allpreds = []
        alllabels = []
        allsubjs = []
        probs = None

        model.cuda()

        for data in tqdm(dataset):
            raw_template = "In [year], [X] holds the position of [Y]."
            t1 = data["query"].find("holds")
            subject = data["query"][0:t1-1]
            allsubjs.append(subject)
            prompt_based_input = data["query"].replace("_X_","[MASK] in {}".format(data["date"]))
            prompt_only_input = raw_template.replace("[X]",tokenizer.mask_token).replace("[Y]",tokenizer.mask_token).replace("[year]",data["date"])
            label = data["answer"][0]["name"]
            label = tokenizer.convert_tokens_to_ids(label)

            model_input = tokenizer(prompt_based_input,return_tensors="pt")
            model_input = {k: v.cuda() for k, v in model_input.items()}

            # find [mask] position
            y_mask_index = 0
            mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
            input_token_ids = model_input["input_ids"][0].tolist()
            for i,e in enumerate(reversed(input_token_ids)):
                if e == mask_id:
                    y_mask_index = -(i+1) 
                    break
            assert y_mask_index < 0

            if debias==False:
                output = model(**model_input).logits
                batched_debiased_logits = output[0][y_mask_index]
                batched_debiased_logits = batched_debiased_logits.unsqueeze(dim=0)
            else:
                transformer_output = transformer_blocks(**model_input)[0]
                mask_token_transformer_output = transformer_output[:,y_mask_index,:]
                mask_token_features = tranform_layer(mask_token_transformer_output)
                norm2 = torch.norm(mask_token_features[0], p=2)
                batched_debiased_logits = decoder(mask_token_features) / norm2    
                bias_logits, bias_vector = self.get_manual_prompt_bias(model,tokenizer,prompt_only_input)
                bias_logits = bias_logits.to(batched_debiased_logits.device)
                bias_vector = bias_vector.to(batched_debiased_logits.device)
                batched_debiased_logits -= bias_logits
                # scale the debiased logits to a resonable range
                norm_bias = torch.norm(bias_vector,p=2)
                norm_preds = norm2
                normalized_pred_vecs = (mask_token_features.T/norm_preds).T
                normlaized_bias_vec = bias_vector/norm_bias
                debais_vecs = normalized_pred_vecs - normlaized_bias_vec
                norm_debias = torch.norm(debais_vecs,dim=-1)
                batch_scaling_vec = norm_preds/norm_debias
                batched_debiased_logits = (batched_debiased_logits.T*batch_scaling_vec).T
            
            if vocab_subset_indices != None:
                vocab_subset_indices = vocab_subset_indices.to(model.device)
                batched_debiased_logits = batched_debiased_logits.index_select(dim=1,index=vocab_subset_indices)

            predict_index = torch.argmax(batched_debiased_logits, dim=-1)
            if probs == None:
                probs = batched_debiased_logits
            else:
                probs = torch.cat((probs,batched_debiased_logits),dim=0)
            
            if vocab_subset_indices != None:
                predict_index = vocab_subset_indices[predict_index]
            allpreds.extend(predict_index.cpu().tolist())
            alllabels.append(label)
    
        acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
        # both allpreds and alllabels contains indices of the original vocabulary.
        return acc, (probs, allsubjs, allpreds, alllabels)


    def continue_prompt(self, relation, embedding_save_dir, continue_prompt="prefix", num_tokens=5, vocab_subset_filter=True, vocab_subset="answer_type_tokens", 
        random_init="none", evaluate_mode=False, filter_biased_token_nums=0,
        ablation_no_normalization=False,
        ablation_no_rescale=False,):
        """Probe factual knowledge in the model with a continue prompt.

        Args:
            relation (str): the type of probed facts.
            embedding_save_dir (str): the save path of the continue prompt's embeddings,
                 which had been tuned on the training dataset.
            continue_prompt (str, optional): type of the continue prompt to use. Defaults to "prefix".
            num_tokens (int, optional): the number of the soft tokens in the continue prompt. Defaults to 5.
            vocab_subset_filter (bool, optional): whether to limit the output vocabulary space of the model. Defaults to True.
            vocab_subset (str, optional): A smaller vocab used to probe facts. Defaults to "common_vocab".
            random_init (str, optional): Whether use the random weigths to init the soft tokens. Defaults to "none".
            evaluate_mode (bool, optional): Whether this func is run in evaluate mode. Defaults to False, means that 
                first train the prompt then evaluate it. If this param is `True`, then only evaluate.

        Raises:
             ValueError: when `vocab_subset` is set to an invalid value.

        Returns:
            tuple : a tuple of (model_bias_output, results). `model_bias_output` refers to 
                the biased logits on the vocabulary. `results` is an dict containing raw and debiased outputs
                on three test datasets.
        """
        plm, tokenizer, model_config, WrapperClass = self.plm,self.tokenizer,self.model_config,self.WrapperClass
        
        if random_init=="all":
            plm = AutoModelForMaskedLM.from_config(model_config)
       
        dataset = {}
        lama_data_path = os.path.join(self.lama_data_dir,f"{relation}.jsonl")
        uni_data_path = os.path.join(self.wiki_uni_data_dir,f"{relation}.json")
        lama_whu_data_path = os.path.join(self.lama_whu_data_dir,f"{relation}.jsonl")
        data_paths = [lama_data_path,uni_data_path,lama_whu_data_path]

        autoprompt_train_data_path = os.path.join(self.auto_prompt_dir,relation,"train.jsonl")
        autoprompt_valid_data_path = os.path.join(self.auto_prompt_dir,relation, "dev.jsonl")
        
        # TODO add dataset filter

        raw_train_data, resample_weight = self.load_data(autoprompt_train_data_path,return_resample_weights=True)
        raw_valid_data, _ = self.load_data(autoprompt_valid_data_path)


        dataset["train"] = self.wrap_input_examples(raw_train_data, tokenizer)
        dataset["valid"] = self.wrap_input_examples(raw_valid_data, tokenizer)
        
        
        if continue_prompt=="prefix":
            current_template = '{"placeholder":"text_a"} '
            current_template += '{"mask"} .'

            prompt_template = SoftTemplate(
                model=plm, tokenizer=tokenizer, text=current_template,num_tokens=num_tokens)
            # following OptiPrompt to use a normal distribution to init the weights 
            prompt_template.soft_embeds.data.normal_(mean=0.0, std=model_config.initializer_range)
        elif continue_prompt=="optiprompt":
            current_template = '{"placeholder":"text_a"} '
            current_template += '{"soft":None,"duplicate":' + str(num_tokens) + '}'
            current_template += '{"mask"} .'
            prompt_template = MixedTemplate(
                model=plm, tokenizer=tokenizer, text=current_template)

            prompt_template.soft_embedding.weight.data.normal_(mean=0.0, std=model_config.initializer_range)
        else:
            print(f"invalid continue prompt paramters: {continue_prompt}")
            exit(-1)

        # A pre-knowledge that all samples in our test datases are fewer than 64 tokens
        max_tokens_len = 64

        # If you want to uniformly resample the training data according their types,
            # please use the custom_sampler instead as follows 
        # sampler = WeightedRandomSampler(resample_weight, len(resample_weight))
        # train_dataloader = PromptDataLoader(dataset=dataset["train"], template=prompt_template, tokenizer=tokenizer,
        #         tokenizer_wrapper_class=WrapperClass, max_seq_length=max_tokens_len+5,
        #         batch_size=16,custom_sampler=sampler)
        train_dataloader = PromptDataLoader(dataset=dataset["train"], template=prompt_template, tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass, max_seq_length=max_tokens_len,
                batch_size=16)
        valid_dataloader = PromptDataLoader(dataset=dataset["valid"], template=prompt_template, tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass, max_seq_length=max_tokens_len,
                batch_size=16)
    
        if vocab_subset_filter:
            if vocab_subset=="common_vocab":
                subset_indices = self.get_common_vocab_indices(self.tokenizer)
            elif vocab_subset=="answer_type_tokens":
                subset_indices = self.get_answer_entity_indices(relation, tokenizer)
            else:
                raise ValueError(f"invalid param:{vocab_subset}")
        else:       
            subset_indices = None
    
        promptModel = PromptModel(
            template = prompt_template,
            plm = plm,
            freeze_plm=True
            
        )
        promptModel.cuda()
        best_ckpt = os.path.join(embedding_save_dir, f"weight_save/{relation}/model_full-prompt_random_init.ckpt")
        if evaluate_mode==True:
            # only do evaluation
            if not os.path.exists(best_ckpt):
                # try to find the ckpt file in parent dir
                dirs = embedding_save_dir.split('/')
                best_ckpt_dir = '/'.join(dirs[:-3]+[dirs[-1]])
                best_ckpt = os.path.join(best_ckpt_dir, f"weight_save/{relation}/model_full-prompt_random_init.ckpt")
        else:
            # start training
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
            create_dir(best_ckpt)
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

        model_bias_output, bias_logits, bias_vector = self.get_continue_prompt_bias(promptModel, prompt_template, continue_prompt, best_ckpt,num_tokens)
        subset_indices = subset_indices.to(bias_logits.device)
        filtered_logits = bias_logits.index_select(dim=0, index=subset_indices)
        if filter_biased_token_nums<=filtered_logits.shape[0]:
            _, prompt_biased_indices = torch.topk(filtered_logits, k=filter_biased_token_nums)
            prompt_biased_vocab_indices = subset_indices[prompt_biased_indices].tolist()
        else:
            # 超出了数据集大小，直接丢弃掉这个数据集
            prompt_biased_vocab_indices = subset_indices.tolist()


        test_dataloader = {}
        raw_test_data = []
        for data_path in data_paths:
            data,_ = self.load_data(data_path, common_vocab=self.common_vocab, filter_out_tokens=prompt_biased_vocab_indices)
            raw_test_data.append(data)
        dataset["test"] = {}
        dataset["test"]["lama"] =  self.wrap_input_examples(raw_test_data[0], tokenizer)
        dataset["test"]["uni"] = self.wrap_input_examples(raw_test_data[1], tokenizer)
        dataset["test"]["lama_whu"] = self.wrap_input_examples(raw_test_data[2], tokenizer)
        
        sub_labels = {"lama":{},"uni":{},"lama_whu":{}}
        sub_labels["lama"] = [d["sub_label"] for d in raw_test_data[0]]
        sub_labels["uni"] = [d["sub_label"] for d in raw_test_data[1]]
        sub_labels["lama_whu"] = [d["sub_label"] for d in raw_test_data[2]]

        results  = {}
        for _ in ["lama","uni","lama_whu"]:
            if len(dataset["test"][_])==0:
                origin_output = 0, (None, [], [], [])
                debias_output = 0, (None, [], [], [])
            else:
                test_dataloader[_] = PromptDataLoader(dataset=dataset["test"][_], template=prompt_template, tokenizer=tokenizer,
                    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_tokens_len,
                    batch_size=16)
                debias_output = self.evaluate(promptModel, test_dataloader[_], bias_logits=bias_logits, bias_vector=bias_vector, vocab_subset_indices=subset_indices,
                    soft_tempalte_ckpt=best_ckpt,continue_prompt=continue_prompt,num_tokens=num_tokens,
                    ablation_no_normalization=ablation_no_normalization,
                    ablation_no_rescale=ablation_no_rescale,)
                origin_output = self.evaluate(promptModel, test_dataloader[_], vocab_subset_indices=subset_indices, soft_tempalte_ckpt=best_ckpt,continue_prompt=continue_prompt,num_tokens=num_tokens)
                acc_, (probs_, preds_, labels_) = debias_output
                debias_output = acc_, (probs_, sub_labels[_],preds_, labels_)
                acc_, (probs_, preds_, labels_) = origin_output
                origin_output = acc_, (probs_, sub_labels[_],preds_, labels_)
            results[_] = (debias_output,origin_output)

        return model_bias_output, results
            
    
    def experiment_prompt_only_accuracy(self, vocab_subset_filter=True, vocab_subset="answer_type_tokens", ctrl_code=[1,1,1] , prompt="LAMA",num_tokens=5, do_sample=False):
        """This exp is used to show how much the LAMA dataset and WIKI-UNI dataset benefit from prompt bias.

        We first get the prompt bias distribution, then based on the distribution to choose the answer for specified strategy. 

        Args:
            vocab_subset_filter (bool, optional): whether to limit the output vocabulary space of the model. Defaults to True.
            vocab_subset (str, optional): A smaller vocab used to probe facts. Defaults to "answer_type_tokens".
            ctrl_code (list, optional): which datasets are used in this exp. Defaults to [1,1,1], representing use all three datasets [lama, wiki-uni, lama_whu].
            prompt (str, optional): The prompt series used in exp. Defaults to "LAMA". Could also be "LPAQA", "AutoPrompt", "OptiPrompt"
            do_sample (bool, optional): sample from the output distribution probabilities to choose the final token. Defaults to False.
        """

        # save important metrics
        output_save = { 
                "LAMA": {"P":[], },
                "WIKI-UNI":{"P":[]},
                "LAMA-WHU":{"P":[]}
                }

        # choose the right prompts
        IsContinuePrompt = False
        templates = None
        if prompt=="LAMA":
            templates = self.lama_template
        elif prompt=="LPAQA":
            templates = self.LPAQA_template
        elif prompt=="AutoPrompt":
            if self.model_name.startswith("roberta"):
                templates = self.AutoPrompt_tempalte_roberta
            elif self.model_name.startswith("bert"):
                templates = self.AutoPrompt_template_bert
            else:
                raise ValueError(f"an invalid param: {self.model_name}")
        elif prompt=="OptiPrompt":
            IsContinuePrompt=True
            current_template = '{"placeholder":"text_a"} '
            current_template += '{"soft":None,"duplicate":' + str(num_tokens) + '}'
            current_template += '{"mask"} .'
            prompt_template = MixedTemplate(
                model=self.plm, tokenizer=self.tokenizer, text=current_template)
            prompt_template.soft_embedding.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
        else:
            raise ValueError(f"an invalid param: {prompt}")
        
        pbar = tqdm(total=len(self.relations)*sum(ctrl_code))

        # start 
        for relation in self.relations:
            lama_data_path = os.path.join(self.lama_data_dir,f"{relation}.jsonl")
            uni_data_path = os.path.join(self.wiki_uni_data_dir,f"{relation}.json")
            lama_whu_data_path = os.path.join(self.lama_whu_data_dir,f"{relation}.jsonl")
            data_paths = [lama_data_path,uni_data_path,lama_whu_data_path]

            if IsContinuePrompt:
                promptModel = PromptModel(
                    template = prompt_template,
                    plm = self.plm,
                    freeze_plm=True
                    
                )
                promptModel.cuda()
                root_dir = self.work_dir + f"/outputs/openprompt/continue_prompt/{self.model_name}"
                # use the exp1 weights
                save_dir = "optiprompt"+f"_{num_tokens}" + f"/debias_{vocab_subset}/"
                save_dir += "origin_embedding"
                save_dir += f"/exp_1"
                best_ckpt = os.path.join(os.path.join(root_dir, save_dir), f"weight_save/{relation}/model_full-prompt_random_init.ckpt")
                model_bias_output, bias_logits, bias_vector = self.get_continue_prompt_bias(promptModel, prompt_template, prompt, best_ckpt, num_tokens)
            else:
                raw_template =templates[relation]
                prompt_only_tempalte = raw_template.replace("[X]",self.tokenizer.mask_token).replace("[Y]", self.tokenizer.mask_token)
                bias_logits, bias_vector = self.get_manual_prompt_bias(self.plm, self.tokenizer, template=prompt_only_tempalte)
            
            origin_logits = (torch.norm(bias_vector)*bias_logits).cpu()
            
            # filter out logits with vocab_subset
            subvocab_tokens_indices = self.get_answer_entity_indices(relation, self.tokenizer) \
                                                if vocab_subset == "answer_type_tokens" else \
                                            self.get_common_vocab_indices(self.tokenizer)
            
            filtered_logits = origin_logits.index_select(dim=0, index=subvocab_tokens_indices)
            prompt_only_probs = torch.softmax(filtered_logits, dim=0)

            def make_prediction(probs:torch.Tensor, indice:torch.Tensor, do_sample=False, num=1, seed=2023):
                """make prediciton based on prompt only probs and sample strategy

                return: a token id
                """
                if do_sample==False:
                    # always selected the max id
                    index = torch.argmax(probs)
                    id = indice[index].item()
                    return [id]*num
                else:
                    # sample the target index according to probs
                    probs = np.array(probs.tolist())
                    probs /= probs.sum()
                    np.random.seed(seed)
                    index = np.random.choice(probs.shape[0],num, p=probs)
                    id = indice[index]
                    return id.tolist()
            
            # prepare dataset

            # sample output and compare with the labels

            for i,data_path in enumerate(data_paths):
                dataset = list(output_save.keys())[i]
                labels = []
                raw_test_data, _ = self.load_data(data_path, common_vocab=self.common_vocab,)
                for data_item in raw_test_data:
                    # 加入空格，使得roberta生效
                    obj_token = self.tokenizer.tokenize(' ' + data_item["obj_label"])[0]
                    label = self.tokenizer.convert_tokens_to_ids(obj_token)
                    labels.append(label)
                
                preds = make_prediction(prompt_only_probs, subvocab_tokens_indices, num=len(labels), do_sample=do_sample)
                
                acc = sum([int(i==j) for i,j in zip(preds, labels)])/len(preds)
                output_save[dataset]["P"].append({relation:acc})
                pbar.update(1)


        # calculate average accuracy
        avg_accuracy = {
            "LAMA": 0,
            "WIKI-UNI":0,
            "LAMA-WHU":0
        }

        for dataset in output_save.keys():
            all_acc = [list(item.values())[0] for item in output_save[dataset]["P"]] 
            avg_accuracy[dataset] = np.mean(all_acc)
        
        return output_save, avg_accuracy


    def experiment_renormal_vector_debais_for_manual_prompt(self, vocab_subset_filter=True, vocab_subset="answer_type_tokens", embeddings_renormalize=False,
         ctrl_code=[1,1,1], manual_prompt="LAMA", sampling_debias=False, calibrate=False, filter_biased_token_nums=0,
         ablation_no_normalization=False,
         ablation_no_rescale=False):
        """Debiasing experiments for manual prompts. 

        Args:
            vocab_subset_filter (bool, optional): whether to limit the output vocabulary space of the model. Defaults to True.
            vocab_subset (str, optional): A smaller vocab used to probe facts. Defaults to "answer_type_tokens".
            embeddings_renormalize (bool, optional): Whether to renormalize the embeddings of the model before probing. Defaults to False.
            ctrl_code (list, optional): which datasets are used in this exp. Defaults to [1,1,1], representing use all three datasets [lama, wiki-uni, lama_whu].
            manual_prompt (str, optional): The prompt series used in exp. Defaults to "LAMA". Could also be "LPAQA", "AutoPrompt".
            filter_biased_token_nums (int, optional): A filtered applied to the dataset to filter out the current prompt biased data. 
                Default to 0 that do not filter out any data. Set it to 1 means filtering out data prompt biased mostly from the dataset.
        
        Raises:
            ValueError: When `self.model_name` is not start with roberta or bert.
            ValueError: When `prompt` is set to an invalid value.
        
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
            
        root_dir = self.work_dir + f"/outputs/openprompt/manual_prompt/filter_out_{filter_biased_token_nums}_biased_tokens/{self.model_name}"
        
        if calibrate:
            save_dir = f"{manual_prompt}/calibrate_{vocab_subset}/"
        else:
            save_dir = f"{manual_prompt}/debias_{vocab_subset}/"
        
        if embeddings_renormalize==True:
            save_dir += "renormalize_embedding"
        else:
            save_dir += "origin_embedding"

        # additional output path for ablation experiments
        if ablation_no_normalization or ablation_no_rescale:
            save_dir += "/ablations"
            if ablation_no_normalization and ablation_no_rescale:
                save_dir += "/no_norm_no_rescale"
            elif ablation_no_normalization:
                save_dir += "/no_normalization"
            else:
                save_dir += "/no_rescale"
        
            
        dataset_names = ["lama","uni","lama_whu"]
        dataset_num = len(ctrl_code)
        files = [None]*dataset_num
        img_save_dir = [None]*dataset_num

        KL_save_path = os.path.join(root_dir, f"{save_dir}/KL.csv")
        create_dir(KL_save_path)
        KL_save_f = open(KL_save_path,"w")
        KL_save_f.write("Relation")
        preds_save_path = os.path.join(root_dir, f"{save_dir}/preds.pt")


        total_diff = [[],[],[]]
        # save important metrics
        output_save = { 
                        "LAMA": {"P":[], "P_d":[],"KL":[], "KL_d":[], "TT_CE":[], "TT_CE_d":[], "FF_E":[], "FF_E_d":[]},
                        "WIKI-UNI":{"P":[], "P_d":[],"KL":[], "KL_d":[], "TT_CE":[], "TT_CE_d":[], "FF_E":[], "FF_E_d":[]},
                        "LAMA-WHU":{"P":[], "P_d":[],"KL":[], "KL_d":[], "TT_CE":[], "TT_CE_d":[], "FF_E":[], "FF_E_d":[]}
                      }
 
        # save all preds for further analysis
        preds_save = {
            "LAMA": {},
            "WIKI-UNI": {},
            "LAMA-WHU":{},
        }

        for i in range(dataset_num):
            save_path = os.path.join(root_dir, f"{save_dir}/{dataset_names[i]}_difference_debias.csv") 
            create_dir(save_path)
            img_save_path =  os.path.join(root_dir, f"{save_dir}/img/{dataset_names[i]}")
            img_save_dir[i] = img_save_path
            if ctrl_code[i]==1:
                f = open(save_path,"w")
                f.write("relation,diff,origin,debias\n")
                files[i] = f

                KL_save_f.write(f",{dataset_names[i]}_before,{dataset_names[i]}_after")



        pbar = tqdm(total=len(self.relations)*sum(ctrl_code))
        
        
        # the first dataset evaluated in this exp 
        first_dataset_index = ctrl_code.index(1)

        # choose the right prompts
        templates = None
        if manual_prompt=="LAMA":
            templates = self.lama_template
        elif manual_prompt=="LPAQA":
            templates = self.LPAQA_template
        elif manual_prompt=="AutoPrompt":
            if self.model_name.startswith("roberta"):
                templates = self.AutoPrompt_tempalte_roberta
            elif self.model_name.startswith("bert"):
                templates = self.AutoPrompt_template_bert
            else:
                raise ValueError(f"an invalid param: {self.model_name}")
        else:
            raise ValueError(f"an invalid param: {manual_prompt}")

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
            # For the calucaltion of  KL divergence, we need transform the probs to a distriburion.
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
                    prompt=manual_prompt,
                    filter_biased_token_nums=filter_biased_token_nums,
                    ablation_no_normalization=ablation_no_normalization,
                    ablation_no_rescale=ablation_no_rescale,)
                acc_debias,(probs_after,_, preds_after, _) = self.manual_prompt(
                    relation, debias=True, 
                    vocab_subset_filter=vocab_subset_filter, 
                    vocab_subset=vocab_subset,
                    test_data_path=data_paths[i],
                    prompt=manual_prompt,
                    sampling_debias=sampling_debias,
                    calibrate=calibrate,
                    filter_biased_token_nums=filter_biased_token_nums,
                    ablation_no_normalization=ablation_no_normalization,
                    ablation_no_rescale=ablation_no_rescale,)
                
                # 处理数据集为空的情况
                if probs_before == None:
                    TT_CE_before = 0
                    TT_CE_after = 0
                    FF_E = 0
                    FF_E_d = 0
                    KL_before_list = 0
                    KL_after_list = 0
                    print("数据集为空")
                    pbar.update(1)
                    postfox = {
                        "lama_improve": 0 if ctrl_code[0]==0 or len(total_diff[0])==0 else round(sum(total_diff[0])/len(total_diff[0]), 3) * 100,
                        "uni_improve": 0 if ctrl_code[1]==0 or len(total_diff[1])==0 else round(sum(total_diff[1])/len(total_diff[1]), 3) * 100,
                        "lama_whu_improve": 0 if ctrl_code[2]==0 or len(total_diff[2])==0 else round(sum(total_diff[2])/len(total_diff[2]), 3) * 100,
                        }

                    pbar.set_postfix(postfox)
                    continue

                subvocab_labels = torch.tensor([subvocab_indices_list.index(l) for l in labels]).cuda()
                loss = torch.nn.CrossEntropyLoss()

                equality_mask = torch.eq(torch.tensor(preds_before), torch.tensor(labels)) & torch.eq(torch.tensor(preds_after), torch.tensor(labels))
                common_acc_index =  torch.where(equality_mask==True)[0]

                def cross_entropy(logits, labels):
                    loss_per_sample = F.cross_entropy(logits, labels, reduction='none')
                    loss_dict = {}
                    for label,loss in zip(labels.tolist(), loss_per_sample.tolist()):
                        if label in loss_dict:
                            loss_dict[label].append(loss)
                        else:
                            loss_dict[label] = [loss]
                    loss_list = []
                    for key in loss_dict.keys():
                        loss_list.append(np.mean(loss_dict[key]))
                    
                    return np.mean(loss_list)

                TT_CE_before = 0 if torch.numel(common_acc_index)==0 else cross_entropy(probs_before[common_acc_index], subvocab_labels[common_acc_index])
                TT_CE_after = 0 if torch.numel(common_acc_index)==0 else cross_entropy(probs_after[common_acc_index], subvocab_labels[common_acc_index])
                # TT_CE_before = cross_entropy(probs_before, subvocab_labels)
                # TT_CE_after = cross_entropy(probs_after, subvocab_labels)
                # print(TT_CE_before, TT_CE_after)
                # print(f"TT CE before: {TT_CE_before} TT CE after {TT_CE_after}")
                
                equality_mask = ~torch.eq(torch.tensor(preds_before), torch.tensor(labels)) & ~torch.eq(torch.tensor(preds_after), torch.tensor(labels))
                common_acc_index =  torch.where(equality_mask==True)[0]
                # 给定pred和labels, 计算得到分布

                def convert_to_probability(predictions):
                    total_predictions = len(predictions)
                    unique_values, counts = np.unique(predictions, return_counts=True)
                    probabilities = counts / total_predictions
                    return probabilities

                def entropy(probabilities):
                    entropy_value = -np.sum(probabilities * np.log2(probabilities))
                    return entropy_value

                FF_E = 0 if common_acc_index==[] else entropy(convert_to_probability(torch.tensor(preds_before)[common_acc_index].tolist()))
                FF_E_d = 0 if common_acc_index==[] else entropy(convert_to_probability(torch.tensor(preds_after)[common_acc_index].tolist()))
                # FF_E = entropy(convert_to_probability(preds_before))
                # FF_E_d = entropy(convert_to_probability(preds_after))
                
                # FF_CE_before = loss(probs_before[common_acc_index], subvocab_labels[common_acc_index]).item()
                # FF_CE_after = loss(probs_after[common_acc_index], subvocab_labels[common_acc_index]).item()
                # print(f"FF CE before: {FF_CE_before} FF CE after {FF_CE_after}")

                # for KL divergence and drawing picture, transform the probs to a distribution
                num_data = len(preds_before)
                # probs_before is actually logit values
                probs_dis_before = torch.softmax(probs_before,dim=-1).tolist()
                probs_dis_after = torch.softmax(probs_after,dim=-1).tolist()
                KL_before_list = []
                KL_after_list = []
                for index in range(num_data):
                    probs_dis_before_i = probs_dis_before[index]
                    probs_dis_after_i = probs_dis_after[index]
                    probs_dis_before_i = defaultdict(int, zip(subvocab_indices_list,probs_dis_before_i))
                    probs_dis_after_i = defaultdict(int, zip(subvocab_indices_list,probs_dis_after_i))
                    KL_before_list.append(self.KL_divergence(probs_dis_before_i, bias_dis))
                    KL_after_list.append(self.KL_divergence(probs_dis_after_i, bias_dis))

                KL_before = round(np.mean(KL_before_list),5)
                KL_after = round(np.mean(KL_after_list),5) 


                diff = round(acc_debias - acc_origin,5)
            
                print("{} raw acc{} debiased acc{}".format(
                    dataset_names[i],
                    round(acc_origin*100,2),
                    round(acc_debias*100,2)))
                total_diff[i].append(diff)
                f.write(f"{relation},{diff},{acc_origin},{acc_debias}\n")
                f.flush()


                dataset = list(output_save.keys())[i]
                output_save[dataset]["P"].append(acc_origin)
                output_save[dataset]["P_d"].append(acc_debias)

                output_save[dataset]["TT_CE"].append(TT_CE_before)
                output_save[dataset]["TT_CE_d"].append(TT_CE_after)

                output_save[dataset]["FF_E"].append(FF_E)
                output_save[dataset]["FF_E_d"].append(FF_E_d)

                output_save[dataset]["KL"].append(KL_before)
                output_save[dataset]["KL_d"].append(KL_after)

                # save all preds
                obj_labels = self.tokenizer.convert_ids_to_tokens(labels)
                raw_preds = self.tokenizer.convert_ids_to_tokens(preds_before)
                debiased_preds = self.tokenizer.convert_ids_to_tokens(preds_after)
                indices = list(range(len(sub_labels)))
                preds_save[dataset][relation] = {"data":{
                    "index":indices,
                    "sub_label":sub_labels,
                    "obj_labels":obj_labels,
                    "raw_preds":raw_preds,
                    "debiased_preds":debiased_preds,
                    "raw_probs":torch.tensor(probs_dis_before,dtype=torch.float16),
                    "debiased_probs": torch.tensor(probs_dis_after,dtype=torch.float16),
                    "label_subvocab_ids": subvocab_labels,
                    "label_ids": labels,
                    }}

                if i==first_dataset_index:
                    KL_save_f.write(f"\n{relation},{KL_before},{KL_after}")
                else:
                    KL_save_f.write(f",{KL_before},{KL_after}")

                # preds和labels均是原始词汇表里面的，绘图前需要转换成subset_vocab的索引
                preds_before = [subvocab_indices_list.index(i) for i in preds_before]
                preds_after = [subvocab_indices_list.index(i) for i in preds_after]
                labels = [subvocab_indices_list.index(i) for i in labels]
                img_save_path = os.path.join(img_save_dir[i], relation+".png")
                create_dir(img_save_path)

                self.generate_image(preds_before,preds_after,labels,model_bias=bias_tensor, save_path=img_save_path)
                
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

        # calculate metrics
        for i,dataset in enumerate(output_save.keys()):
            if ctrl_code[i]==0:
                continue
            avg_p = np.mean(output_save[dataset]["P"])
            avg_p_d = np.mean(output_save[dataset]["P_d"])
            avg_KL = np.mean(output_save[dataset]["KL"])
            avg_KL_d = np.mean(output_save[dataset]["KL_d"])
            avg_TT_CE = np.mean(output_save[dataset]["TT_CE"])
            avg_TT_CE_d = np.mean(output_save[dataset]["TT_CE_d"])
            avg_FF_E = np.mean(output_save[dataset]["FF_E"])
            avg_FF_E_d = np.mean(output_save[dataset]["FF_E_d"])
            
            self.add_output_item(self.model_name, dataset,prompt=manual_prompt,result=[avg_p,avg_p_d,avg_KL,avg_KL_d,avg_TT_CE,avg_TT_CE_d,avg_FF_E,avg_FF_E_d])
        
        self.print_output()

        torch.save(preds_save,preds_save_path)

        if embeddings_renormalize==True:
            output_embedding_backup = output_embedding_backup.to(self.plm.device)
            self.plm.set_output_embeddings(output_embedding_backup)
            self.plm.bert.embeddings.word_embeddings.weight = output_embedding_backup.weight


    def experiment_renormal_vector_debais_for_templama(self, vocab_subset_filter=True, vocab_subset="answer_type_tokens"):
        """An early debiasing experiment for the TEMPLAMA sub-dataset. 

        Args:
            vocab_subset_filter (bool, optional): whether to limit the output vocabulary space of the model. Defaults to True.
            vocab_subset (str, optional): A smaller vocab used to probe facts. Defaults to "answer_type_tokens".
        """
        root_dir = self.work_dir + f"/outputs/openprompt/manual_prompt/{self.model_name}"
        save_dir = f"TEMPLAMA/debias_{vocab_subset}/"
        dataset_names = ["templama"]
        dataset_num = 1
        files = [None]*dataset_num
        img_save_dir = [None]*dataset_num

        KL_save_path = os.path.join(root_dir, f"{save_dir}/KL.csv")
        create_dir(KL_save_path)
        KL_save_f = open(KL_save_path,"w")
        KL_save_f.write("Relation")
        preds_save_path = os.path.join(root_dir, f"{save_dir}/preds.pt")
        
        total_diff = [[]]
        # save metrics
        output_save = { 
                        "TEMPLAMA": {"P":[], "P_d":[],"KL":[], "KL_d":[]},
                      }
 
        # save all preds
        preds_save = {
            "TEMPLAMA": {},
        }

        for i in range(dataset_num):
            save_path = os.path.join(root_dir, f"{save_dir}/{dataset_names[i]}_difference_debias.csv") 
            create_dir(save_path)
            img_save_path =  os.path.join(root_dir, f"{save_dir}/img/{dataset_names[i]}")
            img_save_dir[i] = img_save_path
            f = open(save_path,"w")
            f.write("relation,diff,origin,debias\n")
            files[i] = f

            KL_save_f.write(f",{dataset_names[i]}_before,{dataset_names[i]}_after")

        # 统计tempLAMA中的relation
        lama_relations = self.relations
        self.relations = ["P39"]
        pbar = tqdm(total=len(self.relations))

        for relation in self.relations:
            templama_data_path = os.path.join(self.templama_data_dir,f"{relation}.jsonl")
            data_paths = [templama_data_path]

            raw_template = "In [year], [X] holds the position of [Y]."
            prompt_only_tempalte = raw_template.replace("[X]",self.tokenizer.mask_token).replace("[Y]", self.tokenizer.mask_token)
            
            subvocab_tokens_indices = self.get_answer_entity_indices(relation, self.tokenizer,include_templama_labels=True) \
                                                if vocab_subset == "answer_type_tokens" else \
                                    self.get_common_vocab_indices(self.tokenizer)
            subvocab_indices_list = subvocab_tokens_indices.tolist()
            
            # bias向量需要一一计算
            bias_distributions = []
            test_dataset = load_jsonl(os.path.join(self.templama_data_dir,f"{relation}.jsonl"))
            date_list = [i["date"] for i in test_dataset]
            for date in date_list:
                prompt_only_tempalte_with_year = prompt_only_tempalte.replace("[year]", date)
                bias_tensor = self.generate_model_bias(
                    self.plm, self.tokenizer, prompt_only_tempalte_with_year, 
                    vocab_subset_indices=subvocab_tokens_indices, 
                        return_logits=True) 
                
            
                bias_probs = torch.softmax(bias_tensor,dim=-1).tolist()
                # 用于计算KL散度
                bias_dis = defaultdict(int, zip(subvocab_indices_list,bias_probs))
                bias_distributions.append(bias_dis)

            for i in range(len(data_paths)):
                f = files[i]
                acc_origin,(probs_before, sub_labels, preds_before, labels) = self.manual_prompt_for_templama(
                    relation, debias=False, 
                    vocab_subset_filter=vocab_subset_filter,
                    vocab_subset=vocab_subset,)
                acc_debias,(probs_after,_, preds_after, _) = self.manual_prompt_for_templama(
                    relation, debias=True, 
                    vocab_subset_filter=vocab_subset_filter, 
                    vocab_subset=vocab_subset,)
                
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
                    KL_before_list.append(self.KL_divergence(probs_dis_before_i, bias_distributions[i]))
                    KL_after_list.append(self.KL_divergence(probs_dis_after_i, bias_distributions[i]))

                KL_before = round(np.mean(KL_before_list),5)
                KL_after = round(np.mean(KL_after_list),5) 

                output_save[dataset]["KL"].append(KL_before)
                output_save[dataset]["KL_d"].append(KL_after)

                KL_save_f.write(f"\n{relation},{KL_before},{KL_after}")
                

                # preds和labels均是原始词汇表里面的，绘图前需要转换成subset_vocab的索引
                preds_before = [subvocab_indices_list.index(i) for i in preds_before]
                preds_after = [subvocab_indices_list.index(i) for i in preds_after]

                labels = [subvocab_indices_list.index(i) for i in labels]

                img_save_path = os.path.join(img_save_dir[i], relation+".png")
                create_dir(img_save_path)

                self.generate_image(preds_before,preds_after,labels,model_bias=bias_tensor, save_path=img_save_path)


                diff = round(acc_debias - acc_origin,5)
                
                print("{} raw acc{} debiased acc{}".format(
                    dataset_names[i],
                    round(acc_origin*100,2),
                    round(acc_debias*100,2)))
                total_diff[i].append(diff)
                f.write(f"{relation},{diff},{acc_origin},{acc_debias}\n")
                f.flush()
                pbar.update(1)
                
        pbar.close()
        
        for f in files:
            if f != None:
                f.close()
        
        KL_save_f.close()

        # 处理输出
        for i,dataset in enumerate(output_save.keys()):
            avg_p = np.mean(output_save[dataset]["P"])
            avg_p_d = np.mean(output_save[dataset]["P_d"])
            avg_KL = np.mean(output_save[dataset]["KL"])
            avg_KL_d = np.mean(output_save[dataset]["KL_d"])
            self.add_output_item(self.model_name, dataset,prompt="manual_with_year",result=[avg_p,avg_p_d,avg_KL,avg_KL_d])
        
        self.print_output()

        torch.save(preds_save,preds_save_path)

        # recover states
        self.relations = lama_relations
        

    def experiment_renormal_vector_debias_for_continue_prompt(self, vocab_subset_filter=True, vocab_subset="answer_type_tokens", embeddings_renormalize=False,
        ctrl_code=[1,1,1], continue_prompt="optiprompt", num_tokens=5, repeat_times=3, evaluate_mode=False, filter_biased_token_nums=0,
        ablation_no_normalization=False,
        ablation_no_rescale=False):
        """Debiasing experiments for continue prompts.

        Args:
            vocab_subset_filter (bool, optional): whether to limit the output vocabulary space of the model. Defaults to True.
            vocab_subset (str, optional): A smaller vocab used to probe facts. Defaults to "answer_type_tokens".
            embeddings_renormalize (bool, optional): Whether to renormalize the embeddings of the model before probing. Defaults to False.
            ctrl_code (list, optional): which datasets are used in this exp. Defaults to [1,1,1], representing use all three datasets [lama, wiki-uni, lama_whu].
            continue_prompt (str, optional): what is the continue prompt used in this exp. Defaults to "optiprompt".
            num_tokens (int, optional): the number of the soft tokens in the continue prompt. Defaults to 5.
            repeat_times (int, optional): repeat this exp multiple times. Defaults to 3.
            evaluate_mode (bool, optional): Whether this func is run in evaluate mode. Defaults to False, means that 
                first train the prompt then evaluate it. If this param is `True`, then only evaluate.
        """

        pbar = tqdm(total=len(self.relations)*repeat_times)
        
        if embeddings_renormalize==True:
            config = self.model_config
            renormal = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            renormal.weight.data = self.plm.cls.predictions.decoder.weight.data.clone()
            renormal.weight.data = F.normalize(renormal.weight.data, p=2,dim=1)
            # renormal.bias = model.cls.predictions.decoder.bias
            output_embedding_backup = self.plm.get_output_embeddings()
            self.plm.set_output_embeddings(renormal)
            self.plm.bert.embeddings.word_embeddings.weight = renormal.weight

        # save states
        self_output_result_bk = copy.copy(self.output_result)

        # results for all exps
        output_results = []
        for _index in range(1,repeat_times+1):
            self.clear_output()
            root_dir = self.work_dir + f"/outputs/openprompt/continue_prompt/filter_out_{filter_biased_token_nums}_biased_tokens/{self.model_name}"
            
            save_dir = continue_prompt+f"_{num_tokens}" + f"/debias_{vocab_subset}/"
            if embeddings_renormalize==True:
                save_dir += "renormalize_embedding"
            else:
                save_dir += "origin_embedding"
            
            # additional output path for ablation experiments
            if ablation_no_normalization or ablation_no_rescale:
                save_dir += "/ablations"
                if ablation_no_normalization and ablation_no_rescale:
                    save_dir += "/no_norm_no_rescale"
                elif ablation_no_normalization:
                    save_dir += "/no_normalization"
                else:
                    save_dir += "/no_rescale"

            save_dir += f"/exp_{_index}"
            embedding_save_dir = os.path.join(root_dir,save_dir)

            dataset_names = ["lama","uni","lama_whu"]
            dataset_num = len(ctrl_code)
            files = [None]*dataset_num
            img_save_dir = [None]*dataset_num

            KL_save_path = os.path.join(root_dir,f"{save_dir}/KL.csv")
            create_dir(KL_save_path)
            KL_save_f = open(KL_save_path,"w")
            KL_save_f.write("Relation")
            preds_save_path = os.path.join(root_dir, f"{save_dir}/preds.pt")

            total_diff = [[],[],[]]
            # svae metrics
            output_save = { 
                            "LAMA": {"P":[], "P_d":[],"KL":[], "KL_d":[], "TT_CE":[], "TT_CE_d":[], "FF_E":[], "FF_E_d":[]},
                            "WIKI-UNI":{"P":[], "P_d":[],"KL":[], "KL_d":[], "TT_CE":[], "TT_CE_d":[], "FF_E":[], "FF_E_d":[]},
                            "LAMA-WHU":{"P":[], "P_d":[],"KL":[], "KL_d":[], "TT_CE":[], "TT_CE_d":[], "FF_E":[], "FF_E_d":[]}
                        }

            # svae all predicts
            preds_save = {
                "LAMA": {},
                "WIKI-UNI": {},
                "LAMA-WHU":{},
            }
            
            for i in range(dataset_num):
                save_path = os.path.join(root_dir, f"{save_dir}/{dataset_names[i]}_difference_debias.csv")
                create_dir(save_path)
                img_save_path = os.path.join(root_dir, f"{save_dir}/img/{dataset_names[i]}")
                img_save_dir[i] = img_save_path
                if ctrl_code[i]==1:
                    f = open(save_path,"w")
                    f.write("relation,diff,origin,debias\n")
                    files[i] = f

                    KL_save_f.write(f",{dataset_names[i]}_before,{dataset_names[i]}_after")

            first_dataset_index = ctrl_code.index(1)

            for relation in self.relations:
                
                model_bias, results = self.continue_prompt(relation,
                                                            embedding_save_dir=embedding_save_dir,
                                                            vocab_subset_filter=vocab_subset_filter,
                                                            vocab_subset=vocab_subset,
                                                            continue_prompt=continue_prompt,
                                                            num_tokens=num_tokens,
                                                            evaluate_mode=evaluate_mode, 
                                                            filter_biased_token_nums=filter_biased_token_nums,
                                                            ablation_no_normalization=ablation_no_normalization,
                                                            ablation_no_rescale=ablation_no_rescale,
                                                            )
                
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
                    if probs_before == None:
                        TT_CE_before = 0
                        TT_CE_after = 0
                        FF_E = 0
                        FF_E_d = 0
                        KL_before_list = 0
                        KL_after_list = 0
                        print("数据集为空")
                        postfox = {
                            "lama_improve": 0 if ctrl_code[0]==0 or len(total_diff[0])==0 else round(sum(total_diff[0])/len(total_diff[0]), 3) * 100,
                            "uni_improve": 0 if ctrl_code[1]==0 or len(total_diff[1])==0 else round(sum(total_diff[1])/len(total_diff[1]), 3) * 100,
                            "lama_whu_improve": 0 if ctrl_code[2]==0 or len(total_diff[2])==0 else round(sum(total_diff[2])/len(total_diff[2]), 3) * 100,
                            }

                        pbar.set_postfix(postfox)
                        continue
                    # 添加TT_CE 和FF_E
                    subvocab_labels = torch.tensor([subvocab_indices_list.index(l) for l in labels]).cuda()
                    loss = torch.nn.CrossEntropyLoss()

                    equality_mask = torch.eq(torch.tensor(preds_before), torch.tensor(labels)) & torch.eq(torch.tensor(preds_after), torch.tensor(labels))
                    common_acc_index =  torch.where(equality_mask==True)[0]
                    TT_CE_before = 0 if torch.numel(common_acc_index)==0 else loss(probs_before[common_acc_index], subvocab_labels[common_acc_index]).item()
                    TT_CE_after = 0 if torch.numel(common_acc_index)==0 else loss(probs_after[common_acc_index], subvocab_labels[common_acc_index]).item()
                    equality_mask = ~torch.eq(torch.tensor(preds_before), torch.tensor(labels)) & ~torch.eq(torch.tensor(preds_after), torch.tensor(labels))
                    common_acc_index =  torch.where(equality_mask==True)[0]
                    # 给定pred和labels, 计算得到分布

                    def convert_to_probability(predictions):
                        total_predictions = len(predictions)
                        unique_values, counts = np.unique(predictions, return_counts=True)
                        probabilities = counts / total_predictions
                        return probabilities

                    def entropy(probabilities):
                        entropy_value = -np.sum(probabilities * np.log2(probabilities))
                        return entropy_value

                    FF_E = 0 if common_acc_index==[] else entropy(convert_to_probability(torch.tensor(preds_before)[common_acc_index].tolist()))
                    FF_E_d = 0 if common_acc_index==[] else entropy(convert_to_probability(torch.tensor(preds_after)[common_acc_index].tolist()))
                    # FF_E = entropy(convert_to_probability(preds_before))
                    # FF_E_d = entropy(convert_to_probability(preds_after))

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
                        KL_before_list.append(self.KL_divergence(probs_dis_before_i, bias_dis))
                        KL_after_list.append(self.KL_divergence(probs_dis_after_i, bias_dis))

                    KL_before = round(np.mean(KL_before_list),5)
                    KL_after = round(np.mean(KL_after_list),5) 
                    # print(f"debais前KL散度为{KL_before} debias后KL散度为{KL_after}")



                    diff = round(acc_debias - acc_origin,5)
                    
                    print("{} raw acc{} debiased acc{}".format(
                        dataset_names[i],
                        round(acc_origin*100,2),
                        round(acc_debias*100,2)))
                    total_diff[i].append(diff)
                    f.write(f"{relation},{diff},{acc_origin},{acc_debias}\n")
                    f.flush()

                    dataset = list(output_save.keys())[i]
                    output_save[dataset]["P"].append(acc_origin)
                    output_save[dataset]["P_d"].append(acc_debias)
                    output_save[dataset]["KL"].append(KL_before)
                    output_save[dataset]["KL_d"].append(KL_after)
                    output_save[dataset]["TT_CE"].append(TT_CE_before)
                    output_save[dataset]["TT_CE_d"].append(TT_CE_after)
                    output_save[dataset]["FF_E"].append(FF_E)
                    output_save[dataset]["FF_E_d"].append(FF_E_d)

                    # 保存preds结果
                    obj_labels = self.tokenizer.convert_ids_to_tokens(labels)
                    raw_preds = self.tokenizer.convert_ids_to_tokens(preds_before)
                    debiased_preds = self.tokenizer.convert_ids_to_tokens(preds_after)
                    indices = list(range(len(sub_labels)))
                    preds_save[dataset][relation] = {"data":{"index":indices,"sub_label":sub_labels,"obj_labels":obj_labels,"raw_preds":raw_preds,"debiased_preds":debiased_preds}}

                    if i==first_dataset_index:
                        KL_save_f.write(f"\n{relation},{KL_before},{KL_after}")
                    else:
                        KL_save_f.write(f",{KL_before},{KL_after}")

                    # preds和labels均是原始词汇表里面的，绘图前需要转换成subset_vocab的索引
                    preds_before = [subvocab_indices_list.index(i) for i in preds_before]
                    preds_after = [subvocab_indices_list.index(i) for i in preds_after]
                    labels = [subvocab_indices_list.index(i) for i in labels]
                    img_save_path = os.path.join(img_save_dir[i], relation+".png")
                    create_dir(img_save_path)

                    self.generate_image(preds_before,preds_after,labels,model_bias=bias_tensor, save_path=img_save_path)
                    
                    
                    postfox = {
                        "run nums": _index,
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

            # add outputs of this time
            for i,dataset in enumerate(output_save.keys()):
                if ctrl_code[i]==0:
                    continue
                avg_p = np.mean(output_save[dataset]["P"])
                avg_p_d = np.mean(output_save[dataset]["P_d"])
                avg_KL = np.mean(output_save[dataset]["KL"])
                avg_KL_d = np.mean(output_save[dataset]["KL_d"])
                avg_TT_CE = np.mean(output_save[dataset]["TT_CE"])
                avg_TT_CE_d = np.mean(output_save[dataset]["TT_CE_d"])
                avg_FF_E = np.mean(output_save[dataset]["FF_E"])
                avg_FF_E_d = np.mean(output_save[dataset]["FF_E_d"])

                self.add_output_item(self.model_name, dataset,prompt=continue_prompt+'_'+str(num_tokens),result=[avg_p,avg_p_d,avg_KL,avg_KL_d,avg_TT_CE,avg_TT_CE_d,avg_FF_E,avg_FF_E_d])
            
            self.save_output(os.path.join(root_dir, save_dir+"/result.json"))
            self.print_output()
            temp = copy.copy(self.output_result)
            output_results.append(temp)


        # recover states
        self.clear_output()
        self.output_result = self_output_result_bk

        # calcuate the avg results on all exps
        for i,dataset in enumerate(output_save.keys()):
            if ctrl_code[i]==0:
                continue
            P = [result[self.model_name][dataset][continue_prompt+'_'+str(num_tokens)]["P"] for result in output_results]
            P_d = [result[self.model_name][dataset][continue_prompt+'_'+str(num_tokens)]["P_d"] for result in output_results]
            KL = [result[self.model_name][dataset][continue_prompt+'_'+str(num_tokens)]["KL"] for result in output_results]
            KL_d = [result[self.model_name][dataset][continue_prompt+'_'+str(num_tokens)]["KL_d"] for result in output_results]
            TT_CE = [result[self.model_name][dataset][continue_prompt+'_'+str(num_tokens)]["TT_CE"] for result in output_results]
            TT_CE_d = [result[self.model_name][dataset][continue_prompt+'_'+str(num_tokens)]["TT_CE_d"] for result in output_results]
            FF_E = [result[self.model_name][dataset][continue_prompt+'_'+str(num_tokens)]["FF_E"] for result in output_results]
            FF_E_d = [result[self.model_name][dataset][continue_prompt+'_'+str(num_tokens)]["FF_E_d"] for result in output_results]
            
            avg_p = np.mean(P)
            avg_p_d = np.mean(P_d)
            avg_KL = np.mean(KL)
            avg_KL_d = np.mean(KL_d)
            avg_TT_CE = np.mean(TT_CE)
            avg_TT_CE_d = np.mean(TT_CE_d)
            avg_FF_E = np.mean(FF_E)
            avg_FF_E_d = np.mean(FF_E_d)

            self.add_output_item(self.model_name, dataset,prompt=continue_prompt+'_'+str(num_tokens),result=[avg_p,avg_p_d,avg_KL,avg_KL_d,avg_TT_CE,avg_TT_CE_d,avg_FF_E,avg_FF_E_d])

        torch.save(preds_save,preds_save_path)

        # recover states
        if embeddings_renormalize==True:
            output_embedding_backup = output_embedding_backup.to(self.plm.device)
            self.plm.set_output_embeddings(output_embedding_backup)
            self.plm.bert.embeddings.word_embeddings.weight = output_embedding_backup.weight

        pbar.close()


    def quantify_prompt_bias(self, vocab_subset_filter=True, vocab_subset="answer_type_tokens", prompt="LAMA", meanless_word="[MASK]", sampling=False):
        """Quantify the prompt bias with KL divergence and support set.
        
        This func will quantify the specific prompt bias of in current Model.

        Args:
            vocab_subset_filter (bool, optional): whether to limit the output vocabulary space of the model. Defaults to True.
            vocab_subset (str, optional): A smaller vocab used to probe facts, it is the label space that we quantify the prompt bias. Defaults to "answer_type_tokens".
            prompt (str, optional): The prompt used to probe facts. Can be "LAMA", "LPAQA", "AutoPrompt", "Optiprompt".
        
        Returns:
            A tuple of (avg_raw_KL, avg_debias_KL, KL), where the KL denates to the detailed KL values for all relations.
        """

        # prepare the prompts
        templates = None
        if prompt=="LAMA":
            templates = self.lama_template
        elif prompt=="LPAQA":
            templates = self.LPAQA_template
        elif prompt=="AutoPrompt":
            if self.model_name.startswith("roberta"):
                templates = self.AutoPrompt_tempalte_roberta
            elif self.model_name.startswith("bert"):
                templates = self.AutoPrompt_template_bert
            else:
                raise ValueError(f"an invalid param: {self.model_name}")
        elif prompt=="OptiPrompt":
            # TODO 还需要进一步实现
            templates = ""
            pass
        else:
            raise ValueError(f"an invalid param: {prompt}")
        
        KL= {}
        print(f"There are {len(self.relations)} relations needing to quantify prompt bias.")
        for relation in self.relations:
            # prepare raw WIKI-UNI
            uni_data_path = os.path.join(self.wiki_uni_data_dir,f"{relation}.json")
            raw_template =templates[relation]
            if meanless_word=="[MASK]":
                prompt_only_tempalte = raw_template.replace("[X]",self.tokenizer.mask_token).replace("[Y]", self.tokenizer.mask_token)
            elif meanless_word=="N/A":
                prompt_only_tempalte = raw_template.replace("[X]", "N/A").replace("[Y]", self.tokenizer.mask_token)
            elif meanless_word=="":
                prompt_only_tempalte = raw_template.replace("[X]", "").replace("[Y]", self.tokenizer.mask_token)

            
            subvocab_tokens_indices = self.get_answer_entity_indices(relation, self.tokenizer) \
                                                if vocab_subset == "answer_type_tokens" else \
                                    self.get_common_vocab_indices(self.tokenizer)
            subvocab_indices_list = subvocab_tokens_indices.tolist()

            # wrap dataset
            dataset = {}
            first_token = raw_template.split()[0]
            autoprompt_postfix = first_token[3:] if prompt=="AutoPrompt" else ''
            raw_test_data, _ = self.load_data(uni_data_path, common_vocab=self.common_vocab)
            # if len(raw_test_data) < self.support_dataset:
            #     print("样本不足, 跳过该relation")
            #     continue
            dataset["test"] = self.wrap_input_examples(raw_test_data, self.tokenizer,autoprompt_postfix=autoprompt_postfix)

            # transform to the template with OpenPrompt's format.
            template = raw_template.replace(first_token,'{"placeholder":"text_a"}') if prompt=="AutoPrompt" else \
                        raw_template.replace("[X]", '{"placeholder":"text_a"}')
            template = template.replace("[Y]", '{"mask"}')
            prompt_template = ManualTemplate(tokenizer=self.tokenizer, text=template)

            # support dataset for sampling
            support_sampler = FewShotSampler(num_examples_total=self.support_dataset, also_sample_dev=False)
            dataset['support'] = support_sampler(dataset['test'], seed=6)
            support_dataloader = PromptDataLoader(dataset=dataset["support"], template=prompt_template, tokenizer=self.tokenizer,
                tokenizer_wrapper_class=self.WrapperClass, max_seq_length=512,
                batch_size=16, shuffle=False)
            # bias_logits, bias_vector = self.get_prompt_bias_by_sample(promptModel, support_dataloader)

            # prepare the model
            promptModel = PromptModel(
            template = prompt_template,
            plm = self.plm,
            freeze_plm=True
            )

            promptModel.cuda()
            
            # 打印测试精度
            # test_dataloader = PromptDataLoader(dataset=dataset["support"], template=prompt_template, tokenizer=self.tokenizer,
            #     tokenizer_wrapper_class=self.WrapperClass, max_seq_length=512,
            #     batch_size=16, shuffle=False)

            # alllabels = []
            # allpreds = []
            # subvocab_tokens_indices = subvocab_tokens_indices.cuda()
            # for inputs in test_dataloader:
            #     inputs = inputs.cuda()
            #     logits = promptModel(inputs).logits
            #     mask_logits = logits[torch.where(inputs['loss_ids']>0)]
            #     if subvocab_tokens_indices != None:
            #         mask_logits = mask_logits.index_select(dim=1,index=subvocab_tokens_indices)
                
            #     labels = inputs["label"]
            #     alllabels += labels.cpu().tolist()
            #     predict_index = torch.argmax(mask_logits, dim=-1)
            #     if subvocab_tokens_indices != None:
            #         predict_index = subvocab_tokens_indices[predict_index]
            #     allpreds.extend(predict_index.cpu().tolist())
        
            # acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
            # print(acc)
            # exit()


            if not sampling:
                bias_logits, bias_vector = self.get_manual_prompt_bias(self.plm, self.tokenizer, template=prompt_only_tempalte)
            else:
                bias_logits, bias_vector = self.get_prompt_bias_by_sample(promptModel, support_dataloader)


            # 得到一个平均概率值
            avg_probs,avg_probs_debias = self.get_probs_by_sample(promptModel, support_dataloader, subvocab_indices_list, bias_vector=bias_vector)
            uniform_probs = torch.ones(avg_probs.shape)/avg_probs.shape[0]
            avg_probs, avg_probs_debias, uniform_probs = avg_probs.tolist(), avg_probs_debias.tolist(), uniform_probs.tolist()

            # 计算和均匀分布之间的距离
            raw_KL = self.KL_divergence(avg_probs, uniform_probs)
            debiased_KL = self.KL_divergence(avg_probs_debias, uniform_probs)
            # print(f"raw_KL: {raw_KL}, debiased_KL {debiased_KL}")
            #  dataset
            KL[relation] = {"raw_KL":raw_KL, "debiased_KL":debiased_KL}
            print("Finish relation {}, raw_KL:{}, debiased_KL:{}".format(relation, round(raw_KL,5), round(debiased_KL,5)))

        avg_raw_KL = np.mean([value["raw_KL"] for value in KL.values()])
        avg_debias_KL = np.mean([value["debiased_KL"] for value in KL.values()])

        return avg_raw_KL, avg_debias_KL, KL  


    def wrap_input_examples(self, raw_dataset:list, tokenizer:PreTrainedTokenizer, autoprompt_postfix=""):
        """wrap the raw dataset into an OpenPrompt need fromat.

        Args:
            raw_dataset (list): the raw dataset
            tokenizer (PreTrainedTokenizer): the tokenizer of the model
            autoprompt_postfix (str, optional): a speical param designed for autoprompt. Defaults to "".

        Returns:
            list: the wraped dataset
        """
        wrapped = []
        for data in raw_dataset:
            # 加入空格，使得roberta生效
            obj_token = tokenizer.tokenize(' ' + data["obj_label"])[0]
            input_example = InputExample(text_a=data["sub_label"]+autoprompt_postfix,label=tokenizer.convert_tokens_to_ids(obj_token))
            wrapped.append(input_example)

        return wrapped


    def compute_max_pad(self, dataset, WrapperClass, tokenizer, OpenPromptTemplate:Template):
        """calculate the need max pad number for the dataset
        """
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


    def evaluate(self, promptModel:PromptModel, data_loader:PromptDataLoader,
                bias_logits=None, bias_vector=None, 
                vocab_subset_indices=None,
                soft_tempalte_ckpt=None, 
                continue_prompt="prefix",num_tokens=5,
                calibrate=False, calib_logits=None,
                ablation_no_normalization=False,
                ablation_no_rescale=False,):
        """Evaluate the raw or debiased acc in a dataset.

        Args:
            promptModel (PromptModel): an Openprompt wrapped model
            data_loader (PromptDataLoader): the evaluation dataset loader 
            bias_logits (tensor, optional): the normalized bias vector. Defaults to None.
            bias_vector (tensor, optional): the raw bias vector.. Defaults to None.
            vocab_subset_indices (tensor, optional): filter the model bias distribution on the specified vocab subset. 
                Defaults to None.
            soft_tempalte_ckpt (str, optional): the saved weights of the continue prompt. Defaults to None.
            continue_prompt (str, optional): type of the continue prompt to use. Defaults to "prefix".
            num_tokens (int, optional): the number of the soft tokens in the continue prompt. Defaults to 5.
            calibrate (bool, optional): use calibrate rather our debiasing method to mitigate prompt. Defaults to False.
            calib_logits (tensor, optional): the calibrate logits. Defaults to None.

        Returns:
            tuple: a tuple of (acc, (probs, allpreds, alllabels))
        """             
        
        promptModel.eval()

        # The following is an unified framework splitting the process of language model into 3 parts
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

        # calibrate 需要对verbalizer操作一下
        if calibrate:
            if isinstance(promptModel, PromptForClassification):
                promptModel.verbalizer.register_calibrate_logits(calib_logits)
                

        for step, inputs in enumerate(data_loader):
            inputs =  inputs.cuda()
            with torch.no_grad():
                if bias_logits!=None:
                    if isinstance(promptModel, PromptModel):
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


                        if calibrate:
                            mask_logits = decoder(prediction_vectors)
                            assert len(mask_logits.shape)==2
                            if vocab_subset_indices != None:
                                mask_logits = mask_logits.index_select(dim=1,index=vocab_subset_indices)
                            # 手动实现calibrate
                            calib_logits = calib_logits.to(mask_logits.device)
                            calib_logits_answer_space = calib_logits.index_select(dim=0,index=vocab_subset_indices)
                            calib_probs = torch.softmax(calib_logits_answer_space,dim=0)
                            mask_probs = torch.softmax(mask_logits, dim=1)
                            mask_probs /= calib_probs
                            # normalize
                            mask_probs = (mask_probs.T / torch.sum(mask_probs, dim=1)).T
                            mask_logits = torch.log(mask_probs)
                        else:
                            batched_debiased_logits = None
                            D_T = bias_vector / torch.norm(bias_vector) if not ablation_no_normalization else bias_vector
                            for i in range(mask_token_features.shape[0]):
                                D_Tx = mask_token_features[i] / torch.norm(mask_token_features[i]) if not ablation_no_normalization else mask_token_features[i]
                                D_debias = (D_Tx - D_T) / torch.norm(D_Tx - D_T) if not ablation_no_normalization else D_Tx - D_T
                                V_debias = D_debias * torch.norm(mask_token_features[i]) if not ablation_no_rescale else D_debias
                                debiased_logits = decoder(V_debias).unsqueeze(dim=0)
                                if batched_debiased_logits == None:
                                    batched_debiased_logits = debiased_logits
                                else:
                                    batched_debiased_logits = torch.cat((batched_debiased_logits,debiased_logits),dim=0)
                            
                            mask_logits = batched_debiased_logits
                            if vocab_subset_indices != None:
                                mask_logits = mask_logits.index_select(dim=1,index=vocab_subset_indices)
                            
                    elif isinstance(promptModel, PromptForClassification):
                        # 一定有verbalizer 來做分类
                        assert promptModel.verbalizer != None
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
                        batched_debiased_logits = None
                        D_T = bias_vector / torch.norm(bias_vector) if not ablation_no_normalization else bias_vector
                        for i in range(mask_token_features.shape[0]):
                            D_Tx = mask_token_features[i] / torch.norm(mask_token_features[i]) if not ablation_no_normalization else mask_token_features[i]
                            D_debias = (D_Tx - D_T) / torch.norm(D_Tx - D_T) if not ablation_no_normalization else D_Tx - D_T
                            V_debias = D_debias * torch.norm(mask_token_features[i]) if not ablation_no_rescale else D_debias
                            debiased_logits = decoder(V_debias).unsqueeze(dim=0)
                            if batched_debiased_logits == None:
                                batched_debiased_logits = debiased_logits
                            else:
                                batched_debiased_logits = torch.cat((batched_debiased_logits,debiased_logits),dim=0)
                        
                        mask_logits = batched_debiased_logits
                        
                        # 对于分类任务来说，不需要typed quering，因为verbalizer已经限定了answer space
                        
                        # 从forward逻辑中中抄过来的
                        mask_logits =  promptModel.verbalizer.process_outputs(mask_logits, batch=inputs)

                else:
                    if isinstance(promptModel, PromptForClassification):
                        mask_logits = promptModel(inputs)
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
       
        # calibrate 副作用需要手动恢复
        if calibrate and isinstance(promptModel, PromptForClassification):
            promptModel.verbalizer._calibrate_logits = None
        # allpreds and alllabels are indice of the raw dataset
        return acc, (probs, allpreds, alllabels)


    def KL_divergence(self, dis_a, dis_b):
        """calculate the KL divergence between two distributions

        Args:
            dis_a ([defaultdict, list]): distribution a. The true obversed distribution.
            dis_b ([defaultdict, list]): distribution b. The theoretical distribution.

        Returns:
            float: the KL divergence
        """
        assert .0 not in dis_b and 0 not in dis_b, "distribution b should not containing zero!"
        if isinstance(dis_a, defaultdict):
            keys = dis_a.keys()
            KL = 0.0
            for key in keys:
                if dis_a[key]==0:
                    continue
                KL += dis_a[key] * log(dis_a[key] / dis_b[key])
        elif isinstance(dis_a, list):
            KL = 0.0
            for i in range(len(dis_a)):
                if dis_a[i]==0:
                    continue
                KL += dis_a[i] * log(dis_a[i] / dis_b[i])
        
        return KL


    def preprocess_tempLAMA(self):
        """get a subset of raw tempLAMA dataset
        """
        # apply some filters to the raw dataset to generate the subset used in our exp
        # filter out all samples with single token output
        data_dir = "/workspace/data/users/xuziyang/code/PromptBias/TruelyKnow/data/"
        test_dataset = load_jsonl(data_dir+"TEMPLAMA/filtered_test.json")    


        def in_vocab_single_token(x):
            label = x["answer"][0]["name"]
            return label in exp.common_vocab
        
        test_dataset = list(filter(in_vocab_single_token, test_dataset))

        def before_2019(x):
            return int(x["date"]) < 2019
        
        test_dataset = list(filter(before_2019, test_dataset))

        # split the dataset into different parts according to the relations
        relations = [item["relation"] for item in test_dataset]
        relations = set(relations)
        
        for relation in relations:
            def with_relation(x):
                return x["relation"]==relation
            filtered_data = list(filter(with_relation, test_dataset))
            # save it
            with open(f"/workspace/data/users/xuziyang/code/PromptBias/TruelyKnow/data/TEMPLAMA/{relation}.jsonl", 'w') as f:
                for i in filtered_data:
                    line = json.dumps(i)
                    # print(line)
                    f.write(line)
                    f.write("\n")
            print(f"relation {relation}: {len(filtered_data)}")


if __name__ == '__main__':
    # An example
    exp = Experiment()
    exp.set_model("bert","bert-base-cased")
    exp.set_common_vocab(exp.work_dir + "/common_vocabs/common_vocab_cased.txt")
    
    exp.experiment_renormal_vector_debais_for_manual_prompt(calibrate=False,)

    # exp.load_output("/mnt/code/users/xuziyang/PromptBias/results/filter_out_4_biased_tokens/bert-base-cased/common_vocab_cased/typed_querying/LAMA_result.json")
    # exp.print_output()
    # out = exp.manual_prompt("P463", debias=True, vocab_subset="answer_type_tokens", filter_biased_token_nums=1)
    # exp.relations = ["P413"]
    # out = exp.experiment_prompt_only_accuracy(prompt="OptiPrompt")
    # print(out)