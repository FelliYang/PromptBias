"""
使用openprompt 来重构整个代码
"""
from transformers import AutoTokenizer,AutoModelForMaskedLM
import torch
import json
import os
from utils import load_vocab,load_jsonl,load_json
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt import PromptModel,PromptDataLoader,PromptForClassification,PromptForGeneration
from openprompt.prompts import ManualTemplate
import numpy as np

class Experiment():
    def __init__(self) -> None:
        self.lama_data_dir = "/home/jiao/code/prompt/OptiPrompt/data/LAMA-TREx"
        self.auto_prompt_dir = "/home/jiao/code/prompt/OptiPrompt/data/autoprompt_data"
        self.lama_whu_data_dir = "/home/jiao/code/prompt/OptiPrompt/data/LAMA-TREx_UHN"
        self.wiki_uni_data_dir = "/home/jiao/code/prompt/OptiPrompt/data/wiki_uni"
        self.common_vocab_path = "/home/jiao/code/prompt/OptiPrompt/common_vocabs/common_vocab_cased.txt"
        self.lama_template_path = "/home/jiao/code/prompt/OptiPrompt/relation_metainfo/LAMA_relations.jsonl"
        # 导入LAMA manual prompt
        data = load_jsonl(self.lama_template_path)
        self.lama_template = dict([(item["relation"], item["template"]) for item in data])

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
                    return data
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
            def filter_out_repeat(item):
                t = (item["sub_label"],item["obj_label"])
                if t in sub_obj_set:
                    return False
                else:
                    sub_obj_set.add(t)
                    return True
            data = list(filter(filter_out_repeat, data))
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


    def generate_model_bias(self, model, tokenizer:AutoTokenizer, prompt_only, save_path="model_bias/bert/manual_prompt/P19.json"):
        """
        该函数目的是针对某个prompt作用下的model，得到一个bias的分布表
        prompt_only形式为 [MASK] was born in [MASK] . 这种形式
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

        model_output = model(**model_input)
        distribution = model_output[0][0][y_mask_index]
        probs = torch.softmax(distribution,0).tolist()
        index = list(range(len(probs)))
        tokens = tokenizer.convert_ids_to_tokens(index)
        output = dict(zip(index,zip(probs,tokens)))

        self.create_dir(save_path)
        with open(save_path, "w") as f:
            # f.write("data format: vocab_index:[prob, vocab]\n")
            output = json.dump(output,f,indent=4)

    def manual_prompt(self, relation):
        # 获取预训练模型
        plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")

        # 构造数据集        
        dataset = {}
        dataset["test"] = []
        lama_data_path = os.path.join(self.lama_data_dir,f"{relation}.jsonl")
        # lama_data_path = os.path.join(self.auto_prompt_dir,relation,"test.jsonl")
        lama_vocab_subset = load_vocab(self.common_vocab_path)
        raw_test_data, _ = self.load_data(lama_data_path, vocab_subset=lama_vocab_subset,)
        for data in raw_test_data:
            input_example = InputExample(text_a=data["sub_label"],label=tokenizer.convert_tokens_to_ids(data["obj_label"]))
            dataset["test"].append(input_example)

        # 定义手动模板
        raw_template = self.lama_template[relation]
        # 将模板转换成openprompt
        template = raw_template.replace("[X]",'{"placeholder":"text_a"}')
        template = template.replace("[Y]", '{"mask"}')
        prompt_template = ManualTemplate(tokenizer=tokenizer, text=template)


        # 加载数据集
        # 计算lama需要的最大pad数量
        max_tokens_len = 0
        wrapped_tokenizer = WrapperClass(tokenizer=tokenizer,max_seq_length=100)
        for data in dataset["test"]:
            wrapped_example = prompt_template.wrap_one_example(data)
            token_list = wrapped_tokenizer.tokenize_one_example(wrapped_example,teacher_forcing=False)
            current_len = sum(token_list["attention_mask"])
            if current_len > max_tokens_len:
                max_tokens_len =current_len

        assert max_tokens_len < 100

        test_dataloader = PromptDataLoader(dataset=dataset["test"], template=prompt_template, tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass, max_seq_length=100, 
            batch_size=16,shuffle=False)
        
        # 准备prompt Model
        promptModel = PromptModel(
            template = prompt_template,
            plm = plm,
            freeze_plm=True
        )
        promptModel.eval()
        promptModel.cuda()

        # 开始预测
        allpreds = []
        alllabels = []
        for inputs in test_dataloader:
            inputs = inputs.cuda()
            mask_logits = promptModel(inputs).logits
            mask_logits = mask_logits[torch.where(inputs['loss_ids']>0)]
            mask_logits = mask_logits.view(inputs['loss_ids'].shape[0], -1, mask_logits.shape[1])
            labels  = inputs["label"]
            if mask_logits.shape[1] == 1:
                mask_logits = mask_logits.view(mask_logits.shape[0], mask_logits.shape[2])
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(mask_logits, dim=-1).cpu().tolist())
        
        acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
        print("precision: {}".format(acc))

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
            mask_token = output[0][-3]
            index = torch.argmax(mask_token).item()
            if index == tokenizer.convert_tokens_to_ids(data["obj_label"]):
                correct += 1
            total += 1
        
        print(correct/total)
            
        


        

exp = Experiment()


# b, _ = exp.load_data(os.path.join(exp.lama_data_dir,"P19.jsonl"))
# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# model = AutoModelForMaskedLM.from_pretrained("bert-base-cased")
# prompt_only = "[MASK] was born in [MASK] ."
# exp.generate_model_bias(model,tokenizer,prompt_only,save_path="model_bias/bert/manual_prompt/P19.json")