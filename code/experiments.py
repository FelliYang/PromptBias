import argparse
import sys
import json
import logging
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
from collections import defaultdict
from transformers import (AdamW, BertForMaskedLM,
                          get_linear_schedule_with_warmup)

from models import Prober
from utils import *


parser = argparse.ArgumentParser()

#for model
parser.add_argument('--model_name', type=str, default='bert-base-cased', help='the huggingface model name')
parser.add_argument('--model_dir', type=str, default=None, help='the model directory (if not using --model_name)')
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--k', type=int, default=5, help='how many predictions will be outputted')
# for template
parser.add_argument('--init_manual_template', action='store_true', help='whether to use manual template to initialize the dense vectors')
parser.add_argument('--num_vectors', type=int, default=5, help='how many dense vectors are used in OptiPrompt')

args = parser.parse_args()

set_seed(args.seed)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.WARNING)
logger = logging.getLogger(__name__)




class Experiment():
    def __init__(self) -> None:
        """
        sub_reduce: whether control the subject space of max class to the sum of space of other classes
        obj_resample: whether resample the data to make the obj balanced in training data
        """
        self.relations = []
        self.output_dir = "outputs/experiment/"
        self.data_dir = "/home/jiao/code/prompt/OptiPrompt/data/autoprompt_data"
        self.common_vocab_path = "common_vocabs/common_vocab_cased.txt"
        self.train_batchsize =16
        self.valid_batchsize = 8
        self.num_epochs = 10
        self.learning_rate = 3e-3
        self.warmup_proportion = 0.1
        self.sub_reduce = False
        self.obj_resample = True
        self.augment_train_data = False
        self.noise_std = 0.1
        self.bert_vocab = list(Prober(args).tokenizer.ids_to_tokens.values())
        self.bert_vocab.remove("[UNK]")
        self.bert_vocab_index = Prober(args).init_indices_for_filter_logprobs(self.bert_vocab)
        self.common_vocab_subset =  load_vocab(self.common_vocab_path)
        self.common_vocab_index = Prober(args).init_indices_for_filter_logprobs(self.common_vocab_subset)

        

        # init the random model which share the backbone weights between differnet random model
        if not os.path.exists(os.path.join(self.output_dir,"random_weight",f"random_all_0.ckpt")):
            dir_path = os.path.join(self.output_dir,"random_weight")
            os.makedirs(dir_path)
            for i in range(10):
                self.generate_init_model("all", os.path.join(dir_path,f"random_all_{i}.ckpt"))
                self.generate_init_model("embedding", os.path.join(dir_path,f"random_embedding_{i}.ckpt"))

        # collect relations from jsonl
        data =  load_file("/home/jiao/code/prompt/OptiPrompt/relation_metainfo/LAMA_relations.jsonl")
        relations = [ d["relation"] for d in data]
        for r in relations:
            if os.path.exists(os.path.join(self.data_dir, r, "train.jsonl")):
                self.relations.append(r)


    def experiment_for_one_relation(self, relation:str, random_weight_index=0, ctrl_code=[1,1,1]):
        """
        ctrl code is used to control which experiment will execute
        """
        collect=[]
        if ctrl_code[0]==1:
            ckpt_1,(best_epoch_1,valid_loss_1)  = self.train_model([relation], random_init="all",random_weight_index=random_weight_index)
            precision_1,test_loss_1 = self.evaluate_model([relation], random_init="all", prompt_ckpt_path=ckpt_1,random_weight_index=random_weight_index)
            collect.append([relation, precision_1, best_epoch_1, valid_loss_1, test_loss_1])
        if ctrl_code[1]==1:
            ckpt_2,(best_epoch_2,valid_loss_2)  = self.train_model([relation], random_init="embedding",random_weight_index=random_weight_index)
            precision_2, test_loss_2 = self.evaluate_model([relation], random_init="embedding", prompt_ckpt_path=ckpt_2,random_weight_index=random_weight_index)
            collect.append([relation, precision_2, best_epoch_2, valid_loss_2, test_loss_2])
        if ctrl_code[2]==1:
            ckpt_3,(best_epoch_3,valid_loss_3)  = self.train_model([relation], random_init="none")
            precision_3, test_loss_3 = self.evaluate_model([relation], random_init="none", prompt_ckpt_path=ckpt_3)
            collect.append([relation, precision_3, best_epoch_3, valid_loss_3, test_loss_3])

        for item in collect:
            print(item)
            
        return collect


    def experiment_for_one_relation_new(self, relation:str, train_data:list, valid_data, test_data,ctrl_code=[1,1,1], weightsampler=None,random_weight_index=0):
        """
        TODO: 提供可选的训练集、测试机配置
        """
        # 1. 处理好数据集
        # 让dataloader 直接返回list
        def collate_fn(list_items):
            return list_items
        # 允许训练数据集使用采样器
        if weightsampler:
            train_samples_batches = DataLoader(train_data,batch_size=self.train_batchsize,sampler=weightsampler,collate_fn=collate_fn)
        else:
            train_samples_batches = DataLoader(train_data, batch_size=self.train_batchsize,collate_fn=collate_fn,shuffle=True)
        valid_samples_batches, valid_sentences_batches = batchify(valid_data, self.valid_batchsize)
        test_samples_batches, test_sentences_batches = batchify(test_data, self.valid_batchsize)

        # 2. 按照ctrlcode 来训练不同模型
        collect = []
        if ctrl_code[0] ==1:
            model = Prober(args,random_init="all")
            prepare_for_dense_prompt(model)
            #FIXME: 暂时是随机初始化的，如果先要使用手工模板，需要在这里使用convert_manual_to_dense而外初始化
            save_name="model-all-random_prompt-random-init.npy"
            ckpt,(best_epoch,valid_loss) = self.train_model_new(relation,model,train_samples_batches,valid_samples_batches,valid_sentences_batches,save_name=save_name)
            precision, test_loss = self.evaluate_model_new(model, eval_samples_batches=test_samples_batches, eval_sentences_batches=test_sentences_batches)
            collect.append([relation, precision, best_epoch,valid_loss, test_loss])
        else:
            collect.append([])
        if ctrl_code[1] ==1:
            model = Prober(args,random_init="embedding")
            prepare_for_dense_prompt(model)
            #FIXME: 暂时是随机初始化的，如果先要使用手工模板，需要在这里使用convert_manual_to_dense而外初始化
            save_name="model-embedding-random_prompt-random-init.npy"
            ckpt,(best_epoch,valid_loss) = self.train_model_new(relation,model,train_samples_batches,valid_samples_batches,valid_sentences_batches,save_name=save_name)
            precision, test_loss = self.evaluate_model_new(model, eval_samples_batches=test_samples_batches, eval_sentences_batches=test_sentences_batches)
            collect.append([relation, precision, best_epoch,valid_loss, test_loss])
        else:
            collect.append([])
        if ctrl_code[2] ==1:
            model = Prober(args,random_init="none")
            prepare_for_dense_prompt(model)
            #FIXME: 暂时是随机初始化的，如果先要使用手工模板，需要在这里使用convert_manual_to_dense而外初始化
            save_name="model-none-random_prompt-random-init.npy"
            ckpt,(best_epoch,valid_loss) = self.train_model_new(relation,model,train_samples_batches,valid_samples_batches,valid_sentences_batches,save_name=save_name)
            precision, test_loss = self.evaluate_model_new(model, eval_samples_batches=test_samples_batches, eval_sentences_batches=test_sentences_batches)
            collect.append([relation, precision, best_epoch,valid_loss, test_loss])
        else:
            collect.append([])
        
        for item in collect:
            print(item)

        return collect


    def experiment_reduce_majority(self):
        self.output_dir = "outputs/experiment/reduce_majorty"
        # create some csv file used to save result
        csv_dir = os.path.join(self.output_dir, "csv_result")
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)

        target_csvs = [
            "model-all-random_prompt-random-init.csv",
            "model-embedding-random_prompt-random-init.csv",
            "model-none-random_prompt-random-init.csv",
        ]
        for csv in target_csvs:
            if not os.path.exists(csv):
                with open(os.path.join(csv_dir, csv), "w") as f:
                    f.write("relation,precision,best_epoch,valid_loss,test_loss\n")
        
        for relation in tqdm(self.relations):
            self.experiment_for_one_relation(relation)
            # TODO this fun will be removed
            assert False==True
    
    def experiment_reduce_majority_N_times(self, N, ctrl_code=[1,1,1]):
        self.output_dir = "outputs/experiment/reduce_majorty"
        # create some csv file used to save result
        csv_dir = os.path.join(self.output_dir, "csv_result")
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        target_csvs = [
            f"model-all-random_prompt-random-init_{N}-times.csv",
            f"model-embedding-random_prompt-random-init_{N}-times.csv",
            f"model-none-random_prompt-random-init_{N}-times.csv",
        ]
        for i,csv in enumerate(target_csvs):
            if not os.path.exists(csv) and ctrl_code[i]==1:
                with open(os.path.join(csv_dir, csv), "w") as f:
                    f.write("relation")
                    for i in range(0,N):
                        f.write(f",precision{i+1}")
                    f.write(",mean,std")
                    f.write("\n")
        for relation in tqdm(self.relations):
            collects = []
            for i in range(N):
                collect = self.experiment_for_one_relation(relation,random_weight_index=i, ctrl_code=ctrl_code)
                collects.append(collect)
            """
            collect = [ 
                [relation, precision_1, best_epoch_1, valid_loss_1, test_loss_1],
                [relation, precision_2, best_epoch_2, valid_loss_2, test_loss_2],
                [relation, precision_3, best_epoch_3, valid_loss_3, test_loss_3]
            ]
            collects = [
                collect * N
            ]
            """
            for i, csv in enumerate(target_csvs):
                precision_list = []
                if ctrl_code[i]==1:
                    with open(os.path.join(self.output_dir,"csv_result",csv), "a") as f:
                        f.write(relation)
                        for j in range(N):
                            precision = collects[j][i][1]
                            precision_list.append(precision)
                            f.write(","+str(round(precision,5)))
                        f.write(",{:.5},{:.5}".format(np.mean(precision_list),np.std(precision_list)))
                        f.write("\n")
    
    
    def experiment_raw_N_times(self,N):
        """
        run experiment without any data preprocess N times
        """
        # important setting 
        self.obj_resample = False
        self.output_dir = "outputs/experiment/raw"
        # create some csv file used to save result
        csv_dir = os.path.join(self.output_dir, "csv_result")
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        target_csvs = [
            f"model-all-random_prompt-random-init_{N}-times.csv",
            f"model-embedding-random_prompt-random-init_{N}-times.csv",
            f"model-none-random_prompt-random-init_{N}-times.csv",
        ]
        for csv in target_csvs:
            if not os.path.exists(csv):
                with open(os.path.join(csv_dir, csv), "w") as f:
                    f.write("relation")
                    for i in range(0,N):
                        f.write(f",precision{i+1}")
                    f.write("\n")
        for relation in tqdm(self.relations):
            collects = []
            for i in range(N):
                collect = self.experiment_for_one_relation(relation, random_weight_index=i)
                collects.append(collect)
            """
            collect = [ 
                [relation, precision_1, best_epoch_1, valid_loss_1, test_loss_1],
                [relation, precision_2, best_epoch_2, valid_loss_2, test_loss_2],
                [relation, precision_3, best_epoch_3, valid_loss_3, test_loss_3]
            ]
            collects = [
                collect * 10
            ]
            """
            for i, csv in enumerate(target_csvs):
                with open(os.path.join(self.output_dir,"csv_result",csv), "a") as f:
                    f.write(relation)
                    for j in range(N):
                        precision = collects[j][i][1]
                        f.write(","+str(round(precision,5)))
                    f.write("\n")

    
    def experiment_trained_with_uni_N_times(self,N, ctrl_code=[1,1,1]):
        """
        trained with uni dataset
        """
        def type_of_obj(data:list):
            # 按照data中的obj，来得到一个type数组
            type = defaultdict(int)
            out = []
            for d in data:
                t  = type[d["obj_label"]]
                if t != 0:
                    out.append(t)
                else:
                    type[d["obj_label"]] = len(type)+1
                    out.append(len(type)+1)
            return out
                    

        self.obj_resample = False
        self.output_dir = "outputs/experiment/wiki-uni"
        # create some csv file used to save result
        csv_dir = os.path.join(self.output_dir, "csv_result")
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        target_csvs = [
            f"model-all-random_prompt-random-init_{N}-times.csv",
            f"model-embedding-random_prompt-random-init_{N}-times.csv",
            f"model-none-random_prompt-random-init_{N}-times.csv",
        ]
        for csv in target_csvs:
            if not os.path.exists(csv):
                with open(os.path.join(csv_dir, csv), "w") as f:
                    f.write("relation")
                    for i in range(0,N):
                        f.write(f",precision{i+1}")
                    f.write(",mean,std")
                    f.write("\n")

        
        # 准备数据集需要使用到模型特定的MASK
        # 这里的model是没有用的，因此需要删除
        model = Prober(args)
        model_MASK = model.MASK
        del model
        pbar = tqdm(total=N*len(self.relations))

        for relation in self.relations:
            # 准备数据集，list格式
            template = init_template(args,model=None)

            lama_vocab_subset = load_vocab(self.common_vocab_path)
            
            lama_data_path = os.path.join(self.data_dir,relation,"test.jsonl")
            lama_data, _ = load_data(lama_data_path, template, vocab_subset=lama_vocab_subset, mask_token=model_MASK)
            uni_data_dir = "/home/jiao/code/prompt/OptiPrompt/data/wiki_uni"
            uni_data_path = os.path.join(uni_data_dir,relation)
            uni_data, weight = load_data(uni_data_path, template, vocab_subset=self.bert_vocab, mask_token=model_MASK,  weight_sample=self.obj_resample)
            # type_list = type_of_obj(uni_data)
            weight = None

            autoprompt_train_data_path = os.path.join(self.data_dir,relation,"train.jsonl")
            # dev_data_path = os.path.join(self.data_dir,relation,"dev.jsonl")
            # train_data, weight = load_data(train_data_path, template, vocab_subset=vocab_subset,
            #                            mask_token=model_MASK, weight_sample=self.obj_resample, sub_reduce=self.sub_reduce)
            # valid_data, _ = load_data(dev_data_path, template, vocab_subset=vocab_subset, mask_token=model_MASK)
            
            

            test_data = lama_data
            train_data = uni_data
            # use auto prompt trainning data as valid data
            valid_data,_ = load_data(autoprompt_train_data_path, template, vocab_subset=self.bert_vocab,
                                       mask_token=model_MASK,)
            collects = []
            for i in range(N):
                collect = self.experiment_for_one_relation_new(relation,train_data,valid_data,test_data,ctrl_code=ctrl_code,weightsampler=weight)
                collects.append(collect)
                pbar.update(1)

            for i,csv in enumerate(target_csvs):
                precision_list = []
                if ctrl_code[i]!=1:
                    continue
                with open(os.path.join(self.output_dir,"csv_result",csv), "a") as f:
                    f.write(relation)
                    for j in range(N):
                        precision = collects[j][i][1]
                        precision_list.append(precision)
                        f.write(","+str(round(precision,5)))
                    f.write(",{},{}".format(
                        round(np.mean(precision_list), 5),
                        round(np.std(precision_list), 5))
                    )
                    f.write("\n")
            
            

    def experiment_reduce_majoriy_more(self):
        self.output_dir = "outputs/experiment/reduce_majorty_more"
        self.sub_reduce = True
        # create some csv file used to save result
        csv_dir = os.path.join(self.output_dir, "csv_result")
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)

        target_csvs = [
            "model-all-random_prompt-random-init.csv",
            "model-embedding-random_prompt-random-init.csv",
            "model-none-random_prompt-random-init.csv",
        ]
        for csv in target_csvs:
            if not os.path.exists(csv):
                with open(os.path.join(csv_dir, csv), "w") as f:
                    f.write("relation,precision,best_epoch,valid_loss,test_loss\n")
        
        for relation in tqdm(self.relations):
            self.experiment_for_one_relation(relation,write_to_file=True)
    
    def experiment_one_relation_N_times(self, relation, N, message="none"):
        out = []
        for i in tqdm(range(N)):
            _ = self.experiment_for_one_relation(relation,write_to_file=False)
            out.append(_)
        
        stdout_back = sys.stdout
        sys.stdout =  open("log.txt","w")
        print(message)
        res1 = [[], [], []]
        for item in out:
            p1 = item[0][1]
            p2 = item[1][1]
            p3 = item[2][1]
            res1[0].append(p1)
            res1[1].append(p2)
            res1[2].append(p3)
            print(p1,p2,p3)
        
        mean = [0]*3
        std = [0]*3
        for i in range(3):
            mean[i] = np.mean(res1[i])
            std[i] = np.std(res1[i])
        print(mean)
        print(std)

        sys.stdout = stdout_back

    def generate_init_model(self, random_init, save_path):
        model = Prober(args,random_init)
        torch.save(model.mlm_model.state_dict(), save_path)


    def train_model(self, relation:list, random_init="none", prompt_init="random", random_weight_index=0):
        """
        在relation上训练出来一个模型，并保存验证集合上最好的模型
        """
        model = Prober(args, random_init=random_init)
        # reinit
        if random_init is not "none":
            try:
                 model.mlm_model.load_state_dict(torch.load(
                        os.path.join(self.output_dir,"random_weight", f"random_{random_init}_{random_weight_index}.ckpt")))
            except:
                try:
                    model.mlm_model.load_state_dict(torch.load(os.path.join(os.path.dirname(
                    os.path.abspath(self.output_dir)),"random_weight", f"random_{random_init}_{random_weight_index}.ckpt")))
                except:
                    logger.error("train model : cannot load the init weight of random model")
                    exit(-1)


        prepare_for_dense_prompt(model)
        template = init_template(args,model)
        vocab_subset = load_vocab(self.common_vocab_path)
        filter_indices, index_list = model.init_indices_for_filter_logprobs(vocab_subset)
        original_vocab_size = len(list(model.tokenizer.get_vocab()))
        
        # 准备data
        train_data_path = [ os.path.join(self.data_dir,r,"train.jsonl") for r in relation]
        dev_data_path = [ os.path.join(self.data_dir,r,"dev.jsonl") for r in relation]
        
        train_data, weight = load_data(train_data_path, template, vocab_subset=vocab_subset,
                                       mask_token=model.MASK, weight_sample=self.obj_resample, sub_reduce=self.sub_reduce)
        valid_data, _ = load_data(dev_data_path, template, vocab_subset=vocab_subset, mask_token=model.MASK)
        # TODO use dataloader

        train_data = train_data
        valid_data = valid_data

        
        assert len(relation) == 1 # we only support one relation to train now
        def collate_fn(list_items):
            return list_items
        if self.obj_resample:
            weightsampler = WeightedRandomSampler(weight, len(weight))
            train_samples_batches = DataLoader(train_data, batch_size=self.train_batchsize,sampler=weightsampler,collate_fn=collate_fn)
        else:
            train_samples_batches = DataLoader(train_data, batch_size=self.train_batchsize,
            collate_fn=collate_fn)
        # valid_samples_batches = DataLoader(valid_data, batch_size=self.valid_batchsize,)
        valid_samples_batches, valid_sentences_batches = batchify(valid_data, self.valid_batchsize)
        
        # Valid set before train
        best_result,eval_loss, _ = evaluate(model, valid_samples_batches, valid_sentences_batches, filter_indices, index_list)
        best_epoch = -1
        save_dir = os.path.join(self.output_dir, relation[0])
        save_name=f"model-{random_init}-random_prompt-{prompt_init}-init.npy"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_optiprompt(save_dir, model, original_vocab_size, save_name=save_name)

        # Add word embeddings to the optimizer
        optimizer = AdamW([{'params': model.base_model.embeddings.word_embeddings.parameters()}], lr=self.learning_rate, correct_bias=False)
        t_total = len(train_samples_batches) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, int(t_total * self.warmup_proportion), t_total)

        # Train!!!
        global_step = 0
        eval_step = len(train_samples_batches)
        DEBUG = False
        if DEBUG:
            # use to check if the resample work!
            type = {}

        for _ in range(self.num_epochs):
            for samples_b in train_samples_batches :
                if DEBUG:
                    for i in samples_b:
                        if i["obj_label"] not in type:
                            type[i["obj_label"]] = 1
                        else:
                            type[i["obj_label"]] += 1


                sentences_b = [s["input_sentences"] for s in samples_b]

                loss = model.run_batch(sentences_b, samples_b, training=True, augment_training_data=self.augment_train_data, noise_std=self.noise_std)
                loss.backward()

                # set normal tokens' gradients to be zero
                for p in model.base_model.embeddings.word_embeddings.parameters():
                    # only update new tokens
                    p.grad[:original_vocab_size, :] = 0.0

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                if eval_step > 0 and global_step % (eval_step) == 0:
                    # Eval during training
                    logger.info('Global step=%d, evaluating...'%(global_step))
                    precision, eval_loss_,predict_result = evaluate(model, valid_samples_batches, valid_sentences_batches, filter_indices, index_list)
                    if precision > best_result:
                        best_result = precision
                        eval_loss = eval_loss_
                        logger.info('!!! Updated Best valid (epoch=%d): %.2f' %
                            (_, best_result * 100))
                        best_epoch = _
                        save_optiprompt(save_dir, model, original_vocab_size, save_name=save_name)

        if DEBUG:
            print(type)

        logger.info('Best Valid: %.2f'%(best_result*100))
        return os.path.join(save_dir,save_name), (best_epoch, eval_loss)

    def train_model_new(self, relation:str, model:Prober, train_samples_batches:DataLoader, valid_samples_batches, valid_sentences_batches, save_name="model-none-random_prompt-random-init.npy"):
        # evaluate before train
        # validation的过程中默认使用 bert vocab 而不是common_vocab
        vocab_subset = self.bert_vocab
        filter_indices, index_list = self.bert_vocab_index
        best_result,eval_loss, _ = evaluate(model, valid_samples_batches, valid_sentences_batches,filter_indices, index_list)
        best_epoch = -1
        original_vocab_size = len(list(model.tokenizer.get_vocab()))-MAX_NUM_VECTORS
        save_dir = os.path.join(self.output_dir, relation)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_optiprompt(save_dir, model, original_vocab_size, save_name=save_name)

        # Add word embeddings to the optimizer
        optimizer = AdamW([{'params': model.base_model.embeddings.word_embeddings.parameters()}], lr=self.learning_rate, correct_bias=False)
        t_total = len(train_samples_batches) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, int(t_total * self.warmup_proportion), t_total)

        # Train!!!
        global_step = 0
        eval_step = len(train_samples_batches)
        for _ in range(self.num_epochs):
            for samples_b in train_samples_batches :

                sentences_b = [s["input_sentences"] for s in samples_b]

                loss = model.run_batch(sentences_b, samples_b, training=True, augment_training_data=self.augment_train_data, noise_std=self.noise_std)
                loss.backward()

                # set normal tokens' gradients to be zero
                for p in model.base_model.embeddings.word_embeddings.parameters():
                    # only update new tokens
                    p.grad[:original_vocab_size, :] = 0.0

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                if eval_step > 0 and global_step % (eval_step) == 0:
                    # Eval during training
                    logger.info('Global step=%d, evaluating...'%(global_step))
                    precision, eval_loss_,predict_result = evaluate(model, valid_samples_batches, valid_sentences_batches, filter_indices, index_list)
                    if precision > best_result:
                        best_result = precision
                        eval_loss = eval_loss_
                        logger.info('!!! Updated Best valid (epoch=%d): %.2f' %
                            (_, best_result * 100))
                        best_epoch = _
                        save_optiprompt(save_dir, model, original_vocab_size, save_name=save_name)

        logger.info('Best Valid: %.2f'%(best_result*100))
        return os.path.join(save_dir,save_name), (best_epoch, eval_loss)

    def evaluate_model(self, relation:list, random_init="none",prompt_ckpt_path="none",random_weight_index=0, show_test_distribution=False, output_file="none"):

        # create model
        model = Prober(args, random_init=random_init)
        # reinit to the weight same as train
        if random_init is not "none":
            try:
                 model.mlm_model.load_state_dict(torch.load(
                        os.path.join(self.output_dir, "random_weight",f"random_{random_init}_{random_weight_index}.ckpt")))
            except:
                try:
                    model.mlm_model.load_state_dict(torch.load(os.path.join(os.path.dirname(
                    os.path.abspath(self.output_dir)),"random_weight", f"random_{random_init}_{random_weight_index}.ckpt")))
                except:
                    logger.error("evalute model : cannot load the init weight of random model")
                    exit(-1)
        # create template
        template = init_template(args,model)
        vocab_subset = load_vocab(self.common_vocab_path)
        filter_indices, index_list = model.init_indices_for_filter_logprobs(vocab_subset)
        original_vocab_size = len(list(model.tokenizer.get_vocab()))
        prepare_for_dense_prompt(model)
        
        # load template from prompt_ckpt_path
        logger.info("Loading OptiPrompt's [V]s..")
        with open(prompt_ckpt_path, 'rb') as f:
            vs = np.load(f)
        # copy fine-tuned new_tokens to the pre-trained model
        with torch.no_grad():
            model.base_model.embeddings.word_embeddings.weight[original_vocab_size:] = torch.Tensor(vs)

        # prepare data
        test_data_path = [ os.path.join(self.data_dir,r,"test.jsonl") for r in relation]
        test_data, _ = load_data(test_data_path, template, vocab_subset=vocab_subset, mask_token=model.MASK)
        eval_samples_batches, eval_sentences_batches = batchify(test_data,self.valid_batchsize)

        precision,loss, predict_result  = evaluate(model, eval_samples_batches, eval_sentences_batches, filter_indices, index_list)

        if show_test_distribution:
            self.show_prediction_distribution(predict_result[relation[0]], output_file=output_file)

        return precision,loss


    def evaluate_model_new(self, model:Prober, eval_samples_batches, eval_sentences_batches):
        vocab_subset = self.common_vocab_subset
        filter_indices, index_list = self.common_vocab_index

        precision,loss, predict_result  = evaluate(model, eval_samples_batches, eval_sentences_batches, filter_indices, index_list)
        return precision,loss




    def show_prediction_distribution(self, result, output_file="none"):
        """show the prediction output statistic info with a specified model weight
        result = {
                'uuid': sample['uuid'],
                'relation': sample['predicate_id'],
                'sub_label': sample['sub_label'],
                'obj_label': sample['obj_label'],
                'masked_sentences': sample['input_sentences'],
                'topk': topk,
            }
        将正确的按照类别打印出来
        """
        assert output_file != "none"

        correct = []
        for r in result:
            # 该预测正确
            if r["topk"][0]["token"] == r["obj_label"]:
                correct.append(r)

        stdout_back = sys.stdout
        sys.stdout =  open(output_file,"w")
        print("precision is {:4}\n".format(len(correct)/len(result)))
        # 统计正确预测的类别
        obj_class = {}
        for c in correct:
            if not obj_class.__contains__(c["obj_label"]):
                obj_class[c["obj_label"]] = []
            obj_class[c["obj_label"]].append(c)
        print("\nclass statistic")
        for key,value in obj_class.items():
            print("{} {} ".format(len(value),key))
        print("\nspecifically")
        for key,value in obj_class.items():
            print(key)
            for item in value:
                print("\t {}".format(item["sub_label"]))
            
        sys.stdout = stdout_back
    

    def pre_experiment(self, relation_train:list, relation_test:list, ):
        # pattern_split = exam_pattern_split(relation_train, relation_test, random_init="all")
        # print(pattern_split)
        # res = exam_pattern_split(relation_train, relation_test, random_init="embedding")
        # with open (os.path.join(output_dir,"result.txt"), "a") as f:
        #     f.writelines(str(res)+"\n")
        res = self.exam_pattern_split(relation_train, relation_test, random_init="none")
        with open (os.path.join(self.output_dir,"result.txt"), "a") as f:
            f.writelines(str(res)+"\n")


    def exam_pattern_split(self, relation_a:list, relation_b:list, random_init):
        # 随机初始化，在a上训练，在b上测试，得到分数
        ckpt_path_a,_ = self.train_model(relation_a,random_init)
        precision_a_a = self.evaluate_model(relation_a, random_init,prompt_ckpt_path=ckpt_path_a)
        precision_a_b = self.evaluate_model(relation_b, random_init,prompt_ckpt_path=ckpt_path_a)
        ckpt_path_b,_  = self.train_model(relation_b,random_init)
        precision_b_b = self.evaluate_model(relation_b, random_init, prompt_ckpt_path=ckpt_path_b)
        precision_b_a = self.evaluate_model(relation_a, random_init, prompt_ckpt_path=ckpt_path_b)
        
        return precision_a_a, precision_a_b, precision_b_a, precision_b_b   

    def show_data_distribution(self):
        """
        this func used to show the data distribution for different relation
        """
        model = Prober(args, random_init="none")
        template = init_template(args,model)
        vocab_subset = load_vocab(self.common_vocab_path)
        with open(os.path.join(self.output_dir,"train_data_distribuation.csv"), "w") as f:
            f.write("Relation")
            for i in range(1000):
                f.write(f",class{i}")
            f.write("\n")
        for r in self.relations:
            train_data_path = os.path.join(self.data_dir,r,"train.jsonl")
            save_file = os.path.join(self.output_dir,"train_data_distribuation.csv")
            load_data([train_data_path], template, vocab_subset=vocab_subset, mask_token=model.MASK,sub_reduce=True,save_data_info=save_file)
        with open(os.path.join(self.output_dir,"valid_data_distribuation.csv"), "w") as f:
            f.write("Relation")
            for i in range(1000):
                f.write(f",class{i}")
            f.write("\n")
        for r in self.relations:
            train_data_path = os.path.join(self.data_dir,r,"dev.jsonl")
            save_file = os.path.join(self.output_dir,"valid_data_distribuation.csv")
            load_data([train_data_path], template, vocab_subset=vocab_subset, mask_token=model.MASK,sub_reduce=True,save_data_info=save_file)
          
        with open(os.path.join(self.output_dir,"test_data_distribuation.csv"), "w") as f:
            f.write("Relation")
            for i in range(1000):
                f.write(f",class{i}")
            f.write("\n")
        for r in self.relations:
            train_data_path = os.path.join(self.data_dir,r,"test.jsonl")
            save_file = os.path.join(self.output_dir,"test_data_distribuation.csv")
            load_data([train_data_path], template, vocab_subset=vocab_subset, mask_token=model.MASK,sub_reduce=True,save_data_info=save_file)

    def show_wiki_uni_distribution(self):
        """
        统计uni数据级的类别分布
        """
        with open(os.path.join(self.output_dir,"uni_data_distribuation.csv"), "w") as f:
            f.write("Relation")
            f.write(",all_num,class_num")
            for i in range(500):
                f.write(f",class{i}")
            f.write("\n")
            for relation in self.relations:
                path = os.path.join("/home/jiao/code/prompt/OptiPrompt/data/wiki_uni",relation)
                try:
                    with open(path, "r") as f2:
                        data = json.load(f2)
                        obj_count = defaultdict(int)
                        for d in data:
                            obj_count[d["obj_label"]]+=1
                        # 按照字母表排序
                        temp = sorted(obj_count.items(), key = lambda obj:obj[0])   
                        f.write(relation)
                        f.write(",{},{}".format(temp[0][1]*len(temp),len(temp))) 
                        for t in temp:
                            f.write(f",{t[0]}:{t[1]}")
                        f.write('\n')
                except:
                    print("load Wiki UNI error")
                    exit(-1)
    
    
    def show_prediction_distribution(self, relation:str, prompt_ckpt_path):
        """
        该函数验证bert权重加上obj_resampled训练出来的连续prompt在LAMA和wikiUNI数据集上的分布是否一致
        输出两个prediction的类别以及个数，按照字母表排序
        """
        model = Prober(args)

        # create template
        prepare_for_dense_prompt(model)
        template = init_template(args,model)
        vocab_subset = load_vocab(self.common_vocab_path)
        filter_indices, index_list = model.init_indices_for_filter_logprobs(vocab_subset)
        original_vocab_size = len(list(model.tokenizer.get_vocab()))
        
        
        # load template from prompt_ckpt_path
        logger.info("Loading OptiPrompt's [V]s..")
        with open(prompt_ckpt_path, 'rb') as f:
            vs = np.load(f)
        # copy fine-tuned new_tokens to the pre-trained model
        with torch.no_grad():
            model.base_model.embeddings.word_embeddings.weight[original_vocab_size:] = torch.Tensor(vs)
        
        lama_data_path = os.path.join(self.data_dir,relation,"test.jsonl")
        lama_data, _ = load_data(lama_data_path, template, vocab_subset=vocab_subset, mask_token=model.MASK)
        lama_eval_samples_batches, lama_eval_sentences_batches = batchify(lama_data,self.valid_batchsize)
        lama_precision,lama_loss, lama_predict_result  = evaluate(model, lama_eval_samples_batches, lama_eval_sentences_batches, filter_indices, index_list)
        uni_data_dir = "/home/jiao/code/prompt/OptiPrompt/data/wiki_uni"
        uni_data_path = os.path.join(uni_data_dir,relation)
        uni_data, _ = load_data(uni_data_path, template, vocab_subset=vocab_subset, mask_token=model.MASK, )
        uni_eval_samples_batches, uni_eval_sentences_batches = batchify(uni_data,self.valid_batchsize)
        uni_precision,uni_loss, uni_predict_result  = evaluate(model, uni_eval_samples_batches, uni_eval_sentences_batches, filter_indices, index_list)

        # 接下来分析两个结果
        # 1. 比较精度
        print("***精度比较****")
        print("lama precision {:.5} vs uni precision {:5}".format(lama_precision, uni_precision))
        # 2. 比较loss
        print("***loss比较***")
        print("lama loss {:.5} vs uni loss {:.5}".format(lama_loss,uni_loss))
        # 3. 比较分布
        obj_count1 = defaultdict(int)
        obj_count2 = defaultdict(int)
        # 只比较prediction
        for d in lama_predict_result[relation]:
            obj_count1[d["topk"][0]["token"]]+=1
        for d in uni_predict_result[relation]:
            obj_count2[d["topk"][0]["token"]]+=1
        
        lama_predict = sorted(obj_count1.items(), key = lambda obj:obj[1],reverse=True)
        uni_predict = sorted(obj_count2.items(), key = lambda obj:obj[1], reverse=True)
        print("***分布比较***")
        print(lama_predict)
        print("\n\n")
        print(uni_predict)

exp = Experiment()

# exp.experiment_reduce_majority_N_times(10,ctrl_code=[1,0,0])
# exp.show_prediction_distribution("P19","/home/jiao/code/prompt/OptiPrompt/outputs/experiment/reduce_majorty/P19/model-none-random_prompt-random-init.npy")
exp.experiment_trained_with_uni_N_times(10,ctrl_code=[1,0,0])
# exp.show_wiki_uni_distribution()
# exp.obj_resample = True
# exp.augment_train_data = True
# exp.noise_std = 0.1
# exp.evaluate_model(["P1303"], random_init="all", prompt_ckpt_path="/home/jiao/code/prompt/OptiPrompt/outputs/experiment/reduce_majorty/P1303/model-all-random_prompt-random-init.npy",
#                    show_test_distribution=True, output_file="log1.txt")
# exp.evaluate_model(["P1303"], random_init="all", prompt_ckpt_path="/home/jiao/code/prompt/OptiPrompt/outputs/experiment/raw/P1303/model-all-random_prompt-random-init.npy",
#                    show_test_distribution=True, output_file="log2.txt")
# exp.output_dir = "/home/jiao/code/prompt/OptiPrompt/outputs/temp"

# set_seed(1)
# exp.generate_init_model("all")
# ckpt,_ = exp.train_model(["P30"],random_init="all")
# # print(f"save in :{ckpt}")
# ckpt = "/home/jiao/code/prompt/OptiPrompt/outputs/experiment/reduce_majorty/P30/model-all-random_prompt-random-init.npy"
# exp.evaluate_model(["P30"],show_test_distribution=True,output_file="log3.txt",prompt_ckpt_path=ckpt,random_init="all")