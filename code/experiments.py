import argparse
import sys
from ast import arg, parse
from asyncore import write
from distutils.debug import DEBUG
from gc import collect
import json
import logging
import os
import random
import sys
from black import out
from click import option, prompt
from joblib import Parallel

import numpy as np
from tokenizers import trainers
import torch
from torch import optim, rand, save
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
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
        

        # init the random model which share the backbone weights between differnet random model
        if not os.path.exists(os.path.join(self.output_dir,f"random_all.ckpt")):
            os.makedirs(self.output_dir)
            self.generate_init_model("all")
            self.generate_init_model("embedding")

        # collect relations from jsonl
        data =  load_file("/home/jiao/code/prompt/OptiPrompt/relation_metainfo/LAMA_relations.jsonl")
        relations = [ d["relation"] for d in data]
        for r in relations:
            if os.path.exists(os.path.join(self.data_dir, r, "train.jsonl")):
                self.relations.append(r)


    def experiment_for_one_relation(self, relation:str, write_to_file=False):
        collect=[]
        ckpt_1,(best_epoch_1,valid_loss_1)  = self.train_model([relation], random_init="all")
        precision_1,test_loss_1 = self.evaluate_model([relation], random_init="all", prompt_ckpt_path=ckpt_1)
        collect.append([relation, precision_1, best_epoch_1, valid_loss_1, test_loss_1])
        ckpt_2,(best_epoch_2,valid_loss_2)  = self.train_model([relation], random_init="embedding")
        precision_2, test_loss_2 = self.evaluate_model([relation], random_init="embedding", prompt_ckpt_path=ckpt_2)
        collect.append([relation, precision_2, best_epoch_2, valid_loss_2, test_loss_2])
        ckpt_3,(best_epoch_3,valid_loss_3)  = self.train_model([relation], random_init="none")
        precision_3, test_loss_3 = self.evaluate_model([relation], random_init="none", prompt_ckpt_path=ckpt_3)
        collect.append([relation, precision_3, best_epoch_3, valid_loss_3, test_loss_3])
        if not write_to_file:
            for item in collect:
                print(item)
            
            return collect

        target_csvs = [
            "model-all-random_prompt-random-init.csv",
            "model-embedding-random_prompt-random-init.csv",
            "model-none-random_prompt-random-init.csv",
        ]
        for i, csv in enumerate(target_csvs):
            with open(os.path.join(self.output_dir,"csv_result",csv), "a") as f:
                for t in collect[i]:
                    if t!=collect[i][-1]:
                        f.write(str(t)+",")
                    else:
                        f.write(str(t)+"\n")
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
            self.experiment_for_one_relation(relation,write_to_file=True)
    
    def experiment_reduce_majority_N_times(self, N):
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
                collect = self.experiment_for_one_relation(relation)
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
                collect = self.experiment_for_one_relation(relation)
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

    def generate_init_model(self, random_init):
        model = Prober(args,random_init)
        torch.save(model.mlm_model.state_dict(), os.path.join(self.output_dir,f"random_{random_init}.ckpt"))


    def train_model(self, relation:list, random_init="none", prompt_init="random"):
        """
        在relation上训练出来一个模型，并保存验证集合上最好的模型
        """
        model = Prober(args, random_init=random_init)
        # reinit
        if random_init is not "none":
            try:
                model.mlm_model.load_state_dict(torch.load(os.path.join(os.path.dirname(
                    os.path.abspath(self.output_dir)), f"random_{random_init}.ckpt")))
            except:
                try:
                    model.mlm_model.load_state_dict(torch.load(
                        os.path.join(self.output_dir, f"random_{random_init}.ckpt")))
                except:
                    logger.error("cannot load the init weight of random model")
        template = init_template(args,model)
        vocab_subset = load_vocab(self.common_vocab_path)
        filter_indices, index_list = model.init_indices_for_filter_logprobs(vocab_subset)
        original_vocab_size = len(list(model.tokenizer.get_vocab()))
        prepare_for_dense_prompt(model)
        # 准备data
        train_data_path = [ os.path.join(self.data_dir,r,"train.jsonl") for r in relation]
        dev_data_path = [ os.path.join(self.data_dir,r,"dev.jsonl") for r in relation]
        
        train_data, weight = load_data(train_data_path, template, vocab_subset=vocab_subset,
                                       mask_token=model.MASK, weight_sample=self.obj_resample, sub_reduce=self.sub_reduce)
        valid_data, _ = load_data(dev_data_path, template, vocab_subset=vocab_subset, mask_token=model.MASK)
        # TODO use dataloader

        train_data = PromptDataset(train_data)
        valid_data = PromptDataset(valid_data)

        
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
        best_result,eval_loss = evaluate(model, valid_samples_batches, valid_sentences_batches, filter_indices, index_list)
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
        # DEBUG = True
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
                    precision, eval_loss_ = evaluate(model, valid_samples_batches, valid_sentences_batches, filter_indices, index_list)
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


    def evaluate_model(self, relation:list, random_init="none",prompt_ckpt_path="none"):

        # create model
        model = Prober(args, random_init=random_init)
        # reinit to the weight same as train
        if random_init is not "none":
            try:
                model.mlm_model.load_state_dict(torch.load(os.path.join(os.path.dirname(
                    os.path.abspath(self.output_dir)), f"random_{random_init}.ckpt")))
            except:
                try:
                    model.mlm_model.load_state_dict(torch.load(
                        os.path.join(self.output_dir, f"random_{random_init}.ckpt")))
                except:
                    logger.error("cannot load the init weight of random model")
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

        precision,loss  = evaluate(model, eval_samples_batches, eval_sentences_batches, filter_indices, index_list)
        return precision,loss


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
            
        with open(os.path.join(self.output_dir,"test_data_distribuation.csv"), "w") as f:
            f.write("Relation")
            for i in range(1000):
                f.write(f",class{i}")
            f.write("\n")
        for r in self.relations:
            train_data_path = os.path.join(self.data_dir,r,"test.jsonl")
            save_file = os.path.join(self.output_dir,"test_data_distribuation.csv")
            load_data([train_data_path], template, vocab_subset=vocab_subset, mask_token=model.MASK,sub_reduce=True,save_data_info=save_file)


# evaluate_model( [ "P101"], prompt_ckpt_path="outputs/pre_experiment/trained_on_['P101']_random_none.npy")

# generate_init_model(random_init="all")
# generate_init_model(random_init="embedding")

# experiment(["P101"],["P103"])

# ckpt_path_a = train_model(["P101"],"none")
# precision_a = evaluate_model(["P101"], "none",prompt_ckpt_path=ckpt_path_a) 
# ckpt_path_b = train_model(["P36"],"none")
# precision_b = evaluate_model(["P36"], "none",prompt_ckpt_path=ckpt_path_b)
# print(precision_a, precision_b)
# print(precision_a)


# model = Prober(args)
# template = init_template(args, model)
# vocab_subset = load_vocab(common_vocab_path)
# load_data_one_to_one("/home/jiao/code/prompt/OptiPrompt/data/autoprompt_data/P1376/train.jsonl",template, vocab_subset)

def experiment_data_reduce_with(relation,run_full_data=False, augement_data=False,sub_reduce=False):  

    exp = Experiment()
    exp.augment_train_data = augement_data
    # exp.experiment_reduce_majority()
    random_times = 10
    if run_full_data:
        exp.sub_reduce = False
        out1 = []
        for i in tqdm(range(random_times)):
            set_seed(args.seed+i)
            _ = exp.experiment_for_one_relation(relation,write_to_file=False)
            out1.append(_)

    out2 = []
    if sub_reduce:
        exp.sub_reduce = True
        for i in tqdm(range(random_times)):
            set_seed(args.seed+i)
            _ = exp.experiment_for_one_relation(relation,write_to_file=False)
            out2.append(_)

    if run_full_data:
        res1 = [[]] * 3
        print("*****for reduce majority*******")
        for item in out1:
            p1 = item[0][1]
            p2 = item[1][1]
            p3 = item[2][1]
            res1[0].append(p1)
            res1[1].append(p2)
            res1[2].append(p3)
            print(p1,p2,p3)


    if sub_reduce:
        res2 = [[]] * 3
        print("***** for reduce more *********")
        for item in out2:
            p1 = item[0][1]
            p2 = item[1][1]
            p3 = item[2][1]
            res2[0].append(p1)
            res2[1].append(p2)
            res2[2].append(p3)
            print(p1,p2,p3)

    print("*******for average ******")
    if run_full_data:
        mean = [0]*3
        std = [0]*3
        for i in range(3):
            mean[i] = np.mean(res1[i])
            std[i] = np.std(res1[i])
        print(mean)
        print(std)
    if sub_reduce:
        mean = [0]*3
        std = [0]*3
        for i in range(3):
            mean[i] = np.mean(res2[i])
            std[i] = np.std(res2[i])
        print(mean)
        print(std)

    
# experiment_data_reduce_with("P140",run_full_data=True)
# exp = Experiment()

# exp = Experiment()
# # exp.show_data_distribution()
# # exp.experiment_reduce_majority_N_times(10)
# exp.augment_train_data = True

# collect=[]
# relation = "P30"
# for i in tqdm(range(5)):
#     ckpt_1,(best_epoch_1,valid_loss_1)  = exp.train_model([relation], random_init="all")
#     precision_1,test_loss_1 = exp.evaluate_model([relation], random_init="all", prompt_ckpt_path=ckpt_1)
#     collect.append([relation, precision_1, best_epoch_1, valid_loss_1, test_loss_1])

# precisions = []
# for i in range(5):
#     print(collect[i])
#     precisions.append(collect[i][1])
# print("mean {} std {}".format(np.mean(np.array(precisions)),np.std(np.array(precisions)) ) )
# exp.experiment_raw_N_times(10)

# experiment_data_reduce_with("P30",run_full_data=)
exp = Experiment()
exp.obj_resample = True
exp.augment_train_data = True
exp.noise_std = 0.1
exp.experiment_one_relation_N_times("P1303",5)