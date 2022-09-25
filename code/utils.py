from email.parser import Parser
import imp
import json
import os
from typing import OrderedDict
from tqdm import tqdm
import sys
import logging

import random
logger = logging.getLogger(__name__)
from models import Prober
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

MAX_NUM_VECTORS = 10

def load_vocab(vocab_filename):
    with open(vocab_filename, "r") as f:
        lines = f.readlines()
    vocab = [x.strip() for x in lines]
    return vocab

def load_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data

def parse_template(template, subject_label, object_label='[MASK]'):
    SUBJ_SYMBOL = "[X]"
    OBJ_SYMBOL = "[Y]"
    template = template.replace(SUBJ_SYMBOL, subject_label)
    template = template.replace(OBJ_SYMBOL, object_label)
    return [template]

def convert_tokens_to_string(tokens):
    out_string = " ".join(tokens).replace(" ##", "").strip()
    return out_string

def get_relation_meta(args):
    relations = load_file(args.relation_profile)
    for relation in relations:
        if relation['relation'] == args.relation:
            return relation
    raise ValueError('Relation info %s not found in file %s'%(args.relation, args.relation_profile))

def batchify(data, batch_size):
    list_samples_batches = []
    list_sentences_batches = []
    current_samples_batch = []
    current_sentences_batches = []
    
    c = 0
    for sample in data:
        input_sentences = sample['input_sentences']
        current_samples_batch.append(sample)
        current_sentences_batches.append(input_sentences)
        c += 1
        if c >= batch_size:
            list_samples_batches.append(current_samples_batch)
            list_sentences_batches.append(current_sentences_batches)
            current_samples_batch = []
            current_sentences_batches = []
            c = 0

    if current_samples_batch and len(current_samples_batch) > 0:
        list_samples_batches.append(current_samples_batch)
        list_sentences_batches.append(current_sentences_batches)

    return list_samples_batches, list_sentences_batches


def save_model(model, args):
    logger.info('Saving model...')
    model_to_save = model.mlm_model
    model_to_save.save_pretrained(args.output_dir)
    model.tokenizer.save_pretrained(args.output_dir)

def output_result(result, eval_loss):
    # descrepted
    logger.info('* Evaluation result *')
    cor = 0
    tot = 0
    macro = 0.0
    loss = 0.0
    for rel in result:
        cor_, tot_, avg_, loss_ = result[rel]
        cor += cor_
        tot += tot_
        macro += avg_
        loss_ /= tot_
        loss += loss_
        logger.info('%s\t%.5f\t%d\t%d\t%.5f' % (rel, avg_, cor_, tot_, loss_))
    macro = cor / tot if tot > 0 else 0.0
    micro = macro / len(result) if len(result) > 0 else 0.0
    logger.info('Macro avg: %.5f' % macro)
    logger.info('Micro avg: %.5f, Eval_loss: %.5f, Eval_loss (common vocab): %.5f' %(micro, eval_loss / tot, loss / len(result) if len(result) > 0 else 0.0))
    sys.stdout.flush()
    return micro, macro

def output_result_simple(result, eval_loss):
    logger.info('* Evaluation result *')
    cor = 0
    tot = 0
    loss = 0.0
    for rel in result:
        cor_, tot_, avg_, loss_ = result[rel]
        cor += cor_
        tot += tot_
        loss += loss_
        loss_ /= tot_
        logger.info('%s\t%.5f\t%d\t%d\t%.5f' % (rel, avg_, cor_, tot_, loss_))
    logger.info('Eval_loss: %.5f, Eval_loss (common vocab): %.5f' %(eval_loss / tot, loss / tot if len(result) > 0 else 0.0))
    sys.stdout.flush()
    return cor/tot, eval_loss/tot

def evaluate(model:Prober, samples_batches, sentences_batches, filter_indices=None, index_list=None, output_topk=None):
    vocab_to_common_vocab = None
    if index_list is not None:
        vocab_to_common_vocab = {}
        for cid, idx in enumerate(index_list):
            vocab_to_common_vocab[idx] = cid

    cor_all = 0
    tot_all = 0
    result = {}
    list_of_predictions = {}
    eval_loss = 0.0
    common_eval_loss = 0.0
    filtered_examples_num = 0

    for i in range(len(samples_batches)):
        samples_b = samples_batches[i]
        sentences_b = sentences_batches[i]
        log_probs, cor_b, tot_b, pred_b, topk_preds, loss, common_vocab_loss = model.run_batch(sentences_b, samples_b, training=False, filter_indices=filter_indices, index_list=index_list, vocab_to_common_vocab=vocab_to_common_vocab, filtered_example_num=filtered_examples_num)
        cor_all += cor_b
        tot_all += tot_b
        for pred, sample, topk, vocab_loss in zip(pred_b, samples_b, topk_preds, common_vocab_loss):
            rel = sample['predicate_id']
            if rel not in result:
                result[rel] = (0, 0, 0, 0.0)
                list_of_predictions[rel] = []
            cor, tot, _, rel_tot_loss = result[rel]
            tot += 1
            cor += pred
            rel_tot_loss += vocab_loss
            result[rel] = (cor, tot, cor / tot if tot > 0 else 0.0, rel_tot_loss)
            list_of_predictions[rel].append({
                'uuid': sample['uuid'],
                'relation': sample['predicate_id'],
                'sub_label': sample['sub_label'],
                'obj_label': sample['obj_label'],
                'masked_sentences': sample['input_sentences'],
                'topk': topk,
            })
        
        eval_loss += loss.item() * tot_b
    
    if filter_indices is not None  and filtered_examples_num>0:
        logger.info(f"filtered {filtered_examples_num} example(s) in test cause common vocab")

    if output_topk is not None:
        logger.info('Output top-k prediction to %s..'%output_topk)
        for rel in list_of_predictions:
            with open(os.path.join(output_topk, '%s_predictions.jsonl'%rel), 'w') as f:
                f.write('\n'.join([json.dumps(x) for x in list_of_predictions[rel]]))
    # result中包含 {relation :（corrent_prediction, total_prediction, c/t, relation_tot_loss）, xxx}
    precion,eval_loss_avg = output_result_simple(result, eval_loss)
    return precion, eval_loss_avg

def gen_feature_sample(data_sample, template, mask_token='[MASK]'):
    feature_sample = {}
    feature_sample['predicate_id'] = data_sample['predicate_id']
    feature_sample['sub_label'] = data_sample['sub_label']
    feature_sample['obj_label'] = data_sample['obj_label']
    feature_sample['uuid'] = data_sample['uuid'] if 'uuid' in data_sample else ''
    masked_sentence = parse_template(template.strip(), feature_sample['sub_label'].strip(), mask_token)
    feature_sample['input_sentences'] = [masked_sentence[0]]
    return feature_sample

# def load_data_one_relation(data_path, template, vocab_subset=None, mask_token='[MASK]'):
#     all_samples = []

#     distinct_facts = set()
#     raw_samples = load_file(data_path)
#     filtered = 0
#     for data_sample in raw_samples:
#         # follow the LAMA setting, only keep distinct (sub, obj) pairs
#         if (data_sample['sub_label'], data_sample['obj_label']) in distinct_facts:
#             continue
#         if (data_sample['obj_label'] not in vocab_subset):
#             filtered += 1
#             continue
#         # assert data_sample['obj_label'] in vocab_subset

#         distinct_facts.add((data_sample['sub_label'], data_sample['obj_label']))

#         feature_sample = gen_feature_sample(data_sample, template, mask_token)
#         all_samples.append(feature_sample)

#     logger.warning(f"filter {filtered} examples in load_data")
#     return all_samples

def get_new_token(vid):
    assert(vid > 0 and vid <= MAX_NUM_VECTORS)
    return '[V%d]'%(vid)

def convert_manual_to_dense(manual_template, model):
    def assign_embedding(new_token, token):
        """
        assign the embedding of token to new_token
        """
        logger.info('Tie embeddings of tokens: (%s, %s)'%(new_token, token))
        id_a = model.tokenizer.convert_tokens_to_ids([new_token])[0]
        id_b = model.tokenizer.convert_tokens_to_ids([token])[0]
        with torch.no_grad():
            model.base_model.embeddings.word_embeddings.weight[id_a] = model.base_model.embeddings.word_embeddings.weight[id_b].detach().clone()

    new_token_id = 0
    template = []
    for word in manual_template.split():
        if word in ['[X]', '[Y]']:
            template.append(word)
        else:
            tokens = model.tokenizer.tokenize(' ' + word)
            for token in tokens:
                new_token_id += 1
                template.append(get_new_token(new_token_id))
                assign_embedding(get_new_token(new_token_id), token)

    return ' '.join(template)

def init_template(args, model):
    # FIXME 由于args中使用的relation_train 为数组，因此不能直接用于手工template初始化
    if args.init_manual_template:
        relation = get_relation_meta(args)
        template = convert_manual_to_dense(relation['template'], model)
    else:
        template = '[X] ' + ' '.join(['[V%d]'%(i+1) for i in range(args.num_vectors)]) + ' [Y] .'
    return template



def load_data(data_path:list, template, vocab_subset=None, mask_token='[MASK]', weight_sample = False, sub_reduce=False, save_data_info="none"):
    """
    load data only support load one relation one time
    get the filtered data examples and the optional corrensponding sample weight use in train data,
     which helps reduce the obj_label inbalance
    save_data_info is used to save some info like the data distribution in dataset
    """
    assert len(data_path) == 1
    all_samples = []
    weights = []

    distinct_facts = set()
    raw_samples = load_file(data_path[0])
    filtered = 0
    for data_sample in raw_samples:
        # follow the LAMA setting, only keep distinct (sub, obj) pairs
        if (data_sample['sub_label'], data_sample['obj_label']) in distinct_facts:
            continue
        if (data_sample['obj_label'] not in vocab_subset):
            filtered += 1
            continue
        # assert data_sample['obj_label'] in vocab_subset

        distinct_facts.add((data_sample['sub_label'], data_sample['obj_label']))

        feature_sample = gen_feature_sample(data_sample, template, mask_token)
        all_samples.append(feature_sample)

    logger.warning(f"filter {filtered} examples in load_data {os.path.basename(os.path.dirname(data_path[0]))}")

    if sub_reduce:
        # count the num of each obj_label in data
        obj_count = {}
        # record type of sample in dataset
        type = []

        max_class_samples = []
        other_class_samples = []
        for e in all_samples:
            cls = e["obj_label"]
            type.append(cls)
            if not obj_count.__contains__(cls):
                obj_count[cls] = 1
            else:
                obj_count[cls] += 1

        max_type = list(obj_count.keys())[0]
        for key,value in obj_count.items():
            if value > obj_count[max_type]:
                max_type = key
        for i,t in enumerate(type):
            if t == max_type:
                max_class_samples.append(all_samples[i])
            else:
                other_class_samples.append(all_samples[i])
        
        # now we had split the max_class samples with others
        # next, we will random sample |other_class_samples| from the max_class_samples
        if save_data_info!="none":
            with open(save_data_info,"a") as f:
                f.write("{}".format(
                    os.path.basename(os.path.dirname(data_path[0])),
                    )
                )
                # count number of every obj type in dataset
                
                temp = sorted(obj_count.items(), key = lambda obj:obj[1], reverse=True)
                for i in range(len(temp)):
                    class_name = temp[i][0]
                    class_num = temp[i][1]
                    f.write(f",{class_name}:{class_num}")
                f.write("\n")

        index = np.random.choice(len(max_class_samples),int(15))
        reduced_max_class_samples = []
        for i in range(len(max_class_samples)):
            if i in index:
                reduced_max_class_samples.append(max_class_samples[i])
        
        all_samples = reduced_max_class_samples + other_class_samples
          


    if weight_sample:
        # count the num of each obj_label in data
        obj_count = {}
        # record type of sample in dataset
        type = []
        for e in all_samples:
            cls = e["obj_label"]
            type.append(cls)
            if not obj_count.__contains__(cls):
                obj_count[cls] = 1
            else:
                obj_count[cls] += 1
        
        weights = [1 / obj_count[type[i]] for i in range(len(all_samples))] 
        
    
    return all_samples, weights

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(seed)


def prepare_for_dense_prompt(model):
    new_tokens = [get_new_token(i+1) for i in range(MAX_NUM_VECTORS)]
    model.tokenizer.add_tokens(new_tokens)
    ebd = model.mlm_model.resize_token_embeddings(len(model.tokenizer))
    logger.info('# vocab after adding new tokens: %d'%len(model.tokenizer))

def save_optiprompt(save_path, model, original_vocab_size, save_name="prompt_vecs.npy"):
    logger.info("Saving OptiPrompt's [V]s..")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    vs = model.base_model.embeddings.word_embeddings.weight[original_vocab_size:].detach().cpu().numpy()
    with open(os.path.join(save_path, save_name), 'wb') as f:
        np.save(f, vs)



def load_optiprompt(args, ckpt_path="none"):
    # load bert model (pre-trained)
    model = Prober(args, random_init=args.random_init)
    original_vocab_size = len(list(model.tokenizer.get_vocab()))
    prepare_for_dense_prompt(model)
    
    logger.info("Loading OptiPrompt's [V]s..")

    if ckpt_path=="none": 
        ckpt_path = args.ckpt_path
    with open(ckpt_path, 'rb') as f:
        vs = np.load(f)
    
    # copy fine-tuned new_tokens to the pre-trained model
    with torch.no_grad():
        model.base_model.embeddings.word_embeddings.weight[original_vocab_size:] = torch.Tensor(vs)
    return model

class PromptDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.length = len(self.data)
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return self.length


