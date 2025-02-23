# 提取统计结果
import json
root_dir = r"/opt/data/private/PromptBias/results/filter_out_-1_biased_tokens/bert-base-cased/common_vocab_cased/typed_querying"

import os

ablation_path = "/"

results = {}
results["add_scale_add_norm"] = {
    "Manual":root_dir+"/LAMA_result.json",
    "LPAQA":root_dir+"/LPAQA_result.json",
    "AutoPrompt":root_dir+"/AutoPrompt_result.json"}
results["calibration"] = {
    "Manual":root_dir+"/LAMA_result_do_debias.json",
    "LPAQA":root_dir+"/LPAQA_result_do_debias.json",
    "AutoPrompt":root_dir+"/AutoPrompt_result_do_debias.json"
}
results["base"] = {
    "Manual":root_dir+"/ablations/LAMA"+"/no_normal_no_rescale_result.json",
    "LPAQA":root_dir+"/ablations/LPAQA"+"/no_normal_no_rescale_result.json",
    "AutoPrompt":root_dir+"/ablations/AutoPrompt"+"/no_normal_no_rescale_result.json"
}
results["add_rescale"] = {
    "Manual":root_dir+"/ablations/LAMA"+"/no_normalization_result.json",
    "LPAQA":root_dir+"/ablations/LPAQA"+"/no_normalization_result.json",
    "AutoPrompt":root_dir+"/ablations/AutoPrompt"+"/no_normalization_result.json"
}
results["add_norm"] = {
    "Manual":root_dir+"/ablations/LAMA"+"/no_rescale_result.json",
    "LPAQA":root_dir+"/ablations/LPAQA"+"/no_rescale_result.json",
    "AutoPrompt":root_dir+"/ablations/AutoPrompt"+"/no_rescale_result.json"
}


from prettytable import PrettyTable

table = PrettyTable()
table.title = "BERT-base 对比实验&消融实验"

# 
table.field_names = ["datasets","prompts","vanilla", "Prec.^d","Calibration","Prec.^d_add_norm", "Prec.^d_add_rescale", "Prec.^d_add_norm_and_rescale"]
for dataset in ["WIKI-UNI","LAMA","LAMA-WHU"]:
    for prompt in ["Manual","LPAQA","AutoPrompt"]:
        with open(results["base"][prompt],"r") as f:
            res = json.load(f)
        t = next(iter(next(iter(res.values()))[dataset].values()))
        vanilla = t["P"]
        Prec_d = t["P_d"]
        # 对比
        with open(results["calibration"][prompt],"r") as f:
            res = json.load(f)
        t = next(iter(next(iter(res.values()))[dataset].values()))
        assert t["P"] == vanilla
        calibration = t["P_d"]
        # 消融 1
        with open(results["add_scale_add_norm"][prompt],"r") as f:
            res = json.load(f)
        t = next(iter(next(iter(res.values()))[dataset].values()))
        assert t["P"] == vanilla
        add_norm_add_rescale = t["P_d"]
        # 消融 2
        with open(results["add_rescale"][prompt],"r") as f:
            res = json.load(f)
        t = next(iter(next(iter(res.values()))[dataset].values()))
        assert t["P"] == vanilla
        add_rescale = t["P_d"]
        # 消融 3
        with open(results["add_norm"][prompt],"r") as f:
            res = json.load(f)
        t = next(iter(next(iter(res.values()))[dataset].values()))
        assert t["P"] == vanilla
        add_norm = t["P_d"]
        table.add_row([dataset,prompt] + [round(i*100,1) for i in [vanilla,Prec_d,calibration,add_norm,add_rescale,add_norm_add_rescale]])

print(table)

        

        
        

