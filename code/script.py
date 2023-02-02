import argparse
import os
from truely_know import Experiment

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='roberta')
parser.add_argument('--model_name', type=str, default="roberta-large")
parser.add_argument('--prompt', type=str, default="optiprompt")
parser.add_argument('--num_tokens',type=int, default=5, help='only valid for continue prompt')
parser.add_argument('--epochs',type=int, default=10)
parser.add_argument('--learning_rate',type=float, default=3e-3)
parser.add_argument('--common_vocab', type=str, default="common_vocab_cased_be_ro_al.txt")
parser.add_argument('--save_path', type=str,default='none')
parser.add_argument('--repeat_times', type=int,default=1)
parser.add_argument('--evaluate_mode',type=bool,default=True)

args = parser.parse_args()



manual_prompts = ["LAMA", "LPAQA", "AutoPrompt"]

exp = Experiment()
exp.learning_rate = args.learning_rate
exp.num_epochs = args.epochs

exp.init_model(args.model_type, args.model_name)
exp.init_common_vocab(os.path.join(exp.work_dir, "common_vocabs", args.common_vocab))
if args.prompt in manual_prompts:
    exp.experiment_renormal_vector_debais_for_manual_prompt(manual_prompt=args.prompt)
else:
    exp.experiment_renormal_vector_debias_for_continue_prompt(continue_prompt=args.prompt, repeat_times=args.repeat_times,evaluate_mode=args.evaluate_mode)
exp.print_output()

if args.save_path != "none":
    exp.save_output(os.path.join(exp.work_dir, "outputs", args.save_path))


