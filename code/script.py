import argparse
import os
from truely_know import Experiment
from utils import set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="roberta")
parser.add_argument("--model_name", type=str, default="roberta-large")
parser.add_argument("--prompt", type=str, default="optiprompt")
parser.add_argument(
    "--num_tokens", type=int, default=5, help="only valid for continue prompt"
)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=3e-3)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument(
    "--common_vocab", type=str, default="common_vocab_cased_be_ro_al.txt"
)
parser.add_argument("--repeat_times", type=int, default=1)
parser.add_argument("--evaluate_mode", type=str, default="True")
parser.add_argument("--sampling_debias", type=str, default="False")
parser.add_argument("--save_path", type=str, default="none")
parser.add_argument("--typed_querying", type=str, default="True")
parser.add_argument("--do_debias", type=str, default="True")
parser.add_argument("--do_calibrate", type=str, default="False")
parser.add_argument("--eval_bias", type=str, default="False")
parser.add_argument("--eval_output_path", type=str, default="none")
parser.add_argument("--support_dataset", type=int, default=200)
parser.add_argument("--filter_biased_token_nums", type=int, default=0)

args = parser.parse_args()

# transfer str type to bool type
args.evaluate_mode = eval(args.evaluate_mode)
args.sampling_debias = eval(args.sampling_debias)
args.typed_querying = eval(args.typed_querying)
args.do_debias = eval(args.do_debias)
args.do_calibrate = eval(args.do_calibrate)
args.eval_bias = eval(args.eval_bias)


manual_prompts = ["LAMA", "LPAQA", "AutoPrompt"]

vocab_subset = "answer_type_tokens" if args.typed_querying else "common_vocab"

print(f"args.typed_querying is {args.typed_querying }, vocab_subset is: {vocab_subset}")

exp = Experiment(seed=args.seed, support_dataset=args.support_dataset)

if args.do_debias:
    exp.learning_rate = args.learning_rate
    exp.num_epochs = args.epochs

    exp.set_model(args.model_type, args.model_name)
    exp.set_common_vocab(os.path.join(exp.work_dir, "common_vocabs", args.common_vocab))
    if args.prompt in manual_prompts:
        exp.experiment_renormal_vector_debais_for_manual_prompt(
            manual_prompt=args.prompt,
            vocab_subset=vocab_subset,
            sampling_debias=args.sampling_debias,
            calibrate=args.do_calibrate,
            filter_biased_token_nums=args.filter_biased_token_nums,
        )
    else:
        exp.experiment_renormal_vector_debias_for_continue_prompt(
            continue_prompt=args.prompt,
            vocab_subset=vocab_subset,
            repeat_times=args.repeat_times,
            evaluate_mode=args.evaluate_mode,
            # TODO 添加filer_biased_tokens
        )
    exp.print_output()

    if args.save_path != "none":
        exp.save_output(
            os.path.join(exp.work_dir, "results", "filter_out_{}_biased_tokens".format(args.filter_biased_token_nums), args.save_path)
        )

if args.eval_bias:
    exp.set_model(args.model_type, args.model_name)
    exp.set_common_vocab(os.path.join(exp.work_dir, "common_vocabs", args.common_vocab))
    eval_out = exp.quantify_prompt_bias(prompt=args.prompt)

    if args.eval_output_path != "none":
        exp.save_KL_results(
            eval_out, os.path.join(exp.work_dir, "results", args.eval_output_path)
        )
