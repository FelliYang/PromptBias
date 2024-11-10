homedir=/opt/data/private/PromptBias
python=/root/miniconda3/envs/openprompt/bin/python
script=${homedir}/code/zero_shot.py

cd $homedir

$python $script --dataset yahoo --template_id 0
$python $script --dataset yahoo --template_id 1
$python $script --dataset yahoo --template_id 2
$python $script --dataset yahoo --template_id 3
$python $script --dataset agnews --template_id 0
$python $script --dataset agnews --template_id 1
$python $script --dataset agnews --template_id 2
$python $script --dataset agnews --template_id 3
$python $script --dataset dbpedia --template_id 0
$python $script --dataset dbpedia --template_id 1
$python $script --dataset dbpedia --template_id 2
$python $script --dataset dbpedia --template_id 3
$python $script --dataset imdb --template_id 0
$python $script --dataset imdb --template_id 1
$python $script --dataset imdb --template_id 2
$python $script --dataset imdb --template_id 3
# $python $script --dataset amazon --template_id 0
# $python $script --dataset amazon --template_id 1
# $python $script --dataset amazon --template_id 2
# $python $script --dataset amazon --template_id 3
