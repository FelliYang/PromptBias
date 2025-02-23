## 记录本次运行代码的过程，提交给github

运行scripts/debias_vs_calibration.sh 比较calibration和our debias在多个分类数据集上的性能差异
 - 发现transformers版本太低，更新相应版本
 - 无法访问hf，please use `export HF_ENDPOINT=https://hf-mirror.com`
 - 需要添加数据 textclassfication
 - 需要构造manual template，以openprompt/scripts内置的manual template为基础，将其输入部分修改成<input>
