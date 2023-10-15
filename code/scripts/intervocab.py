from transformers import GPT2Tokenizer, GPT2Model
model_path = "/mnt/data/users/xuziyang/huggingface_model/gpt2-large"

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2Model.from_pretrained(model_path)
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)

print(output)