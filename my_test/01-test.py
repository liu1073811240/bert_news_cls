from transformers import BertModel, BertConfig


configuration = BertConfig.from_pretrained("yechen/bert-large-chinese")

configuration.max_position_embeddings = 3000
# print(configuration)

# 初始化模型
model = BertModel(configuration)
print(model)

configuration = model.config

print(configuration)

