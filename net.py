from transformers import BertModel, BertConfig
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1.
# pretrained = BertModel.from_pretrained("yechen/bert-large-chinese").to(DEVICE)
# print(pretrained.embeddings.position_embeddings)  # Embedding(512, 1024)
# 修改预训练模型的通道数
# pretrained.embeddings.position_embeddings = torch.nn.Embedding(3000, 1024).to(device=DEVICE)
# print(pretrained.embeddings.position_embeddings)

# 2. 超长文本只能从头训练。
configuration = BertConfig.from_pretrained("yechen/bert-large-chinese")
print(configuration)
configuration.max_position_embeddings = 3000
# 初始化模型
pretrained = BertModel(configuration).to(DEVICE)
print(pretrained.embeddings.position_embeddings)
print(pretrained)


# 定义下游任务模型（主干网络所提取的特征分类）
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = torch.nn.Linear(1024, 10)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 训练超长文本，主干网络需要放开训练
        # with torch.no_grad():
        #     out = pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        out = pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        out = self.fc(out.last_hidden_state[:, 0])
        out = out.softmax(dim=1)
        return out


