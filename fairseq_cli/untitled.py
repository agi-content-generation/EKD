import torch
import torch.nn.functional as F

# 模拟数据
batch_size = 2
seq_len = 5
vocab_size = 10
temperature = 2.0

# 模拟 teacher 和 student 的 logits
teacher_logits = torch.randn(batch_size, seq_len, vocab_size) * 5
student_logits = torch.randn(batch_size, seq_len, vocab_size) * 2

# 模拟 target 和 padding mask
target = torch.randint(0, vocab_size, (batch_size, seq_len))
padding_idx = 0
pad_mask = (target == padding_idx)

# 确保 logits 数值范围稳定
teacher_logits = teacher_logits - teacher_logits.max(dim=-1, keepdim=True).values
student_logits = student_logits - student_logits.max(dim=-1, keepdim=True).values

# 转化为 log 概率和概率分布
lprobs = F.log_softmax(student_logits, dim=-1)  # 小模型的 log 概率分布
distil_lprobs = F.softmax(teacher_logits / temperature, dim=-1)  # 大模型的概率分布

# 计算 KL 散度
KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none')
KL_loss = KL_loss.sum(dim=-1)

# 应用 pad_mask 遮罩，避免填充位置的损失干扰
KL_loss.masked_fill_(pad_mask, 0.0)
KL_loss = KL_loss.sum()

# 模拟 golden_loss
golden_loss = torch.tensor(2.5)  # 假设 golden_loss 的值为一个标量

# 平衡损失
alpha = 0.5
loss = alpha * golden_loss + (1 - alpha) * KL_loss

# 输出检查
print(f"Loss: {loss}, KL Loss: {KL_loss}")
