# 晚间回顾笔记

## 2026-05-09

### PyTorch / Tensor Indexing

- `self.weight[token_ids]` 是 PyTorch 的张量索引，用在 Embedding 里就是查表。如果 `weight` 形状是 `(vocab_size, d_model)`，`token_ids` 形状是 `(...)`，输出形状就是 `(..., d_model)`。
- `token_ids` 不一定是一个标量索引，也可以是一个整数张量；PyTorch 会把里面每个 token id 替换成 `weight` 中对应的行向量。
- Embedding 的实际实现不是矩阵点积，而是按 id 查表。数学上它等价于 `one-hot(token_id) @ embedding_matrix`，但工程上不会真的构造 one-hot，因为查表更快、更省内存。
- Embedding 矩阵是可训练参数。前向时按 token id 查对应行；反向传播时，loss 只会给本 batch 用到的那些 token 行产生梯度，优化器再更新这些行。
- Transformer 的 token embedding 继承了传统词嵌入的思想：把离散 id 变成连续向量。区别是现代 LLM 通常嵌入的是 tokenizer 产生的 token/subword，而不是完整单词，并且 embedding 通常和整个模型端到端一起训练。
- `Float[Tensor, "vocab_size d_model"]` / `Int[Tensor, "..."]` 是 jaxtyping 风格的类型注解：前者表示浮点 Tensor，形状是 `(vocab_size, d_model)`；后者表示整数 Tensor，形状可以是任意维度。
- `module.load_state_dict({"weight": weights})` 是 PyTorch 加载参数的标准方式。`state_dict` 是“参数名 -> tensor”的字典，测试常用它把固定权重塞进你的模块，保证输出可和 snapshot 比较。
- `nn.Parameter` 本质上也是 Tensor，但会被 `nn.Module` 自动登记为可训练参数；普通 Tensor 不会自动出现在 `module.parameters()` 里。
- 直接写 `module.weight = weights` 会尝试用普通 Tensor 替换已注册的 Parameter，通常会报类型错误；`load_state_dict({"weight": weights})` 是把已有 Parameter 的数值加载成给定 tensor。
- Xavier / Glorot 初始化的目标是让每层输入输出的方差大致稳定，避免信号前向传播时爆炸或消失。常见公式是 `Var(W)=2/(fan_in+fan_out)`；均匀分布版本取值范围是 `[-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out))]`。
- Kaiming / He 初始化主要面向 ReLU 及其变体。因为 ReLU 会把负数置 0，信号方差大约损失一部分，所以常用 `Var(W)=2/fan_in` 来补偿前向传播中的尺度衰减。
