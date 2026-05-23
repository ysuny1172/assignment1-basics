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
- `param.data = tensor` 是直接替换 Parameter 底层数据，能绕过普通赋值的类型检查，但也绕过 autograd 的安全机制；测试加载权重更推荐 `load_state_dict` 或在 `torch.no_grad()` 下复制数值。
- `state_dict` 的 key 来自模块属性路径：对子模块自己加载时常用 `"weight"`，对整个模型加载时要写完整路径如 `"q_proj.weight"`；不确定名字时看 `state_dict().keys()` 或 `named_parameters()`。
- Xavier / Glorot 初始化的目标是让每层输入输出的方差大致稳定，避免信号前向传播时爆炸或消失。常见公式是 `Var(W)=2/(fan_in+fan_out)`；均匀分布版本取值范围是 `[-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out))]`。
- Kaiming / He 初始化主要面向 ReLU 及其变体。因为 ReLU 会把负数置 0，信号方差大约损失一部分，所以常用 `Var(W)=2/fan_in` 来补偿前向传播中的尺度衰减。

## 2026-05-10

### LLM 项目阅读

- `MedicalGPT` 这类项目不是单个医疗问答模型，而是领域大模型训练流水线：PT 让模型适应领域语料分布，SFT 让模型学会按指令回答，DPO/RLHF/ORPO/GRPO/OPD 再做偏好对齐或蒸馏。
- 医疗大模型项目要特别区分“研究训练框架”和“临床可用产品”；README 示例回答不等于经过医学验证，真实使用必须有人类医生和安全评估兜底。

### Attention / Masking

- Attention mask 通常约定 `True` 表示允许看，`False` 表示禁止看；`masked_fill(mask == False, -inf)` 是把禁止位置的 score 变成负无穷。
- 必须在 softmax 前 mask：因为 `exp(-inf)=0`，被禁止位置 softmax 后概率为 0；如果填 0，则 `exp(0)=1`，并不等于“不给注意力”。

### Python / Imports

- `einops` 包名容易拼错，正确导入是 `from einops import rearrange`，不是 `eniops`。
- `ModuleNotFoundError` 要分两层看：先确认包名拼写正确，再确认当前运行代码的 Python 环境里真的安装了这个包。

### Python / argparse

- `argparse.ArgumentParser()` 创建命令行参数解析器；`add_argument("--batch_size", type=int, default=32)` 定义一个可选参数，终端传入 `--batch_size 64` 时会解析成整数 64，不传则使用默认值 32。
- 解析后的结果通常来自 `args = parser.parse_args()`，再用 `args.batch_size` 访问；命令行里的连字符参数名会映射成 Python 属性名。
- `action="store_true"` 用来定义布尔开关：命令行出现该 flag 时值为 `True`，不出现时默认 `False`；适合 `--no_rope`、`--no_rms_norm` 这类消融开关。
- `choices=[...]` 会限制参数只能取指定字符串，传错会让 argparse 直接报错；适合 `--norm_mode pre/post`、`--ffn_type swiglu/silu` 这类实验枚举。

### Multi-Head Attention / Shapes

- `d_model % num_heads == 0` 保证每个 head 的维度 `d_head = d_model / num_heads` 是整数，否则无法把最后一维整齐 reshape 成 `(num_heads, d_head)`。
- 如果不满足整除，后面常见结果是 reshape/rearrange 报 shape mismatch；即使强行截断或补零，也会改变模型参数和信息分配，不再是标准 multi-head attention。
- 先做一次大 Q/K/V 投影再按 head 拆分，数学上等价于把投影矩阵按列切成每个 head 的小矩阵分别投影；工程上合成大矩阵乘法更快、更容易并行。
- 不能把输入 `d_model` 先硬切给不同 head 再各自投影，因为那会让每个 head 只能看输入特征的一部分，表达能力弱于每个 head 都从完整 `d_model` 中学习自己的子空间。
- 多头输出拼接后再过 `W_O` 不是单纯为了改 shape；拼接只是把各 head 并排放好，输出投影负责学习如何重组、混合、旋转不同 head 的信息，再送回 residual stream。
- 手撕 MHA 时按流程检查：Q/K/V 投影后拆成 `(batch, heads, seq, d_head)`，RoPE 只作用在 Q/K，causal mask 要能广播到 `(batch, heads, seq, seq)`，合并 heads 后必须回到 `(batch, seq, d_model)` 并经过 `W_O`。
- RoPE 手撕重点不是背 API，而是掌握：把最后一维两两成对做二维旋转，只作用在 Q/K，不作用在 V；旋转角度由 token position 和频率共同决定，且 Q/K 点积会自然依赖相对位置差。
- MHA 的 `token_positions` 通常是为 RoPE 和增量推理准备的：完整训练序列可默认用 `0..s-1`，但 KV cache/分块续写时当前 chunk 的真实位置不一定从 0 开始，不能覆盖外部传入的位置。
- 增量生成时如果前面已有 100 个 token，当前只喂入 1 个新 token，则这个新 token 的真实 position 是 100，不是当前小张量里的 0；RoPE 若错用 0，会把相对距离关系整体算错。
- 绝对位置编码通常在 token embedding 后、进入 Transformer blocks 前加到 residual stream；RoPE 不是加到输入上，而是在每层 attention 内部对 Q/K 投影结果按位置做旋转。
- 绝对位置编码也可以间接进入 Q/K：先做 `x_i = token_emb_i + pos_emb_i`，再投影得到 Q/K/V；但如果直接把绝对位置向量加到 Q/K 上，attention score 会混入绝对位置和内容-位置交叉项，不像 RoPE 那样天然得到相对位置差。
- 绝对位置编码加在 token embedding 后，是因为 token id 先要被映射到 `d_model` 向量空间，才能和同维度的位置向量相加；它直接告诉模型“当前位置是 i”，相对关系 `i-j` 需要模型后续自己学出来。
- 相对位置编码/RoPE 直接把位置差或会导出位置差的结构放进 attention score；RoPE 中位置 `i` 和 `j` 的 Q/K 旋转点积会自然依赖 `i-j`，所以相对信息是机制自带的。
- `einops.rearrange` 不能无故丢掉输入轴；`b h s d -> b s (h d)` 或 `... h s d -> ... s (h d)` 才能把多头输出合回 `d_model`，同时保留 batch 维。
- `einops` 的省略号必须用英文三个点 `...`，不要用 Unicode 单字符省略号 `…`；后者会被当成普通轴名，容易触发 “Identifiers only on one side”。

### LLM / Alignment

- “对齐”不是单个固定术语，核心问题永远是：让什么对象 A 按什么标准接近对象 B；常见标准可以是 shape 匹配、token 位置匹配、向量空间相似、人类偏好、安全规则或训练/推理分布一致。
- 学到一个新“对齐”时先问三件事：谁和谁对齐、用什么损失或约束衡量、对齐失败会出现什么现象。

### Transformer / FFN

- SwiGLU 可以理解成带门控的 FFN：一条分支产生候选内容，另一条分支经过 SiLU 产生连续门值，再逐元素相乘，决定哪些特征应该通过、放大或压低。
- 相比普通 ReLU/GELU FFN，SwiGLU 的门不是硬 0/1，而是平滑可学习的特征选择；LLM 中常用它提升表达能力，但通常会调整隐藏维度以保持参数量接近。

### PyTorch / Modules

- `nn.Linear` 通常来自 `from torch import nn`，本质就是 `torch.nn.Linear`；公开 API 里一般没有 `torch.Linear` 这种写法。
- `nn.Linear` 是带参数的 Module，会保存 `weight`/`bias` 为 `nn.Parameter`；`torch.nn.functional.linear` 是无状态函数，需要外部传入 weight 和 bias。
- `import torch.nn as nn` 或 `from torch import nn` 之后要写 `nn.Linear`；只有 `from torch.nn import Linear` 才能直接写 `Linear`。

### PyTorch / DTypes

- `bool` 是 Python 标量真假类型，表示一个单独的 `True/False`；`torch.bool` 是 PyTorch tensor 的元素 dtype，表示这个张量里每个元素都是布尔值。
- 创建 mask tensor 时用 `dtype=torch.bool`，因为 attention mask 需要的是一整张布尔矩阵；普通 `bool` 更常用于 `if flag:` 这种单个开关。
- `torch.arange(0, d, 2)` 在边界都是整数时默认产生整数 tensor；如果这些数后面要当作 RoPE 频率公式里的指数/比例参与除法、幂、sin/cos，就应转成 float 或直接指定 `dtype=torch.float32`。

### PyTorch / Optimizer

- 优化器 `step()` 不负责算梯度，梯度来自 `loss.backward()` 后每个参数的 `p.grad`；`step()` 只是在 `torch.no_grad()` 下读取梯度并原地更新参数值。
- AdamW 会给每个参数维护自己的状态：`step`、一阶矩 `exp_avg` 和二阶矩 `exp_avg_sq`；这些状态保存在 `optimizer.state[p]`，不是模型参数本身。
- `state["exp_avg"] = torch.zeros_like(p)` 是给当前参数创建同 shape/dtype/device 的零张量，用来存 Adam 的一阶矩；`exp_avg_sq` 同理存梯度平方的二阶矩。
- `memory_format=torch.preserve_format` 表示尽量保留参数原来的内存布局，避免状态张量和参数布局不一致。
- AdamW 的核心更新分两类：先用梯度的一阶/二阶滑动平均做自适应步长，再做解耦 weight decay；具体先后顺序要以课程 handout/测试约定为准。
- PyTorch 里带下划线的张量方法会原地修改自身：`mul_` 乘、`add_` 加、`addcmul_` 加乘积、`addcdiv_` 加商；优化器常用这些 in-place 操作直接更新 moment buffer 和参数。
- SGD 中 L2 正则和 weight decay 等价，因为 `g + lambda*w` 直接导致 `w <- (1-lr*lambda)w - lr*g`；Adam 中不等价，因为 `lambda*w` 会进入一阶/二阶矩并被自适应分母缩放，导致不同参数维度衰减强度不一致。
- AdamW 的 “W” 是 decoupled weight decay：不要把 `lambda*w` 混进梯度和 Adam 的 m/v 统计，而是在 Adam 梯度更新之外单独对参数做乘法收缩。
- Adam 的“耦合灾难”推导从 L2 后梯度 `g_t + lambda*theta_t` 开始；一旦把它作为 Adam 输入，`lambda*theta_t` 会同时进入 `m_t` 和 `v_t`，所以衰减项被历史动量和自适应分母共同改变，无法整理成简单的 `(1-lr*lambda)theta`。

### ML / Regularization

- L1/L2 正则是在原 loss 外加一个“参数不要太大”的惩罚项：L1 惩罚绝对值和，L2 惩罚平方和；它们不是让训练集 loss 更低，而是用偏好约束换更好的泛化。
- L2 的梯度与参数大小成正比，会平滑地把所有权重往 0 拉；L1 的梯度大小基本恒定且在 0 处有尖点，因此更容易把不重要权重压到精确 0，产生稀疏性。
- `||theta||_2` 是欧几里得长度，来自勾股定理：二维是 `sqrt(theta_1^2 + theta_2^2)`，推广到 n 维就是 `sqrt(sum theta_i^2)`；L2 正则常用平方长度 `||theta||_2^2`，所以惩罚项是平方和。
- L1 不是绝对不如 L2：它适合特征选择和稀疏模型；但在深度网络/LLM 里，L2/weight decay 更常用，因为惩罚是平滑的、优化更稳定、不会强行把大量权重压成精确 0，且密集硬件上稀疏参数通常不自动带来速度收益。

### LLM / LoRA

- LoRA 不是重新训练完整权重 `W`，而是冻结 `W`，只学习一个低秩更新 `Delta W = B A`；若 `W` 是 `(d_out, d_in)`，则 LoRA 参数量是 `r*d_in + d_out*r`，通常远小于 `d_out*d_in`。
- LoRA 会让总存储多出少量 adapter 参数，但训练时可训练参数和优化器状态大幅减少；推理时还能把 `Delta W` 合并进原权重，变成一个普通线性层。

### CS336 Handout / 彩框

- Assignment 1 PDF 里蓝框主要是低资源建议：先 profiling/downscaling，用 CPU/MPS 时注意 device、40M token 小训练、MPS 不开 TF32，可用 `torch.compile`，GPU 少时先在 TinyStories 做架构实验。
- 橙黄框是 Problem 任务框，覆盖 tokenizer、Transformer 组件、loss/optimizer、training loop、decoding、实验和 leaderboard；它们是任务规格和交付要求，不是概念提示。

### LM / Interactive Decoding

- 终端交互续写本质是 REPL：读取用户 prompt，tokenizer 编码，模型自回归生成若干 token，tokenizer 解码后打印，再等待下一轮输入。
- CS336 里的 LM 是 next-token continuation 模型，不是 instruction/chat 模型；想“像对话”需要人为维护文本历史和角色分隔，但模型未必真的学会聊天。

### PyTorch / Broadcasting

- `keepdim=True` 会在 `mean/var/sum` 这类 reduce 操作后保留被约掉的维度，长度变成 1；例如 `(B, S, D)` 沿 `dim=-1` 求均值后得到 `(B, S, 1)`。
- LayerNorm/RMSNorm 里保留最后一维是为了后续 `(x - mean) / sqrt(var + eps)` 能沿特征维正确广播；如果变成 `(B, S)`，通常无法和 `(B, S, D)` 对齐。
- `torch.arange(s)` 得到形状 `(S,)` 的位置序列；`.unsqueeze(0)` 把它变成 `(1, S)`，再 `.expand(B, S)` 逻辑扩展成每个 batch 共用的 token positions。
- `torch.gather(logits, dim=-1, index=targets.unsqueeze(-1))` 用 target id 在 vocab 维取正确类别 logit；`unsqueeze(-1)` 把 index 变成和 logits 同 rank，`squeeze(-1)` 再去掉取值后多出来的最后一维。
