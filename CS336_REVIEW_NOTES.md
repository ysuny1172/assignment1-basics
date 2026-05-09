# CS336 晚间回顾笔记

这个文件用于记录学习过程中反复容易忘的概念、调试思路和术语。它只保留回顾提示，不记录可直接提交的作业实现。

## 2026-05-09

### Tokenizer / BPE

- `str` 是 Python 中已经解码好的 Unicode 文本；`bytes` 是底层字节序列。硬盘上的文本最终存的是 bytes，Python 用 `encoding="utf-8"` 读取后才变成 `str`。
- `b"hi"` 是 bytes 对象，不是普通字符串；遍历 bytes 时得到的是整数，例如遍历 `b"hi"` 得到 `104, 105`。
- `bytes([b])` 用来把单个整数 byte 包装成长度为 1 的 bytes token；`bytes(b)` 的意思是创建长度为 `b`、内容全是 0 的 bytes。
- byte-level BPE 的真实工作单位是 bytes，不是字符，也不一定是词。一个 token 可能是完整英文词，也可能只是一个中文字符的一部分 bytes。
- UTF-8 中非英文字符常常是多字节：`é` 是 2 bytes，`牛` 是 3 bytes，emoji 常常是 4 bytes。如果生成或截断停在中间，直接 `decode("utf-8")` 会报 `UnicodeDecodeError`。
- 解码时更稳的心智模型：先把 token ids 查成 bytes，再把所有 bytes 拼起来，最后整体 UTF-8 decode。
- GPT-2 的 byte-to-unicode 映射不是 BPE 的本质，只是为了把任意 byte 安全写进文本格式的 `vocab.json` / `merges.txt`。内部训练、encode、decode 最好统一用 bytes。
- special token 在接口层通常是 `str`，例如 `"<|endoftext|>"`；加入 vocab 时要转成 UTF-8 bytes，例如 `b"<|endoftext|>"`。训练统计时它是硬边界，不参与普通 merge 统计。

### Pytest / Tests

- `pytest` 通过测试函数参数名自动寻找同名 fixture。`conftest.py` 是测试输入和工具的集中定义处。
- `Snapshot` 用 `.pkl` 比较普通 Python 对象，通常要求完全相等；`NumpySnapshot` 用 `.npz` 比较 tensor / numpy array，允许浮点误差。
- `test_tokenizer.py` 的 GPT-2 vocab/merges 来自 `tests/fixtures/gpt2_vocab.json` 和 `tests/fixtures/gpt2_merges.txt`，测试会先把 GPT-2 可打印 Unicode 表示还原成 bytes，再传给 `adapters.get_tokenizer`。
- Windows 没有 Unix 的 `resource` 标准库，所以 tokenizer 的完整测试更适合在 Linux / AutoDL 上跑。

### 环境 / 路径

- 相对路径是相对当前工作目录 `os.getcwd()`，不是相对 Python 文件本身。运行项目命令前先 `cd` 到项目根目录。
- `python script.py` 用哪个解释器，取决于当前 shell 里的 `python` 指向哪里，不取决于脚本在哪个项目里。
- PyCharm 的 Project Interpreter 和底部 Terminal 可能不是同一个环境。装包或跑测试前用 `python -c "import sys; print(sys.executable)"` 确认。
- 数据、checkpoint、logs 尽量放项目外，例如 `/root/autodl-tmp/data`，不要放进 Git 仓库里。

### Git

- `commit` 只提交暂存区里的内容，不是自动提交整个项目。`git add .` 会把当前目录下未被 `.gitignore` 忽略的改动加入暂存区。
- `.idea/`、`.venv/`、`data/`、`checkpoints/`、大模型权重和日志一般不应该提交。

### Profiling / Downscaling

- 优化前先 profiling，不要凭感觉猜。`cProfile` 可先看 `cumtime` 最大的函数，`py-spy` 适合长时间运行和火焰图。
- 先用小数据 debug，再逐步放大到 TinyStories train / OpenWebText。放大前要确认逻辑正确、速度可接受、内存不会炸。
