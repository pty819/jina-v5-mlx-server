# PRD: 用 Jina Reranker v3 (官方 MLX) 替换 Prism reranker

- 日期: 2026-06-26
- 状态: 设计已与用户对齐,进入实现
- 范围: rerank 能力后端替换;embedding/chat 路径不动

## 1. 背景与动机

当前 `reranking.py` 的 `PrismMLXReranker` 用 pointwise yes/no logit 打分,每篇文档独立 forward
一次,文档之间互不感知。质量不满足需求。

官方 [`jinaai/jina-reranker-v3-mlx`](https://huggingface.co/jinaai/jina-reranker-v3-mlx) 是 0.6B
参数的 **listwise** reranker:所有候选文档拼进同一个 prompt(每篇末尾接 `<|embed_token|>`,
query 末尾接 `<|rerank_token|>`),一次 forward 取这些特殊 token 位置的 hidden state,
经 MLP 投影后做 cosine 相似度。文档在同一上下文窗口内经 causal self-attention 互相参照,
rank 质量显著优于 pointwise。

用户决策(已确认):
- **彻底移除 Prism**,不留 fallback,不做目录嗅探双分支。
- **官方 fp16 版本**(`jinaai/jina-reranker-v3-mlx`,~1.2 GB),不用 4bit 量化版。
- **不做文档截断**(实际遇不到超长文档)。
- **不分窗**(给几个文档就一次 forward 几个)。
- **不暴露 embedding**:`return_embeddings` 字段保留以兼容请求体,但恒返回 `null`;
  投影后的 512 维向量仅用于内部算 cosine,用完即弃。
- **不支持 instruction 参数**(官方 prompt 模板里有,我们不用,省略 `<instruct>` 段)。

许可证注意:Jina v3 权重为 CC-BY-NC-4.0(非商用)。用户已知悉。

## 2. 加载机制(已验证)

官方 `rerank.py` 自身就用 `mlx_lm.load()` 读 backbone:

```python
from mlx_lm import load
self.model, self.tokenizer = load(model_path)
```

可行性根因在仓库 `config.json` 的字段分工:
- `model_type: "qwen3"` ← mlx-lm 按此路由成标准 Qwen3 backbone
- `architectures: ["JinaForRanking"]` 与 `auto_map` ← 只对 HF transformers 有意义,mlx-lm 忽略

因此不需要 `trust_remote_code`,不需要 vendor 目录,不需要仓库里的 `modeling.py`。
只需要 `model.safetensors`(backbone) + `projector.safetensors`(3 MB 投影头),
外加 tokenizer/config,`mlx_lm.load()` 一次全拿。

### 调用约定(必须原样保留)

官方取 hidden state 用 `self.model.model([input_ids])` —— 注意是 **list 包一个 list**
(模拟 batch=1),返回 `[1, seq_len, hidden_size]`,再 `[0]` 去 batch 维。这个调用约定
必须原样照抄,否则 hidden state 形状对不上。

### 特殊 token id

官方硬编码 `doc_embed_token_id = 151670`、`query_embed_token_id = 151671`。
我们改为从 tokenizer 反查:
```python
self._query_token_id = self.tokenizer.encode("<|rerank_token|>", add_special_tokens=False)[0]
self._doc_token_id   = self.tokenizer.encode("<|embed_token|>",   add_special_tokens=False)[0]
```
防止 tokenizer 配置变动导致静默错位,代价仅两次一次性 encode。

## 3. 架构与文件边界

### `reranking.py`(重写)

```
常量:
  DEFAULT_JINA_RERANKER_REPO_ID = "jinaai/jina-reranker-v3-mlx"
  DEFAULT_RERANKER_DIR          = models/jinaai/jina-reranker-v3-mlx
  RERANK_MODEL_ID               = DEFAULT_JINA_RERANKER_REPO_ID
  DEFAULT_IDLE_SECONDS          = 20 * 60   (不变)

dataclass(不变):
  RerankResult(index, relevance_score, document=None, embedding=None)
  RerankResponse(results, total_tokens)

辅助函数:
  _normalize_embedding(value)         (不变,保留)
  _clear_mlx_cache()                  (不变,保留)

新增(内联官方 rerank.py 逻辑,不改算法):
  class MLPProjector(nn.Module)       linear1=1024→512, linear2=512→512, 无 bias
  _load_projector(path) -> MLPProjector   safe_open 读 linear1/linear2.weight
  _format_listwise_prompt(query, docs, *, special_tokens, no_thinking=True) -> str
  _gather_query_hidden(hidden, input_ids, token_id) -> mx.array   [hidden_size]
  _gather_doc_hiddens(hidden, input_ids, token_id) -> mx.array    [num_docs, hidden_size]
  _cosine_scores(query_emb, doc_embs) -> mx.array  [num_docs]

class JinaV3Reranker:                  (替代 PrismMLXReranker)
  __init__(model_dir): load backbone + projector + 反查 token id
  rerank(query, documents, *, top_n=None, return_embeddings=False) -> list[dict]
    - 拼官方 listwise prompt(no_thinking=True,无 instruction)
    - self.model.model([input_ids])[0] 取 hidden
    - 投影 query/doc → cosine scores → mx.eval(scores)
    - doc_embs 不 eval、不返回
    - 组装 list[dict],每项 {document, relevance_score, index, embedding: None}
    - 按 relevance_score 降序,top_n 截断

class MLXRerankService:                (改名自 OfficialMLXRerankService)
  对外接口签名不变: rerank(query, documents, *, top_n, return_embeddings) -> RerankResponse
  _load 简化为单分支: 目录存在性检查 → JinaV3Reranker(...) → 装 token_counter → evictor.start
  model_id = RERANK_MODEL_ID (硬编码,删掉 _model_id_for_dir)
  其余(_evict_reranker / _count_tokens / IdleEvictor 接线 / clear_cache_after_inference)不变

删除:
  PrismMLXReranker
  _is_prism_reranker_dir
  _model_id_for_dir
  PRISM_RERANK_MODEL_ID
  DEFAULT_PRISM_RERANKER_REPO_ID
```

### 其他源码触点

- `main.py`: `from ... import OfficialMLXRerankService` → `MLXRerankService`(2 处:import + run_serve 内构造)。
- `schema.py`: `VALID_RERANK_MODELS` 改为 Jina v3 别名集合(`jinaai/jina-reranker-v3-mlx` + 短别名)。`ensure_rerank_model` 逻辑不变。
- `routes.py` / `rerank_queue.py` / `server.py`: **不动**(服务接口签名不变)。

### 测试触点

- `tests/test_reranking.py`: 重写。删除 `ScoredPrismReranker` / `FakePrismTokenizer` 等 Prism 专用 fake;
  保留 `OfficialMLXRerankService` → `MLXRerankService` 的服务层测试(fake reranker 注入)。
  新增针对 `_format_listwise_prompt` / `_cosine_scores` / `_gather_*` 的纯函数单测(不依赖真实权重)。
- `tests/test_model_lifecycle.py`: import 名改 `MLXRerankService`。
- `tests/test_main.py`: `patch("main.OfficialMLXRerankService")` → `patch("main.MLXRerankService")`。
- `tests/test_server.py`: 把 `pty819/prism-qwen3.5-reranker-0.8b-optiq-5bpw-cal24` 字符串
  全部替换为 `jinaai/jina-reranker-v3-mlx`(FakeRerankService.model_id 及所有断言)。

### 运维触点

- `launchd/start-jina-gateway.sh`: `--reranker-dir` 路径
  `models/pty819/prism-qwen3.5-reranker-0.8b-optiq-5bpw-cal24` → `models/jinaai/jina-reranker-v3-mlx`。
- `README.md`: 下载命令、模型描述、`return_embeddings` 说明、rerank 响应示例里的 model id、
  OpenViking 配置示例里的 model、default reranker directory 一句。

### 历史文档(不改)

`.omx/plans/prism-*.md`、`.omx/context/prism-*.md`、
`docs/superpowers/plans/2026-05-22-reranker-stats-serving.md` 是历史快照,不修改。

## 4. 行为变化清单

| 项 | Prism(旧) | Jina v3(新) |
|---|---|---|
| 打分范式 | pointwise yes/no,逐文档独立 forward | listwise,所有文档同 prompt 一次 forward |
| 分数范围 | sigmoid `[0,1]` | cosine `[-1,1]` |
| `return_embeddings` | 恒 `null` | **恒 `null`**(投影向量内部用完即弃,不暴露) |
| 模型 id | `pty819/prism-...` | `jinaai/jina-reranker-v3-mlx` |
| 权重大小 | ~630 MB | ~1.2 GB |
| 许可证 | 社区量化 | CC-BY-NC-4.0 |
| 文档截断 | 有(`max_doc_length=2048`) | **无** |
| 分窗 | 无(逐文档循环) | **无**(一次 forward 全量) |
| OpenViking threshold 语义 | sigmoid 概率 | cosine 相似度(数值意义不同,默认 0.1 仍可用,需用户知晓) |

## 5. 非目标(YAGNI)

- 不做文档截断 / 分窗 / max_docs 限制
- 不暴露 doc embedding
- 不支持 instruction 参数
- 不做 4bit 量化支持(路径可扩展:换 `--reranker-dir` 指向 4bit repo 即可,无需改代码,
  因为加载逻辑只认 `model.safetensors`+`projector.safetensors`,但本次不实现也不测)
- 不改 embedding / chat 任何路径

## 6. 验收标准(自验证 loop)

1. `uv run pytest -q` 全绿(含重写的 reranking 测试 + 更新后的 server/main/lifecycle 测试)。
2. `uv run ruff check .` 无新增告警。
3. 静态自检:
   - `grep -rn "Prism\|prism\|pty819\|OfficialMLXRerankService"` 在 `jina_v5_mlx_demo/`、
     `tests/`、`launchd/`、`README.md`、`main.py` 内 **0 命中**(历史 .omx 文档除外)。
   - `JinaV3Reranker` / `MLXRerankService` 在 `reranking.py` 内定义,
     被 `main.py`、`test_reranking.py`、`test_model_lifecycle.py`、`test_main.py` 正确引用。
4. 真实权重 smoke test(若本地权重已下载):
   `uv run python -c "from jina_v5_mlx_demo.reranking import MLXRerankService; s=MLXRerankService(); print(s.rerank('What is MLX?', ['MLX runs on Apple silicon.', 'A sourdough starter is for baking.'], top_n=1))"`
   预期: 第一篇排前,relevance_score 为 `[-1,1]` 内浮点,total_tokens > 0。
   若权重未下载: 跳过此步并在交付报告注明。

## 7. 交付步骤顺序

1. `reranking.py` 重写(常量 + dataclass 保留 + JinaV3Reranker + MLXRerankService + 内联辅助)
2. `schema.py` VALID_RERANK_MODELS 更新
3. `main.py` import 名更新
4. `launchd/start-jina-gateway.sh` 路径更新
5. `README.md` 更新
6. `tests/test_reranking.py` 重写
7. `tests/test_server.py` 字符串替换
8. `tests/test_model_lifecycle.py` import 名更新
9. `tests/test_main.py` import 名更新
10. 自验证 loop: pytest + ruff + grep 静态检查 + (条件)真实权重 smoke
