# Zotero Crypto Paper Daily

每天根据你的 Zotero 文献库，自动从 arXiv 和 IACR ePrint 拉取近期论文，按你的研究兴趣排序，并把推荐结果发到邮箱。

这个仓库适合密码学、安全、隐私保护方向使用，默认偏向：

- 应用密码学、新应用场景、新威胁模型
- 新密码学原语、协议抽象、安全定义
- PAKE、匿名凭证、盲签名、零知识证明、MPC、PIR、PSI、后量子协议
- 隐私保护系统、认证、钱包、密钥管理、安全消息、加密数据库、隐私保护 AI

## 来源与许可

本项目基于 [TideDra/zotero-arxiv-daily](https://github.com/TideDra/zotero-arxiv-daily) 修改，继承原项目的 AGPLv3 许可证。原始项目主要面向 Zotero + arXiv 每日邮件推荐；本仓库在此基础上加入了更适合密码学预印本跟踪的配置和功能。详细来源与改动见 [NOTICE.md](./NOTICE.md)。

不要把 Zotero API Key、邮箱授权码、LLM API Key 写进公开代码。请使用 GitHub Secrets。

## 主要改动

- 支持 `arXiv + IACR ePrint` 混合检索。
- 默认拉取最近 `10` 天论文，降低周末和 ePrint 发布节奏造成的漏报。
- 邮件最多输出 `30` 篇，并默认保证 arXiv 至少 `10` 篇、ePrint 至少 `10` 篇。
- 每篇论文显示 `Source`，arXiv/ePrint 有作者备注时会显示在来源后面。
- 使用大模型生成中文标题翻译和更完整的中文 TLDR。
- 增加显式兴趣画像，让排序更偏向应用密码学、新场景、新原语和隐私保护密码协议。
- 可选使用 OpenAlex 的论文引用量、作者影响力、发表质量信号做轻量加权。
- 可选邮件反馈按钮，点击“推送满意/不太满意”后可用于后续排序偏好。

## 快速部署

### 1. Fork 仓库

点击 GitHub 右上角 `Fork`，把仓库 fork 到你自己的账号。

Fork 后建议检查：

- `Settings -> Actions -> General`：确认 Actions 没有被禁用。
- `Actions` 页面：如果出现启用提示，点击启用 workflows。
- 默认定时任务是 `0 22 * * *`，也就是 UTC 22:00，北京时间每天 06:00。

### 2. 准备 Zotero

进入 Zotero 的 API 设置页：

[https://www.zotero.org/settings/security](https://www.zotero.org/settings/security)

需要两个值：

- `ZOTERO_ID`：页面上的数字 User ID，不是用户名。
- `ZOTERO_KEY`：新建一个 Zotero API Key，至少需要 library read 权限。

### 3. 准备发件邮箱

默认配置使用 163 邮箱 SMTP：

- SMTP server: `smtp.163.com`
- SMTP port: `465`

`SENDER_PASSWORD` 应填写邮箱 SMTP 授权码，不是网页登录密码。163 邮箱需要先在邮箱设置里开启 SMTP/IMAP 服务并生成授权码。

如果你使用 Gmail、QQ、Outlook 或学校邮箱，见下面的 `CUSTOM_CONFIG` 示例修改 `email.smtp_server` 和 `email.smtp_port`。

### 4. 准备大模型 API

默认示例使用硅基流动兼容 OpenAI 的接口：

- `OPENAI_API_BASE`: `https://api.siliconflow.cn/v1`
- `OPENAI_API_KEY`: 你的 API Key
- 默认摘要模型: `Qwen/Qwen3-8B`
- 默认 embedding/rerank 模型: `BAAI/bge-m3`

其它 OpenAI-compatible 服务也可以，只要支持 chat completions 和 embeddings。

### 5. 配置 GitHub Secrets

在 fork 后的仓库中进入：

`Settings -> Secrets and variables -> Actions -> Secrets`

新增以下 Secrets：

| Secret | 说明 |
| --- | --- |
| `ZOTERO_ID` | Zotero 数字 User ID |
| `ZOTERO_KEY` | Zotero API Key |
| `SENDER` | 发件邮箱，例如 `yourname@163.com` |
| `RECEIVER` | 收件邮箱 |
| `SENDER_PASSWORD` | 发件邮箱 SMTP 授权码 |
| `OPENAI_API_KEY` | 大模型服务 API Key |
| `OPENAI_API_BASE` | OpenAI-compatible API 地址 |

默认提交的 [config/custom.yaml](./config/custom.yaml) 已经引用这些 Secrets。使用 163 邮箱和硅基流动时，通常只需要填 Secrets。

### 6. 手动触发一次

进入 `Actions -> Test -> Run workflow`。

这个 workflow 会设置 `DEBUG=true`，用于手动测试邮件发送。运行结束后检查：

- GitHub Actions 日志是否显示 `Email sent successfully`
- 收件箱、垃圾邮件、广告邮件目录
- 163 邮箱是否拦截 SMTP 登录

### 7. 每天自动发送

主 workflow 是 `Send emails daily`：

```yaml
schedule:
  - cron: '0 22 * * *'
```

这表示北京时间每天 06:00 左右运行。GitHub Actions 的 schedule 不是精确闹钟，可能会延迟几分钟到几十分钟。

## 自定义配置

如果你想调整邮箱服务、模型、论文来源、分类、兴趣画像，可以在：

`Settings -> Secrets and variables -> Actions -> Variables`

新增变量 `CUSTOM_CONFIG`，填入完整 YAML。workflow 检测到 `CUSTOM_CONFIG` 后会覆盖仓库里的 `config/custom.yaml`。

下面是一个可直接改的模板：

```yaml
zotero:
  user_id: ${oc.env:ZOTERO_ID}
  api_key: ${oc.env:ZOTERO_KEY}
  include_path: null
  ignore_path: null

email:
  sender: ${oc.env:SENDER}
  receiver: ${oc.env:RECEIVER}
  smtp_server: smtp.163.com
  smtp_port: 465
  sender_password: ${oc.env:SENDER_PASSWORD}

llm:
  api:
    key: ${oc.env:OPENAI_API_KEY}
    base_url: ${oc.env:OPENAI_API_BASE}
  generation_kwargs:
    model: Qwen/Qwen3-8B
    extra_body:
      enable_thinking: false
  language: Chinese
  tldr:
    max_sentences: 3
    max_words: 160

source:
  arxiv:
    category: ["cs.CR"]
    include_cross_list: true
    days_back: 10
  eprint:
    category: [
      "cryptography",
      "cryptographic protocols",
      "public-key cryptography",
      "secret-key cryptography",
      "privacy",
      "private information retrieval",
      "password-authenticated key exchange",
      "PAKE",
      "zero-knowledge",
      "oblivious transfer",
      "private set intersection",
      "applied cryptography",
      "post-quantum cryptography"
    ]
    days_back: 10

executor:
  debug: ${oc.env:DEBUG,null}
  source: ['arxiv','eprint']
  send_empty: true
  max_paper_num: 30
  source_min_papers:
    arxiv: 10
    eprint: 10
  reranker: api
  quality_boost: true
  citation_weight: 0.15
  author_weight: 0.15
  venue_weight: 0.05
  interest_profile_weight: 0.9
  interest_profile:
    - "Applied cryptography papers that introduce new real-world scenarios, deployment models, threat models, or security goals."
    - "New cryptographic primitives, constructions, or protocol abstractions."
    - "Privacy-preserving cryptographic protocols for emerging applications."

reranker:
  api:
    key: ${oc.env:OPENAI_API_KEY}
    base_url: ${oc.env:OPENAI_API_BASE}
    model: BAAI/bge-m3
    batch_size: 32

feedback:
  enabled: false
  endpoint: null
  history_path: null
  history_max_items: 200
  positive_weight: 0.0
  negative_weight: 0.0
```

### 常用配置规则

- `source.arxiv.category`: arXiv 分类。密码学通常用 `["cs.CR"]`。
- `source.arxiv.include_cross_list`: 是否包含 cross-list 到 `cs.CR` 的论文。建议密码学方向设为 `true`。
- `source.eprint.category`: ePrint 的关键词/分类匹配列表。匹配逻辑会做大小写归一化和包含匹配。
- `days_back`: 拉取最近 N 天，默认 `10`。
- `max_paper_num`: 邮件最多展示多少篇，默认 `30`。
- `source_min_papers`: 混排后每个来源至少保留多少篇。候选不足时不会强行补。
- `interest_profile`: 直接告诉 reranker 你偏好的论文类型。
- `quality_boost`: 是否用 OpenAlex 查引用量和作者质量信号。网络偶发失败时会跳过，不影响邮件发送。

## 反馈按钮

默认关闭：

```yaml
feedback:
  enabled: false
```

如果你有自己的公网 webhook，可以开启：

```yaml
feedback:
  enabled: true
  endpoint: https://your-domain.example/feedback
  history_path: /absolute/path/to/feedback.csv
  positive_weight: 0.8
  negative_weight: 0.4
```

邮件中的按钮会向 `endpoint` 发送 GET 请求，参数包括：

- `action`: `liked` 或 `dislike`
- `source`: `arxiv` 或 `eprint`
- `title`: 论文标题
- `paper_url`: 论文链接

仓库提供了一个最小收集服务：

```bash
python feedback_server.py \
  --host 0.0.0.0 \
  --port 8080 \
  --out /absolute/path/to/feedback.csv \
  --retention-days 30
```

注意：GitHub-hosted Actions 的文件系统是临时的，`history_path` 只有在你使用持久化服务、自托管 runner，或把反馈 CSV 放在可持久化的位置时才会跨天生效。

## 本地运行

推荐使用 `uv`：

```bash
uv sync --dev
uv run pytest -q
uv run src/zotero_arxiv_daily/main.py
```

本地运行前需要设置环境变量：

```bash
export ZOTERO_ID=...
export ZOTERO_KEY=...
export SENDER=...
export RECEIVER=...
export SENDER_PASSWORD=...
export OPENAI_API_KEY=...
export OPENAI_API_BASE=https://api.siliconflow.cn/v1
```

## 排错

- 没收到邮件：先看 Actions 日志是否 `Email sent successfully`，再查垃圾邮件和邮箱 SMTP 授权码。
- schedule 没跑：检查 fork 仓库 Actions 是否启用；默认北京时间 06:00，不是本地晚上。
- 论文数量不到 30：`30` 是上限，不是保证数量；候选论文不足或过滤后不足时会少于 30。
- ePrint 太多或太少：调整 `source.eprint.category` 和 `interest_profile`。
- TLDR 太短：调大 `llm.tldr.max_sentences` 和 `llm.tldr.max_words`。
- arXiv/ePrint 比例不合适：调整 `executor.source_min_papers`。

## License

AGPLv3. See [LICENSE](./LICENSE).
