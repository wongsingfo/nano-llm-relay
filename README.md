# nano-llm-api

一个极简的本地 LLM 代理服务，目标是用尽量少的依赖把 OpenAI 和 Anthropic 两套常见协议桥接起来，支持本地工具链通过统一入口去访问不同后端模型。

## 当前支持

- 入站协议：
  - `POST /v1/chat/completions`
  - `POST /v1/responses`
  - `POST /v1/messages`
- 出站协议：
  - `openai_chat`
  - `openai_responses`
  - `anthropic_messages`
- 其他接口：
  - `GET /v1/models`
  - `GET /healthz`
- 能力范围：
  - 文本请求与响应
  - 函数工具调用
  - SSE 流式转发与事件形状转换
  - YAML 配置热重载
  - 文件日志

## 非目标

- 数据库、鉴权后台、负载均衡、管理 UI
- 多模态块、MCP、自定义工具类型的完整兼容
- 维护服务端会话状态；`previous_response_id` 只会原样转发给 `openai_responses` 类型后端

## 安装

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

## 配置

从 [config.example.yaml](/home/chengke/Public/nano-llm-api/config.example.yaml) 开始，复制为 `config.yaml` 后修改模型映射和密钥来源。

```yaml
providers:
  anthropic:
    base_url: https://api.anthropic.com
    api_key_env: ANTHROPIC_API_KEY
    headers:
      anthropic-version: "2023-06-01"

models:
  claude-sonnet:
    provider: anthropic
    protocol: anthropic_messages
    target_model: claude-3-7-sonnet-latest
```

关键字段：

- `providers.<name>.base_url`: 上游根地址
- `providers.<name>.api_key` 或 `api_key_env`: 上游密钥
- `providers.<name>.auth_header` / `auth_prefix`: 自定义鉴权头，适配非标准 OpenAI 兼容后端
- `models.<alias>.protocol`: 目标协议类型
- `models.<alias>.target_model`: 实际发给上游的模型名
- `models.<alias>.extra_body`: 额外固定字段，会合并到上游请求体

## 启动

```bash
NANO_LLM_CONFIG=config.yaml .venv/bin/nano-llm-api
```

或直接：

```bash
NANO_LLM_CONFIG=config.yaml .venv/bin/python -m nano_llm_api
```

## 示例

OpenAI chat 客户端请求 Anthropic 后端：

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "claude-sonnet",
    "messages": [{"role": "user", "content": "Say hello"}]
  }'
```

Anthropic 客户端请求 OpenAI chat 后端：

```bash
curl http://127.0.0.1:8080/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{
    "model": "gpt-4o-chat",
    "max_tokens": 128,
    "messages": [{"role": "user", "content": "Say hello"}]
  }'
```

## 测试

```bash
.venv/bin/pytest
```

测试使用 `httpx.MockTransport`，不会访问外网。
