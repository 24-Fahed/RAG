# 熟人访问令牌

管理员在服务器为熟人生成一条令牌，通过已有渠道发给对方；对方在客户端保存一次即可。管理员可以按 ID 延长有效期，令牌字符串保持不变；也可以永久撤销。没有注册、登录或用户自行领令牌的接口。

这些令牌拥有现有的 `rag:api` 权限，**可以建库、查询和调用单轮回答，并非只读权限**；备注不构成账号、角色或数据隔离。`subject` 用来填写熟人称呼或用途，可以重复，管理操作以唯一 ID 为准。

## 管理员操作

安装 `server/requirements.txt` 后，在仓库根目录执行。管理员命令和 Server 必须使用同一个 SQLite 文件；路径必须为绝对路径，所在目录需要预先由运维分配给运行用户，并限制访问权限。

```bash
export RAG_AUTH_TOKEN_DB_FILE=/var/lib/rag/auth/tokens.sqlite3
python -m server.auth issue-token --subject "同学张三" --days 30
python -m server.auth list-tokens
python -m server.auth renew-token --id <ID> --days 30
python -m server.auth revoke-token --id <ID>
```

`issue-token` 首次执行会创建目录、数据库及表；不依赖 RSA 密钥。默认有效期 30 天，`--days` 接受 1–36500 的整数。创建时，标准输出只包含令牌，标准错误输出 ID、备注、UTC 到期时间和状态，方便单独复制或重定向令牌。令牌形如 `rag_pat_<ID>.<随机秘密>`。

`list-tokens` 只展示 `id`、`subject`、`expires_at` 和 `status`，状态为 `active`、`expired` 或 `revoked`。续期、撤销也只输出这些信息，不会再次展示令牌。服务端数据库仅保存 SHA-256 哈希及管理信息，无法找回原始秘密。令牌遗失时，按 ID 撤销旧条目，再生成新令牌并发送。

## 有效期和撤销

- 所有时间按 UTC 存储，命令输出采用带 `Z` 的 ISO 8601 时间。一天指连续 86400 秒，不按本地午夜结算。
- 创建到期时间为当前时间加指定天数；到期时刻及之后拒绝访问。
- 续期计算为 `max(当前时间, 原到期时间) + 指定天数 × 86400`。未到期的令牌追加天数，已过期但未撤销的令牌从当前时间恢复有效；使用者无需更换令牌文件。
- 撤销是终态，不能通过续期恢复，重复撤销也不会改变最初撤销时间。数据库保留撤销记录和过期记录，不自动清理。
- 每次请求重新读取记录；管理命令提交后，后续鉴权立即采用新状态，正在执行的业务请求不会被中断。SQLite 事务串行处理同时发生的续期和撤销。

## 熟人如何使用

将管理员发来的完整令牌保存为本地受限文件，例如 `access.token`，文件内只放令牌本身。CLI 的 `--token-file` 必须放在 `query` 或 `index` 子命令之前，其他业务参数继续按 OpenAPI 和 CLI 帮助显式填写：

```bash
python -m client.client --mode production --token-file /path/to/access.token query --help
```

直接调用 API 时使用 `Authorization: Bearer <token>`。Swagger `/docs` 的 Authorize 输入框只粘贴令牌，不添加 `Bearer` 前缀。续期无需修改客户端；收到 401 时请管理员检查有效期或撤销状态。403 表示权限不足，503 表示服务器认证配置或存储不可用。

## 持久化与容器配置

数据库是认证状态，必须存放在持久化目录。不要使用 `/run`、`/tmp` 或容器可写层保存它。现有 JWT 的 `/run/rag-auth` 公私钥挂载与这个数据库是不同配置。

部署 Server 时，额外配置以下容器参数，保持宿主机和容器内目录的所有者及权限与运行用户一致：

```text
--mount type=bind,src=/var/lib/rag/auth,dst=/var/lib/rag/auth
--env RAG_AUTH_TOKEN_DB_FILE=/var/lib/rag/auth/tokens.sqlite3
```

挂载整个目录，使 SQLite 事务文件也处于同一持久卷中。管理员在该容器执行 `python -m server.auth ...`，或在宿主机对挂载的同一个数据库操作。数据库按 SQLite 的一致性备份方式备份；它包含备注和哈希，不应发布到 Git。Linux 下新建目录权限为 0700，数据库文件为 0600；已有目录权限和 Windows ACL 由运行环境管理。

鉴权和列表命令不自动创建数据库。缺少配置、文件不可读、表缺失或数据库损坏都拒绝访问并返回 503；未知、错误、过期或撤销的令牌返回 401。当前方案适用于单台 Server 主机上的本地持久卷，不将 SQLite 文件放在跨主机网络文件系统上共享。

## 现有 JWT 兼容

以 `rag_pat_` 开头的令牌进入上述持久化验证路径；原有 RS256 JWT 继续使用 Authlib RFC 9068 验证，命令和密钥环境变量保持兼容：

```bash
python -m server.auth init-keys --directory /secure/rag-auth
export RAG_AUTH_PRIVATE_KEY_FILE=/secure/rag-auth/private.pem
export RAG_AUTH_PUBLIC_KEY_FILE=/secure/rag-auth/public.pem
python -m server.auth generate-token --subject legacy-client --expires-in 3600
```

已有部署继续使用现有密钥，不重新生成。JWT 路径不要求配置 SQLite，新令牌路径不要求配置 RSA。JWT 的签名到期时间仍然固定，不能用 `renew-token` 或 `revoke-token` 管理；新发熟人令牌使用 `issue-token`。两种令牌都使用相同 Bearer 请求头和现有权限。

## 方案依据与边界

持久化的不透明 API 令牌属于成熟模式：Laravel Sanctum 的 API Token 机制使用 Bearer 头，数据库仅保存 SHA-256 哈希并支持独立到期时间，和其 Cookie 会话机制分开。[Sanctum 官方文档](https://laravel.com/framework/docs/13.x/sanctum#issuing-api-tokens)

修改既有 PAT 有效期有实际产品先例；Azure DevOps 将编辑到期时间与重新生成秘密明确区分。不过它禁止修改已过期 PAT。本项目允许管理员续期已过期、未撤销令牌，是针对熟人场景制定的策略，不能等同于所有 PAT 产品的默认行为。[Azure DevOps 官方文档](https://learn.microsoft.com/en-us/azure/devops/organizations/accounts/use-personal-access-tokens-to-authenticate?view=azure-devops#modify-a-pat)

实现复用仓库现有 Authlib 1.8.0 的 `BearerTokenValidator` 查库、过期、撤销和 scope 验证接口；随机秘密使用 Python `secrets.token_urlsafe(32)`，管理存储使用标准库 SQLite，无新增认证框架或在线令牌服务。[Authlib 1.8.0 源码](https://github.com/authlib/authlib/blob/v1.8.0/authlib/oauth2/rfc6750/validator.py)、[Python secrets 文档](https://docs.python.org/3/library/secrets.html)、[SQLite 使用边界](https://www.sqlite.org/whentouse.html)
