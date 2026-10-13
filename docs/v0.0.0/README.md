# 文档导航

`docs/` 用于集中存放项目的架构说明、版本记录和问题记录。

## 当前结构

- `architecture/`
  - PlantUML 源文件与导出的架构图
- `v0.1.0/`
  - `v0.1.0` 阶段记录与已知问题
- `ISSUES.md`
  - 跨版本问题清单
- `workflow.png`
  - 当前知识库主流程示意图

## 推荐阅读顺序

1. [项目总览](../README.md)
2. [部署指南](../DEPLOY_GUIDE.md)
3. [架构图目录](./architecture/)
4. [v0.1.0 记录](./v0.1.0/README.md)

## 说明

- 当前主流程为 `query -> hyde -> retrieve -> rerank -> repack -> compress`
- `generate` 需要额外接入语言模型，本项目当前不提供 `generate`
- 架构图片统一放在 `docs/architecture/` 下，其他文档请使用相对路径引用
