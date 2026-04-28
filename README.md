# Neural Memory Visualizer

基于 D3.js 力导向图的 **neural-memory** 记忆系统可视化面板。通过 MCP（Model Context Protocol）协议与 neural-memory 服务端通信，以图形化方式展示神经元（Neurons）和突触（Synapses）网络，并支持在线增删记忆、触发记忆维护等操作。

![stack](https://img.shields.io/badge/Node.js-Express-green) ![d3](https://img.shields.io/badge/D3.js-v7-orange) ![mcp](https://img.shields.io/badge/MCP-stdio-blue)

## 架构

```
┌──────────────────┐  HTTP/JSON   ┌──────────────────┐  MCP/stdio   ┌────────────────────┐
│  Browser (D3.js) │ ←──────────→ │  Express Server  │ ←──────────→ │  neural-memory MCP │
│  public/         │  :3456       │  server.js       │              │  (Python/FastMCP)  │
└──────────────────┘              └──────────────────┘              └────────────────────┘
                                                                            │
                                                                    ┌───────┴────────┐
                                                                    │  SQLite + FTS5  │
                                                                    │  + Embeddings   │
                                                                    └────────────────┘
```

- **浏览器端** — 纯前端，D3.js 力导向图渲染，无框架依赖
- **Express 中间层** — Node.js 服务，作为 MCP Client 通过 stdio 连接 Python MCP Server，对外暴露 REST API
- **neural-memory MCP Server** — Python 进程，管理记忆的存储、检索、遗忘、关联等全部逻辑

> **所有数据流均通过 MCP 工具调用**，不直接访问数据库。

## 功能

### 图形化展示
- **力导向图** — 神经元为节点，突触为连线，支持拖拽、缩放、平移
- **形状编码** — ⚪ Working / ◇ Short-term / ▢ Long-term
- **颜色编码** — 🔵 Semantic / 🟠 Episodic / 🟢 Procedural / 🟣 Schema
- **连线颜色** — 区分 semantic / temporal / causal / hierarchical / reference 五种突触类型
- **节点大小** — 由 importance 和 strength 共同决定
- **悬停提示** — 显示内容摘要、类型、层级、强度等关键指标
- **点击聚焦** — 高亮选中节点及其关联节点，淡化其余

### 详情面板（右侧）
- 完整内容、ID、类型、层级
- Strength / Stability / Importance
- 情感维度（valence / arousal）
- 访问次数、标签、来源、创建/访问时间
- 删除按钮

### 操作
- **Refresh** — 重新加载图数据
- **Reflect** — 触发记忆维护周期（遗忘曲线衰减、弱记忆修剪、短期→长期提升、Schema 合并、突触衰减）
- **+ Remember** — 弹窗创建新记忆（支持设定类型、重要度、标签）
- **Delete** — 删除选中的神经元

## 快速开始

### 前置条件

1. **Node.js** ≥ 18
2. **Python** ≥ 3.10，且已安装 [neural-memory](https://github.com/sunfengsheng/neural-memory) 插件
3. neural-memory 的嵌入模型已下载（`paraphrase-multilingual-MiniLM-L12-v2`）

### 安装

```bash
cd d:/code/test/human_mem
npm install
```

### 配置

编辑 `server.js` 中的路径常量，指向你的本地环境：

```js
// neural-memory 插件根目录
const PLUGIN_ROOT = path.resolve("C:/Users/<你的用户名>/.claude/plugins/marketplaces/neural-memory");

// 记忆数据库存储目录
const STORAGE_DIR = path.resolve("C:/Users/<你的用户名>/.neural-memory/storage");

// 服务端口
const PORT = 3456;
```

### 启动

```bash
npm start
```

启动后访问 **http://localhost:3456**

控制台应显示：
```
[MCP] connected, capabilities: {...}

  Memory Visualizer running at http://localhost:3456
```

## 项目结构

```
human_mem/
├── server.js          # Express + MCP Client，桥接浏览器与 neural-memory
├── package.json       # 项目依赖
├── public/
│   ├── index.html     # 单页应用入口
│   ├── app.js         # D3.js 图渲染 + 交互逻辑
│   └── style.css      # 暗色主题样式
└── README.md
```

## REST API

Express 服务对外暴露以下接口，均代理到 MCP 工具调用：

| 方法 | 路径 | MCP 工具 | 说明 |
|------|------|----------|------|
| GET | `/api/graph` | `list_graph` | 获取全部神经元（不含 embedding）和突触 |
| GET | `/api/status` | `memory_status` | 获取记忆系统统计信息 |
| GET | `/api/recall?query=...&top_k=20` | `recall` | 语义检索记忆 |
| POST | `/api/remember` | `remember` | 创建新记忆 |
| POST | `/api/forget` | `forget` | 删除记忆 |
| POST | `/api/associate` | `associate` | 创建神经元间突触连接 |
| POST | `/api/reflect` | `reflect` | 触发记忆维护周期 |

## 技术栈

- **前端**: 原生 HTML/CSS/JS + [D3.js v7](https://d3js.org/)
- **后端**: [Express 5](https://expressjs.com/) (Node.js)
- **MCP Client**: [@modelcontextprotocol/sdk](https://www.npmjs.com/package/@modelcontextprotocol/sdk) (stdio transport)
- **MCP Server**: [neural-memory](https://github.com/sunfengsheng/neural-memory) (Python/FastMCP)

## License

MIT
