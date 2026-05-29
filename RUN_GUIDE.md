# RL-WDS Demo 运行指南

## 环境要求

- Windows 10/11
- Anaconda 或 Miniconda 已安装
- `rlwds` conda 环境已配置（包含所有依赖）

## 启动 Demo

### 步骤 1：打开 Anaconda Prompt

从开始菜单打开 **Anaconda Prompt**（不是 PowerShell，不是 Git Bash）。

### 步骤 2：依次执行以下命令

在 Anaconda Prompt 中，**逐条输入**以下命令（每输完一条按回车）：

```batch
conda activate rlwds
```

```batch
d:
```

```batch
cd d:\WORK\test_2\rl-wds
```

```batch
set NO_PROXY=*
```

```batch
set BOKEH_RESOURCES=cdn
```

```batch
bokeh serve --port 8080 --allow-websocket-origin="*" ./demo.py
```

执行完最后一条命令后，终端会显示：

```
Bokeh app running at: http://localhost:8080/demo
Starting Bokeh server with process id: xxxx
```

### 步骤 3：打开浏览器

在浏览器地址栏输入：**http://localhost:8080/demo**

## 使用说明

1. 选择水分配系统（**Anytown** 或 **D-Town**）
2. 选择需求模式和泵速模式（**Original** 或 **Randomized**）
3. 点击 **Load water distribution system** 按钮加载系统
4. 点击 **Optimize pump speeds** 按钮开始优化
5. 优化完成后，可点击 **Replay optimization** 回放优化过程

## 停止 Demo

在 Anaconda Prompt 中按 **Ctrl+C** 停止服务器。

## 常见问题排查

### 1. 提示 "bokeh 不是内部或外部命令"

conda 环境未正确激活，请确保先执行了 `conda activate rlwds`。

### 2. 浏览器页面空白或不加载

`BOKEH_RESOURCES=cdn` 需要从 CDN 加载 JS 资源。如果网络受限（如无科学上网），请改用内联模式：

```batch
set BOKEH_RESOURCES=inline
```

### 3. 端口 8080 被占用

更换端口，例如使用 8088：

```batch
bokeh serve --port 8088 --allow-websocket-origin="*" ./demo.py
```

然后访问 **http://localhost:8088/demo**

### 4. 页面显示但按钮无法交互

尝试使用 `server` 模式加载资源（由 Bokeh 服务器本身提供）：

```batch
set BOKEH_RESOURCES=server
bokeh serve --port 8080 --allow-websocket-origin="*" ./demo.py
```

### 5. 加载 WDS 时报错

确保 `experiments/models/` 目录下有模型文件（`anytownHO1-best.zip` 和 `dtownHO1-best.zip`），`experiments/hyperparameters/` 目录下有配置文件。

## 文件结构

```
rl-wds/
├── demo.py              # Bokeh Panel 演示主程序
├── wdsEnv.py            # Gym 风格的水分配系统环境
├── train.py             # DQN 训练脚本
├── water_networks/      # EPANET .inp 水网模型文件
├── experiments/
│   ├── models/          # 训练好的 DQN 模型 (.zip)
│   └── hyperparameters/ # 超参数配置文件 (.yaml)
└── opti_algorithms/     # 优化算法（DE, NM, PSO, RS）
```
