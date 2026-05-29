# RL-WDS Demo 快速启动

## 每次开机后

在终端中依次执行：

```bash
conda activate rlwds
cd d:/WORK/test_2/rl-wds
set NO_PROXY=*
set BOKEH_RESOURCES=cdn
bokeh serve --port 8080 --allow-websocket-origin="*" ./demo.py
```

浏览器打开：**http://localhost:8080/demo**

## 其他常用命令

```bash
# 训练 Anytown 网络
python train.py --params anytownMaster --seed 7

# 查看已安装环境
conda env list
```



