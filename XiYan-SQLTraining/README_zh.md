# XiYan-SQLTraining 框架


## 新闻🔥

+ `2025-10-27` 🌟我们非常高兴地发布了XiYan-SQL的第一版训练框架**XiYan-SQLTraining**，欢迎大家使用，在未来我们会补充更多的信息来完善这项框架。



## 介绍


XiYan-SQLTraining 框架是XiYan提出的特别针对Text-to-SQL任务的大模型后训练框架，目前主要支持功能如下；
- [x] 原生数据转为训练数据
- [x] 训练数据增强
- [x] 基础模型Text2SQL任务微调
- [x] XiYanSQL MOE多方言模型训练 
- [x] 模型推理/评测
- [ ] Text2SQL继续GRPO训练
- [ ] 不同类型SQL模型集成
- [ ] ...

框架正在正在完善中，欢迎各位使用者贡献～


## 使用方法


### 环境准备

1. **创建Conda环境** 使用以下命令为训练创建并激活新环境：
```bash
conda create -n xiyansql python=3.10
conda activate xiyansql
```


2. **安装依赖** 激活环境后，通过运行以下命令安装必需的依赖包：
```bash
pip install -r requirements.txt
```
nvi驱动cuda版本11.8-12.4测试均可以，但是所需依赖的版本可以根据情况进行升级。


### 数据准备

#### 已经有完整的训练数据

请按照JSON LIST文件格式准备，文件内容样例如下，其中每条数据遵循以下格式：
```json
[
  {
    "id": 0,
    "conversations": [
      {
        "role": "user",
        "content": "你是一名SQLite专家，xxx..."
      },
      {
        "role": "assistant",
        "content": "SELECT xxx..."
      }
    ],
    "sql_type": "sqlite"
  },
  {
    "id": 1,
    "conversations": [
      {
        "role": "user",
        "content": "你是一名SQLite专家，xxx..."
      },
      {
        "role": "assistant",
        "content": "SELECT xxx..."
      }
    ],
    "sql_type": "sqlite"
  }
]
```
训练数据样例文件如`train/datasets/train_examples.json`

#### 你也可以从原始数据开始构建，流程在data/文件夹中：

1. 首先处理原始数据，建议在`data_warehouse`下为每块数据创建一个独立文件夹，如`data_warehouse/bird_train`，然后可以通过如下命令，生成处理好可集成的的数据集：
```bash
bash data_processing.sh
```
输入参数分别为`raw_data_path` 原始数据路径，`db_conn_config`数据库配置，`processed_data_dir`处理好保存的文件夹路径，`save_mschema_dir`是否保存mschema文件，`save_to_configs`处理好的数据保存到数据配置文件中。
该处理过程主要涉及从数据库中读取生成m-schema形式的db schema，并将处理好的数据写入完整的配置文件仓库中，方便后续选择数据使用。
`data_processing.sh`中给出了一个使用样例。

2. 数据组装，将处理好的至少一个数据集打包组装成最终送给模型训练的数据

```bash
bash data_assembler.sh
```
输入参数`dataset_config_path` 为数据配置文件其中可以放置多个数据集块，`save_path`为最终训练数据输出路径。
该过程涉及到数据的组装，数据处理，按照prompt生成训练格式数据等。
`data_assembler.sh`中给出了一个使用样例。


### 模型训练

整体流程在`train/`文件夹中

1. 准备模型，在`train/utils`中提供了下载模型的脚本，可以根据网络情况选择在何种源头下载。
```bash
python model_download.py
```
2. SFT 训练脚本为xiyan_sft.sh
```bash
bash xiyan_sft.sh
```
其中需要准备好如上所述的训练数据，模型，训练超参数，较大的模型可以选择开启lora，（推荐可以先采用QWEN2.5系列模型开启训练）。
3. 如果以lora方式进行训练，保存模型的adapter，需要与原模型进行合并，脚本位于`utils/adapter_merge.py`。


### 模型评测

整体流程在`evaluation/`文件夹中，建议每部分数据独立一个文件夹，如`evaluation/bird_evaluation`

1. 模型推理
```bash
bash sql_infer.sh
```
输入参数`model_name_or_path`模型地址，`expr_version`为版本号，`test_set_path`测试集路径，`batch_size`为并发处理大小。

2. 推理结果评测

```bash
bash sql_eval.sh
```

输入参数`pred_sql_path`为预测sql路径，`test_sql_path`为测试集路径包含ground-truth sql，`db_conn_config`为需连接的数据库配置，`save_eval_path`为保存路径。



## 联系我们

如果您对我们的研究或产品感兴趣，请随时联系我们。

#### 联系信息:

刘义富, zhencang.lyf@alibaba-inc.com

#### 加入我们的钉钉群

<a href="https://github.com/alibaba/XiYan-SQL/XiYan-SQLTraining/blob/imgs/xiyansql_dingding.png">Ding Group钉钉群</a> 

## 应用
欢迎大家体验基于XiYan-SQL打造的智能问数解决方案——**析言GBI**。
登录阿里云百炼-应用广场-析言GBI，任何产品体验及效果优化建议欢迎与我们交流。

产品介绍请访问：https://help.aliyun.com/zh/model-studio/user-guide/brief-introduction-of-gbi-products

体验产品请访问：https://bailian.console.aliyun.com/xiyan

产品钉群：94725009401




## 引用
如果您觉得我们的工作对您有帮助，欢迎给我们一个引用。

```bibtex
@article{XiYanSQL,
      title={XiYan-SQL: A Novel Multi-Generator Framework For Text-to-SQL}, 
      author={Yifu Liu and Yin Zhu and Yingqi Gao and Zhiling Luo and Xiaoxia Li and Xiaorong Shi and Yuntao Hong and Jinyang Gao and Yu Li and Bolin Ding and Jingren Zhou},
      year={2025},
      eprint={2507.04701},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.04701}, 
}
```

