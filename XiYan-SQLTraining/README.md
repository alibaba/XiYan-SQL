# XiYan-SQLTraining Framework

## News 🔥
+ `2025-10-30` 🌟 We are pleased to announce the release of the first version of the XiYan-SQL training framework **XiYan-SQLTraining**. We welcome everyone to use it, and we will be adding more information to enhance this framework in the future.

## Introduction

The XiYan-SQLTraining framework is a post-training framework specifically designed for the Text-to-SQL task developed by XiYan. Currently, it mainly supports the following capabilities:

- [x] Conversion of raw data to training data
- [x] Training data augmentation
- [x] Fine-tuning basic models for Text2SQL tasks
- [x] Training the XiYanSQL MOE multi-dialect model
- [x] Model inference/evaluation
- [ ] Continued GRPO training for Text2SQL
- [ ] Integration of different types of SQL models
- [ ] ...
The framework is continuously being improved, and we welcome contributions from users!


## Usage

### Environment Preparation

1. Create a Conda Environment: Use the following commands to create and activate a new environment for training:
```bash
conda create -n xiyansql python=3.10
conda activate xiyansql
```

2. Install Dependencies After activating the environment, run the following command to install the required dependencies:
```bash
pip install -r requirements.txt
```
NVidia driver CUDA versions 11.8-12.4 have been tested and are compatible, but the versions of required dependencies can be upgraded as needed.

### Data Preparation

#### Pre-existing Training Data
Please prepare the data in JSON LIST file format, as shown below, where each entry follows this structure:

```json
[
  {
    "id": 0,
    "conversations": [
      {
        "role": "user",
        "content": "You are an SQLite expert, xxx..."
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
        "content": "You are an SQLite expert, xxx..."
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
An example training data file can be found at `train/datasets/train_examples.json`.

#### Building from Raw Data
You can also start constructing from raw data. The processes are located in the `data/` folder:

1. First, process the raw data. It is advisable to create a separate folder under `data_warehouse` for each data chunk, e.g., `data_warehouse/bird_train`. You can then generate a processed and integratable dataset using the following command:
```bash
bash data_processing.sh
```
The input parameters are `raw_data_path` (path to raw data), `db_conn_config` (database configuration), `processed_data_dir` (path to save the processed data), `save_mschema_dir` (whether to save the m-schema file), and `save_to_configs` (whether to save the processed data in the data configuration file).
This processing mainly involves reading the database to generate the m-schema form of the database schema and writing the processed data into a complete configuration file warehouse for easy selection in subsequent uses.
A usage example is provided in `data_processing.sh`.

2. Data assembly involves packaging at least one processed dataset into the final data for model training:
```bash
bash data_assembler.sh
```
The input parameter `dataset_config_path` is the data configuration file that can contain multiple dataset blocks, and `save_path` is the final output path for the training data.
This process involves data assembly, data processing, and formatting the training data as per the prompts.
An example of usage is provided in `data_assembler.sh`.

### Model Training
The overall process is located in the `train/` folder:

1. Prepare the model; the script to download the model is provided in `train/utils` to choose a source based on your network conditions:
```bash
python model_download.py
```
2. The SFT training script is xiyan_sft.sh:
```bash
bash xiyan_sft.sh
```
You need to prepare the training data, model, and training hyperparameters as described above. For larger models, consider enabling LoRA (it is recommended to first use the QWEN2.5 series model to start training).

3. If training with LoRA, you need to merge the saved adapter with the original model. The script for this can be found in `utils/adapter_merge.py`.

## Model Evaluation
The overall process is in the `evaluation/` folder; it is recommended to keep each part of the data in a separate folder, such as `evaluation/bird_evaluation`.

1. Model inference:
```bash
bash sql_infer.sh
```
The input parameters are `model_name_or_path` (model path), `expr_version` (version number), `test_set_path` (test set path), and `batch_size` (concurrent processing size).

2. Evaluation of inference results:
```bash
bash sql_eval.sh
```
The input parameters are `pred_sql_path` (predicted SQL path), `test_sql_path` (test set path containing ground-truth SQL), `db_conn_config` (database configuration), and `save_eval_path` (path to save evaluation results).

## Contact Us
If you're interested in our research or products, please feel free to contact us.

#### Contact Information:
Yifu Liu, zhencang.lyf@alibaba-inc.com

#### Join Our DingTalk Group

<a href="https://github.com/alibaba/XiYan-SQL/XiYan-SQLTraining/blob/main/imgs/xiyansql_dingding.png">DingTalk Group</a> 

## Applications
We welcome you to experience the intelligent query solutions developed based on XiYanSQL—**XiYan GBI**.
Log into Alibaba Cloud Bailian - Application Square - XiYan GBI. Any product experience and effect optimization suggestions are welcome for discussion.

For product introduction, please visit: https://help.aliyun.com/zh/model-studio/user-guide/brief-introduction-of-gbi-products

To experience the product, please visit: https://bailian.console.aliyun.com/xiyan

Product Ding Group: 94725009401


## Citation
If you find our work helpful, we welcome you to cite us.

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