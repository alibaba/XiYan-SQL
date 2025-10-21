# XiYan-SQL 

#### This is the new official Alibaba repository for XiYan-SQL, and the previous collection can be accessed [here](https://github.com/XGenerationLab/XiYan-SQL).

## Latest News🔥

+ `Oct. 20, 2025` 🌟 [New SOTA on BIRD-CRITIC](https://bird-critic.github.io/): XiYan-SQL-CRITIC technique has achieved a remarkable **44.53%** success rate on the BIRD-CRITIC-PG benchmark, securing the top position with SOTA performance! Additionally, it recorded an impressive **48.5%** success rate on the BIRD-CRITIC-Flash benchmark, also establishing a new SOTA performance.
  
+ `Oct. 20, 2025` 🌟 The training framework of XiYan-SQL, **XiYan-SQLTraining**, will soon be released in this official Alibaba repository. Stay tuned!
+ ...

## Introduction
**XiYan-SQL** is an innovative natural language to SQL conversion framework designed to address the performance challenges of large language models in SQL generation tasks. 
This framework introduces a multi-generator ensemble strategy, enhancing SQL generation capabilities by integrating various SQL LLMs. 
XiYan-SQL employs multi-task and multi-format training strategies, producing high-quality and diverse SQL models. It also incorporates algorithms such as M-Schema, Schema Filter, and SQL candidate selection to achieve optimal SQL generation performance.

XiYan-SQL has achieved top rankings in several internationally recognized benchmarks, including BIRD-2023, BIRD-Critic, and Spider, demonstrating its robustness and effectiveness across different scenarios. 

For developers, XiYan-SQL offers multiple models and corresponding source code, facilitating further research and application. 
Contributions to the XiYan-SQL project are welcome!

## Timeline
The major events.

| Date    | Event                                                                                                                                                                                                                                                                                                                                                                             |
|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 2025-10 | XiYan-SQL-CRITIC technique has achieved a remarkable **44.53%** success rate on the [BIRD-CRITIC-PG](https://bird-critic.github.io/) benchmark, securing the top position with SOTA performance! Additionally, it recorded an impressive **48.5%** success rate on the [BIRD-CRITIC-Flash](https://bird-critic.github.io/) benchmark, also establishing a new SOTA performance.   |
| 2025-09 | The download count for the **XiYanSQL-QwenCoder** series models on [ModelScope](https://github.com/XGenerationLab/XiYanSQL-QwenCoder) has exceeded **100k** , making it the most influential SQL model in the field.                                                                                                                                                              |                                                    
| 2025-05 | XiYanSQL-CRITIC algorithm achieves a **41%** Pass Rate score on the BIRD-CRITIC-Flash benchmark, setting a new SOTA performance.                                                                                                                                                                                                                                                  |
| 2025-04 | We have released version **2504** of the **XiYanSQL-QwenCoder** series models, which features enhanced performance compared to the previous version. It still includes four different parameter sizes: 3B, 7B, 14B, and 32B. We encourage everyone to utilize these models.                                                                                                       |
| 2025-02 | We have released the **XiYanSQL-QwenCoder** series model, which includes four different sizes: 3B, 7B, 14B, and 32B parameters, to meet the needs of different developers.                                                                                                                                                                                                        |
|         | XiYanSQL-QwenCoder-32B has been released                                                                                                                                                                                                                                                                                                                                          |
| 2025-01 | XiYanSQL-QwenCoder-32B achieves an EX score of 69.03% on BIRD test, new **SOTA** using only single fine-tuned model                                                                                                                                                                                                                                                               |
| 2024-12 | Reaching the **top** of Bird leaderboard with an EX score of **75.63%** and R-VES of 71.41([new SOTA](https://bird-bench.github.io/))                                                                                                                                                                                                                                             |
| 2024-11 | Proposing XiYanSQL technology **A Multi-Generator Ensemble Framework for Text-to-SQL**                                                                                                                                                                                                                                                                                            |
|         | Achieving 41.20% on NL2GQL, and a competitive score of 72.23% on Bird dev ([bird](https://paperswithcode.com/sota/text-to-sql-on-bird-big-bench-for-large-scale))                                                                                                                                                                                                                 |
|         | Achieving 89.65% on Spider test set ([new SOTA](https://paperswithcode.com/sota/text-to-sql-on-spider)), 69.86% on SQL-Eval ([new SOTA](https://paperswithcode.com/sota/text-to-sql-on-sql-eval-1))                                                                                                                                                                               |
| 2024-10 | Proposing an SQL MoE model [MoMQ](https://github.com/XGenerationLab/MoMQ)                                                                                                                                                                                                                                                                                                         |
| 2024-09 | Proposing DateSolver module                                                                                                                                                                                                                                                                                                                                                       |
| 2024-05 | Proposing M-schema, involving ICL in SQL generation                                                                                                                                                                                                                                                                                                                               |
|         | Achieving 86.98% on Spider test set (SOTA 86.6%)                                                                                                                                                                                                                                                                                                                                  |


## XiYan-SQL Collection
...

## Application
Welcome everyone to try the intelligent data querying solution based on XiYan-SQL, which is called XiYan GBI. We welcome any product experiences and suggestions for optimization.

For product introduction, please visit: https://help.aliyun.com/zh/model-studio/user-guide/brief-introduction-of-gbi-products

To try the product, please visit: https://bailian.console.aliyun.com/xiyan

Product DingTalk Group: 94725009401


## Contact us:

If you are interested in our research or products, please feel free to contact us.

#### Contact Information:

Yifu Liu, zhencang.lyf@alibaba-inc.com

#### Join Our DingTalk Group

<a href="https://github.com/XGenerationLab/XiYan-SQL/blob/main/xiyansql_dingding.png">Ding Group钉钉群</a> 


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=alibaba/XiYan-SQL&Date)](https://star-history.com/#alibaba/XiYan-SQL&Date)

## Citation
If you find our work helpful, feel free to give us a cite.
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
```bibtex
@article{xiyansql_pre,
      title={A Preview of XiYan-SQL: A Multi-Generator Ensemble Framework for Text-to-SQL}, 
      author={Yingqi Gao and Yifu Liu and Xiaoxia Li and Xiaorong Shi and Yin Zhu and Yiming Wang and Shiqi Li and Wei Li and Yuntao Hong and Zhiling Luo and Jinyang Gao and Liyu Mou and Yu Li},
      year={2024},
      journal={arXiv preprint arXiv:2411.08599},
      url={https://arxiv.org/abs/2411.08599},
      primaryClass={cs.AI}
}
```