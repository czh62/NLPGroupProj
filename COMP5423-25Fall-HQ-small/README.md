---
license: cc-by-sa-4.0
task_categories:
- question-answering
language:
- en
size_categories:
- 100K<n<1M
configs:
  - config_name: default
    data_files:
      - split: validation
        path: validation.jsonl
      - split: collection
        path: collection.jsonl
      - split: test
        path: test.jsonl
      - split: train
        path: train.jsonl
---

## HotpotQA small subset

Dataset for PolyU COMP5423 NLP Group Project, sampled from HotpotQA.

Evaluation code refer to github https://github.com/polyunlp/COMP5423-25Fall

**Have fun!**


#### Download dataset
Go to `Files and versions`, click `download file` on each file.
```
- collection.jsonl    # Document collection
- train.jsonl         # Questions and labels for train set
- validation.jsonl    # Questions and labels for validation set
```

Or git clone:
```
git clone https://huggingface.co/datasets/izhx/COMP5423-25Fall-HQ-small
```


#### Original dataset

[HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering](https://aclanthology.org/D18-1259)
