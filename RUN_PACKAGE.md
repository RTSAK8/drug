# 单次 missing-disease ICD-11 雷达图运行包

本包已经包含：

- 原始模型代码；
- 完整 `dataset/`；
- 修订后的 disease→ICD-11 映射表；
- 按真实查询疾病分类的 category metrics；
- 一次性 missing-disease 分层留出入口；
- 示例风格三联雷达图程序。

## 运行

```bash
cd project-name
chmod +x run_disease_radar_once.sh
CUDA_VISIBLE_DEVICES=0 \
PYTHON_BIN=/home/xingzhiwen/protein/.venv/bin/python \
bash run_disease_radar_once.sh
```

使用当前环境的 Python 时：

```bash
CUDA_VISIBLE_DEVICES=0 bash run_disease_radar_once.sh
```

## 输出

```text
results/disease_radar_once_v2/
├── category_metrics/
├── category_metric_records/
├── disease_icd11_radar_once.png
├── disease_icd11_radar_once_data.csv
└── .single_run_completed.json
```

## 重新运行

入口成功完成后不会自动运行第二次。主动重跑前删除：

```bash
rm -rf results/disease_radar_once_v2
```

## 说明

`category_stratified_holdout` 是为了类别雷达图覆盖设计的一次内部留出，不替代论文中的 DrugBank external-test 主结果。
