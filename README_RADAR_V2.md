# Disease radar V2：文件替换与运行说明

## 需要覆盖/新增的文件

将本目录中的文件复制到项目中的同名路径：

- `config.py`
- `utils/preprocessing.py`
- `utils/category_metrics.py`
- `utils/experiment/runner.py`
- `utils/experiment/mode_runner.py`
- `utils/experiment/training_loop.py`
- `draw_radar.py`
- `conf/disease_radar_once.yaml`
- `run_disease_radar_once.py`
- `run_disease_radar_once.sh`
- `resources/disease_icd11_mapping.csv`

## 本次关键修复

1. missing-disease 的负候选不再按候选疾病分类，而是按其来源查询的真实疾病分类。
2. 雷达图专用单次运行改为 `category_stratified_holdout`，从 `triplets.csv` 按疾病大类做一次分层留出。
3. 映射表增加 `Paper_Radar_Class_EN`、`Paper_Radar_Class_ZH`、`Paper_Radar_Order`。
4. 可用类别固定为 13 类：
   `Infectious, Neoplasms, Hematol., Endocrine, Psychiatr., Neurol., Ophthalm., Cardiovasc., Respiratory, Gastrointest., Dermatol., Musculosk., Others`。
5. 不对缺失指标做均值插值。若某类确实没有有效正负样本，仍会被安全删除。
6. shell 入口设置 `MPLBACKEND=Agg`，适配无图形界面的服务器。

## 注意

`category_stratified_holdout` 是为疾病类别雷达图准备的一次内部留出评估，目的是尽量覆盖稀有疾病类别；它不能替代论文中的 DrugBank external-test 主结果。主结果仍应使用原来的 independent test。

## 运行

```bash
chmod +x run_disease_radar_once.sh
CUDA_VISIBLE_DEVICES=0 \
PYTHON_BIN=/home/xingzhiwen/protein/.venv/bin/python \
bash run_disease_radar_once.sh
```

结果目录：

```text
results/disease_radar_once_v2/
├── category_metrics/
├── category_metric_records/
├── disease_icd11_radar_once.png
├── disease_icd11_radar_once_data.csv
└── .single_run_completed.json
```

如需重新执行一次，删除专用目录：

```bash
rm -rf results/disease_radar_once_v2
```
