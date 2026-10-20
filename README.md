# 药物–基因–疾病关联补全

根据随附论文《Gene-Aware Hypergraph Learning for Pharmacogenomic Therapeutic Outcome Prediction》整理的实现。`default` 是完整模型；实验协议和消融项通过 Hydra 配置组选择。

**交付状态：按要求未运行项目代码、训练、测试、语法编译或依赖安装。仅做源文件与配置的静态检查、数据文件完整性核对和打包。下面的命令由你在服务器执行。当前版本不能用于声称已复现论文分数。**

## 1. 完整模型对应什么

`python main.py` 默认加载 `conf/default.yaml`，组合为：

- 药物/基因/疾病特征分别经 Linear–BN–ReLU–Dropout 投影到128维。
- 正训练三元组超图 + 四关系 RGCN；三层通道128→512→256→128，前两层门控残差。
- 超图默认实现论文式(4)的对称归一化，超边权重固定为1。
- 拼接两个分支后线性投影，三实体拼接送入三层 MLP。
- `0.3 × BCE + 0.5 × QFAL`，Adam `weight_decay=1e-4`。两个权重保留原 new_loss 的绝对尺度，未重归一化，以便消融只改变一个因素。
- 基因和疾病两个任务在**同一个模型**中等权联合训练。BCE对每个任务的样本均值再取任务均值；QFAL对每个任务采样的唯一query取均值，再取任务均值。
- QFAL的对抗权重使用原始logit和独立温度，权重停止梯度；已知答案来自完整训练分区。
- 不包含论文未描述的 QCCL、FiLM、memory queue、三池调度等旧实验组件。

主稿未给出所有超参数或Base模型的精确定义。默认保留能复用的原配置值，对评估流程与公式进行修正；不能把本配置称为原始表格的已验证配置。QFAL默认每个任务每步采样最多256个唯一query，每个query随机选择一个正确答案；`qfal_max_queries=0` 使用所有训练query。样本不足时负候选数量小于64，会按真实候选数缩放。

## 2. 配置组织

```text
conf/
  default.yaml                  完整模型 + 共用参数
  protocol/
    independent_test.yaml       PharmGKB内部验证，DrugBank最终测试
    cross_validation.yaml       外层五折，每折有内部验证集
  ablation/
    full.yaml
    base.yaml
    no_qfal.yaml
    hypergraph_only.yaml
    rgcn_only.yaml
    feature_only.yaml
    no_residual.yaml
    no_positive_attenuation.yaml
    no_adversarial.yaml
    gated_fusion.yaml
    no_ppi.yaml
    no_dpi.yaml
    no_ddi.yaml
    no_protein_disease.yaml
```

| 配置项 | 改变的因素 | 用途 |
|---|---|---|
| `ablation=full` | 无 | 完整模型，默认 |
| `ablation=no_qfal` | QFAL权重设0 | 论文的“完整架构，不含排名损失” |
| `ablation=base` | 关闭RGCN和QFAL | 明确定义的超图+BCE Base；不声称与未公开的旧Base逐项相同 |
| `ablation=hypergraph_only` | 关闭RGCN，保留QFAL | 单独判断关系分支作用 |
| `ablation=rgcn_only` | 关闭超图 | 判断超图分支作用 |
| `ablation=feature_only` | 关闭两个图分支 | 特征投影+MLP，保留相同监督 |
| `ablation=no_residual` | 关闭两分支残差 | 残差消融 |
| `ablation=no_positive_attenuation` | 不降低其他已知正确答案权重 | QFAL正例降权消融 |
| `ablation=no_adversarial` | 不使用对抗权重 | QFAL难候选加权消融 |
| `ablation=gated_fusion` | concat改为标量gate融合 | 融合方式对照 |
| `ablation=no_ppi/no_dpi/no_ddi/no_protein_disease` | 去掉对应关系 | 逐关系消融 |

`protocol=...` 与 `ablation=...` 可以任意组合。命令行字段覆盖优先于配置组。未知/退役参数会明确报错，避免拼错字段却静默运行。

## 3. 服务器安装与启动

建议 Python 3.10 或 3.11 的独立环境。先按服务器驱动/CUDA选择安装 PyTorch，然后安装其余依赖。`requirements.txt` 是API版本范围，不是经过本机运行验证的锁文件；每次实验都会记录真实环境。

```bash
cd drug
python -m venv .venv
source .venv/bin/activate
# 先安装适合服务器CUDA的torch，然后：
python -m pip install -r requirements.txt

# 完整模型：默认独立测试、seed=42。
python main.py

# 完整模型：重复五折（5个种子 × 5个外层折）。
python main.py protocol=cross_validation 'seeds=[42,43,44,45,46]'

# 独立测试的五次重复。
python main.py protocol=independent_test 'seeds=[42,43,44,45,46]'

# 论文主要三组：完整模型 / 去掉排名损失 / 明确定义的Base。
python main.py -m ablation=full,no_qfal,base protocol=cross_validation 'seeds=[42,43,44,45,46]'

# 结构与逐关系消融。
python main.py -m ablation=hypergraph_only,rgcn_only,no_residual,no_ppi,no_dpi,no_ddi,no_protein_disease

# 指定GPU或目录；相对路径始终相对项目目录。
python main.py device=cuda:0 data_dir=/absolute/path/dataset results_dir=/absolute/path/results

# 如需分别训练单任务模型（与默认联合模型区分记录）。
python main.py 'tasks=[gene]' experiment_name=gene_only
python main.py 'tasks=[disease]' experiment_name=disease_only

# 显存紧张时缩小QFAL采样和评估分块；改变QFAL采样量属于超参数变化。
python main.py qfal_max_queries=64 ranking_query_batch_size=4 ranking_candidate_batch_size=64

# 只查看组合配置，尚不启动实验（也由你在服务器执行）。
python main.py --cfg job
python main.py protocol=cross_validation ablation=no_qfal --cfg job
```

默认训练上限1000 epoch，每10 epoch验证（第1轮和最后一轮也验证）；连续20次验证没有超过最小增益则早停。选择指标是两个任务 filtered query-macro MRR 的均值。`seeds` 才改变重复实验种子，不能用改名字代替改种子。

PyG部分CUDA scatter内核可能无法完全确定性执行；代码启用确定性请求并保留警告，不能承诺跨硬件逐位复现。`device=cpu` 可用于服务器上的小规模排查。

提供了尚未执行的关键回归检查，覆盖负采样、超图矩阵、QFAL温度、filtered排名/多答案Recall以及评估不更新BN。安装依赖后可由你选择在服务器运行：

```bash
python -m unittest discover -s tests -v
```

## 4. 数据和划分协议

原始 `dataset/` CSV 按原样保留。加载时规范化ID首尾空白，实体ID统一按字符串读取；校验数值特征、有限值、维度和重复ID冲突。特征缺失、必需文件缺失、选中的关系映射后为空等都会报错。特征尺寸不符应核对输入，而不是盲目改配置通过检查。

| 文件 | 用途 |
|---|---|
| `drug_embeddings.csv` | Drug_ID + 768维特征 |
| `gene_embeddings.csv` | Gene_ID + 1024维特征 |
| `disease_embeddings.csv` | Disease_ID + 768维特征 |
| `triplets.csv` | PharmGKB正例 |
| `drugbank_triplets.csv` | 仅独立测试协议加载 |
| `ppi.csv` | Gene_ID_1, Gene_ID_2 |
| `drug_protein.csv` | Drug_ID, Gene_ID |
| `drug_drug.csv` | Drug_ID_1, Drug_ID_2 |
| `protein_disease.csv` | Gene_ID, Disease_ID |

所有模型只使用内层训练正例构建超图；负例和验证/测试正例都不构图。辅助关系作为固定外部上下文，过滤到特征词表后去重、去自环，再双向化；反向边沿用同一relation ID。没有额外调用新的数据库，也没有证明辅助图与外部集合的证据来源独立，需要论文继续披露数据来源、版本和时间。

- **独立测试：** 先对PharmGKB去重，从中划15%验证；DrugBank映射、去重并强制剔除与PharmGKB完全相同三元组后作为测试。保留内层训练模型，不在选好epoch后用验证数据重训练。
- **交叉验证：** PharmGKB去重后外层KFold；每个外层训练分区再划15%验证。CV完全不读取DrugBank文件。
- 这两项都是既定特征词表上的传导式补全；随机三元组划分允许实体和query共享，不等于冷启动或按query分组泛化。
- 拆分、各任务各分区负采样、模型初始化、QFAL采样使用独立派生随机流。相同seed和协议的消融拥有相同数据拆分和分类样本。
- 分类负例按全类型候选词表、每个query的正例数×`neg_ratio`无放回采样；不足则明确记录shortfall。验证/测试负例排除其他分区已经抽到的负例。某分区完全没有负例会报错，不输出无意义的AUROC。

**严格外部独立性与未知正例：** 默认 `negative_filter_scope=train_only`，训练负采样只排除训练已知正例；验证构造不读取测试正例。未知组合可能恰好是留出的正确答案，这是positive-unlabeled标签不确定性。系统在 `heldout_label_collision_audit.json` 中报告这种碰撞，不利用测试标签反向清洗训练。若进行全局已知正例过滤的传统CV benchmark，可以显式 `protocol=cross_validation negative_filter_scope=all_known`；独立测试禁止该选项。两种协议不可混用比较。

## 5. 评估定义

编码、打分、损失、排名已分离。验证/测试调用 `model.eval()` 与 inference mode，不计算QFAL，不更新BN。每次评估只编码一次节点，再分块打分，训练每步不再展开全部实体排名。

- AUROC使用logit；`aupr` 是PR曲线梯形面积；另报 `average_precision`，两者不要混写。
- F1阈值由最佳checkpoint在内层验证集上选出，保存为logit阈值；测试不重新选择阈值。
- `mrr_raw/filtered` 和 `hit@K_raw/filtered` 按目标三元组微平均，同一份rank派生所有指标。
- `mrr_*_query_macro` 先对同query的多个目标取均值，再对query平均；早停使用此口径。
- 单目标filtered排名移除同query其他已知正确答案，但始终保留当前目标。验证过滤只用训练+验证，测试过滤用训练+验证+测试。过滤标签仅进入指标计算，不进入模型打分。
- `recall@K_*_query_macro` 是真正的多答案query recall：`检索到的本次评估正确答案数 / 本次评估正确答案总数`。filtered recall只屏蔽不属于该评估答案集合的其他已知正例，保留全部本次评估正确答案。
- rank同分默认average，可选optimistic/pessimistic；Recall的top-K同分固定以全局实体ID升序打破，输出实体映射供追溯。

新版本Recall不再是Hits的别名。MRR微平均、query-macro平均必须在论文表头明确区分。

## 6. 输出、追溯和绘图

每次启动创建独立目录 `results/<实验名>_<协议>_<UTC时间>_<唯一ID>/`。里面包含：

- `config.json`、`environment.json`、源代码/配置和数据SHA256、`data_stats.json`、实体映射及被过滤记录。
- `seed_*/fold_*/`：正例split清单、每个任务的训练/验证/测试标签样本、采样不足统计、未知正例碰撞审计、split hashes。
- `best_model.pt`：权重、最佳epoch、选择指标、配置、seed/fold、验证校准阈值。旧版本checkpoint不兼容新模型结构。
- `training_history.csv`、`training_summary.json`、`thresholds.json`。
- 逐任务 validation/test predictions、raw/filtered ranks、多答案query recall及汇总JSON。
- `metrics_long.csv` 固定长表；`per_seed_summary.csv` 先汇总各seed内部折；`summary.csv` 再汇总独立seed。单seed的repeat_std留空，不伪装成零方差。
- `run.log` 和 `status.json`，异常退出时标记failed，已经完成的fold结果保留。

`summary.csv` 的折均值是等折权重；它不是把所有out-of-fold预测混在一起重新计算AUROC。需要pooled OOF结果时，应另从逐样本文件按目标协议生成并清楚命名。

```bash
# 可传多个消融实验的长表；同experiment/seed/fold重复会报错。
python draw.py results/FULL_RUN/metrics_long.csv results/NO_QFAL_RUN/metrics_long.csv --protocol cross_validation --output-dir figures
```

图中误差条为独立种子均值的SD；只有一个种子时不画误差条。论文还应提供配对置信区间、实际样本数和统一基线协议，绘图脚本不会替你补造这些统计证据。

## 7. 旧配置迁移

| 旧入口 | 新入口 |
|---|---|
| `new_loss.yaml` | `default.yaml` / `ablation=full`，但移除QCCL并修复公式/评估，不复用旧结果 |
| `new_loss_kfold.yaml` | `protocol=cross_validation` |
| `no_res.yaml` | `ablation=no_residual` |
| 旧 `default.yaml`（BCE+QCCL） | 不再充当默认；新的default是论文BCE+QFAL完整模型 |
| `qccl_lite/standard/advanced` | 退役，论文未描述的扩展不混入主实验 |
| `L2` | `weight_decay` |
| `random_state`、`wandb_run_index` | `seeds=[...]`；每seed/fold独立日志 |
| `graph_ablation_configs` | `ablation=...`，需要多组时用Hydra `-m` |

`alpha`、`loss_type`、`max_attempts`、QCCL相关字段和旧宽表绘图输入已退役。原始实现仍可从原始 `drug.tar.gz` 取回，当前目录不保留第二套容易误用的主入口。

详细修改见 `CHANGELOG.md`。源码与数据包SHA256清单由交付时静态生成；服务器实际运行后的结果再保存到上述目录。
