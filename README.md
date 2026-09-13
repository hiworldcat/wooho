# Woohu Multimodal Data Quality Detection

面向 LeRobot v2.1 具身机器人轨迹的多模态数据质量检测、定位、评分与治理流程。

## 当前主线

- `run_v2_pipeline.py`：唯一正式运行入口。
- `scripts/v2_quality_pipeline.py`：V2 检测与 70/20/10 评分主线。
- `scripts/geometry_constraints.py`：Rotation 6D、单臂 SE(3)、双臂约束和弱 State-Vision 检查。
- `scripts/check_submission_readiness.py`：正式结果提交前一致性检查。
- `outputs/target/v2/`：正式测试集结果的默认输出目录。
- `outputs/ablations/`：合成异常与消融验证结果，不属于官方测试集结果。

仓库中历史提交已有的 `outputs/v2/`、`outputs/reports/`、`outputs/diagnostics/`
和 `_upload_package/outputs/` 仅作为 2026-09-03 的历史快照保留。它们混有参考集负对照和旧检测器结果，
不得直接作为最终提交证据。正式结果必须重新生成到 `outputs/target/v2/`。

## 环境

建议使用 Python 3.10 及以上版本：

```powershell
python -m pip install -r requirements.txt
```

原始比赛数据不提交到 Git。`初赛数据/` 和全部 Parquet 文件已被忽略。

## 正式运行

参考集只用于校准阈值，测试集只用于检测与评分。两个目录必须显式提供且不能相同：

```powershell
python run_v2_pipeline.py `
  --reference-root "C:\path\to\reference" `
  --target-root "C:\path\to\target" `
  --output-root "outputs\target\v2" `
  --geometry-config "scripts\geometry_config.json"
```

也可以使用 PowerShell 包装脚本：

```powershell
.\scripts\run_pipeline_v2.ps1 `
  -ReferenceRoot "C:\path\to\reference" `
  -TargetRoot "C:\path\to\target"
```

程序默认拒绝将同一目录同时作为参考集和测试集。只有运行明确的正常集负对照时，
才允许添加 `--allow-same-dataset`，并且必须使用独立输出目录。

## 输出

一次正式运行会生成：

```text
outputs/target/v2/
├── diagnostics/
│   ├── findings_v2.json
│   ├── geometry_constraints_v2.json
│   ├── problem_standards_v2.json
│   ├── reference_baselines_v2.json
│   └── run_manifest_v2.json
└── reports/
    ├── dataset_quality_report_v2.json
    ├── dataset_quality_report_v2.md
    ├── dataset_quality_report_v2_zh.md
    ├── episode_quality_report_v2_detail.md
    └── episode_scores_v2.csv
```

报告中的轨迹文件名使用相对于测试集根目录的路径，不记录本机绝对路径。

## 提交前检查

```powershell
python scripts/check_submission_readiness.py `
  --output-root "outputs\target\v2" `
  --expected-episodes 20
```

只有检查结果为 `PASS`，且典型异常经过人工复核后，才可将数字和案例写入最终 PPT。

快速几何测试：

```powershell
python scripts/test_geometry_constraints_smoke.py
python scripts/test_p1_geometry.py
```

