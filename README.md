# Woohu Multimodal Data Quality Detection

面向 LeRobot v2.1 具身机器人轨迹的多模态数据质量检测、定位、评分与治理流程。

拿到官方数据后请优先阅读 [`HANDOFF.md`](HANDOFF.md)，其中给出了从预检、正式复测、
PPT 回填到最终打包的完整命令和停止条件。

## 当前主线

- `run_v2_pipeline.py`：唯一正式运行入口。
- `scripts/v2_quality_pipeline.py`：V2 检测与 70/20/10 评分主线。
- `scripts/geometry_constraints.py`：Rotation 6D、单臂 SE(3)、双臂约束和弱 State-Vision 检查。
- `scripts/check_submission_readiness.py`：正式结果提交前一致性检查。
- `scripts/preflight_submission.py`：不加载帧内容的依赖、目录、内存与磁盘预检。
- `scripts/export_ppt_metrics.py`：从冻结报告生成 PPT 结果回填表。
- `scripts/check_ppt_consistency.py`：阻止带占位符或旧指标的 PPT 进入最终压缩包。
- `outputs/official_v2/`：本次官方 20 条标准参考轨迹的冻结验证结果。
- `outputs/target/v2/`：未来获得独立目标集时的默认输出目录。
- `outputs/ablations/`：合成异常与消融验证结果，不属于官方测试集结果。

仓库中历史提交已有的 `outputs/v2/`、`outputs/reports/`、`outputs/diagnostics/`
和 `_upload_package/outputs/` 仅作为 2026-09-03 的历史快照保留。它们混有参考集负对照和旧检测器结果，
不得直接作为最终提交证据。本次初赛证据以 `outputs/official_v2/` 为准。

## 环境

建议使用 Python 3.10 及以上版本：

```powershell
python -m pip install -r requirements.txt
```

原始比赛数据不提交到 Git。`初赛数据/` 和全部 Parquet 文件已被忽略。

## 数据模式

官方本次只提供 20 条标准参考轨迹，没有另发独立测试集。因此仓库中的
`outputs/official_v2/` 是对官方参考集的全流程复跑与自一致性验证，不应描述为
“独立测试集成绩”。复现本次结果时，显式启用同集负对照模式并使用独立输出目录：

```powershell
python run_v2_pipeline.py `
  --reference-root "C:\path\to\official-20" `
  --target-root "C:\path\to\official-20" `
  --output-root "outputs\official_v2" `
  --geometry-config "scripts\geometry_config.json" `
  --allow-same-dataset

python scripts/check_submission_readiness.py `
  --output-root "outputs\official_v2" `
  --expected-episodes 20 `
  --allow-reference-only
```

若复赛另发独立目标集，则参考集只用于校准阈值，目标集只用于检测与评分，
两个目录必须显式提供且不能相同：

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

程序默认拒绝将同一目录同时作为参考集和目标集。只有复现本次官方参考集验证时，
才允许添加 `--allow-same-dataset`，并且必须使用 `outputs/official_v2/` 等独立输出目录。

## 输出

一次正式运行会生成：

```text
outputs/official_v2/
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

报告中的轨迹文件名使用相对于数据集根目录的路径，不记录本机绝对路径。

## 提交前检查

```powershell
python scripts/check_submission_readiness.py `
  --output-root "outputs\official_v2" `
  --expected-episodes 20 `
  --allow-reference-only
```

只有检查结果为 `PASS` 后，才可将数字写入最终 PPT。本次结果应统一称为
“官方参考集验证”；若发现异常案例，还须先人工复核再展示。

未来获得独立目标集时，可使用两段式正式运行流程：

```powershell
python scripts/run_final_submission.py `
  --reference-root "参考集目录" `
  --target-root "官方20条测试集目录"

python scripts/prepare_submission_package.py `
  --pptx "最终答辩稿.pptx" `
  --pdf "最终答辩稿.pdf" `
  --application-form "参赛表.pdf" `
  --output-root "outputs\target\v2" `
  --team-name "团队名称" `
  --project-name "作品名称"
```

打包脚本只收集白名单源代码与本次正式输出，并自动生成文件清单、SHA256 和上传留证清单。参考数据、测试数据、历史输出与缓存不会进入压缩包。

未来获得独立目标集并正式运行前，可单独预检：

```powershell
python scripts/preflight_submission.py `
  --reference-root "参考集目录" `
  --target-root "官方20条测试集目录"
```

正式运行通过后，`submission_work/ppt_metrics/` 会生成 `ppt_metrics.json` 和
`PPT_结果回填表.md`。最终打包前，脚本会自动检查 PPT 中是否残留“待回填”、
旧工程指标或与冻结报告不一致的总分。

快速几何测试：

```powershell
python scripts/test_geometry_constraints_smoke.py
python scripts/test_p1_geometry.py
```
