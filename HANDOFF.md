# 正式复测与提交交接说明

本说明用于复现官方 20 条标准参考轨迹的验证结果。官方本次未另发独立测试集，
因此不要把结果描述为“独立测试集成绩”。不要使用仓库中的历史
`outputs/v2/` 或 `_upload_package/outputs/` 作为提交证据。

## 1. 准备目录

准备官方提供的一个数据集目录：

- 官方标准参考集：包含 20 条轨迹，同时用于冻结阈值和执行自一致性验证。

目录应包含 LeRobot V2.1 的 `meta/` 与 `data/` 结构。不要改动原始数据。
若复赛另发独立目标集，再使用严格的参考集/目标集双目录流程。

## 2. 安装环境

在仓库根目录打开 PowerShell：

```powershell
python --version
python -m pip install -r requirements.txt
```

建议关闭浏览器、IDE和其他占用内存的软件。

## 3. 复核冻结结果

仓库已包含本次官方 20 条数据的冻结结果。提交前无需再次加载原始数据，直接执行：

```powershell
python scripts/check_submission_readiness.py `
  --output-root "outputs\official_v2" `
  --expected-episodes 20 `
  --allow-reference-only
```

只有显示 `SUBMISSION READINESS: PASS` 才继续。若需要从原始数据重新生成，再执行下一节。

## 4. 正式运行

```powershell
python run_v2_pipeline.py `
  --reference-root "D:\data\official-20" `
  --target-root "D:\data\official-20" `
  --output-root "outputs\official_v2" `
  --geometry-config "scripts\geometry_config.json" `
  --allow-same-dataset

python scripts/check_submission_readiness.py `
  --output-root "outputs\official_v2" `
  --expected-episodes 20 `
  --allow-reference-only
```

成功标志：

- 输出 `SUBMISSION READINESS: PASS`。
- 冻结结果位于 `outputs/official_v2/`。
- 当前冻结指标为 20 条轨迹、4142 帧、0 个合并报警、质量分 92.76。

只有官方未提供独立目标集时才使用这两个豁免参数。不要把输出改回 `outputs/v2/`。

## 5. 发回以下内容

将这些文件或文件夹完整发回，不要只发截图：

- `outputs/official_v2/diagnostics/`
- `outputs/official_v2/reports/`
- PowerShell 全部运行日志
- 若有高置信异常，附对应的原始轨迹编号和帧段

## 6. PPT 回填与最终检查

PPT 中的结果数字必须来自冻结报告，不得估算，并统一标注为“官方参考集验证”。
更新后运行：

```powershell
python scripts/check_ppt_consistency.py `
  --pptx "最终答辩稿.pptx" `
  --output-root "outputs\official_v2"
```

只有显示 `PPT CONSISTENCY: PASS` 才能打包。

## 7. 生成最终压缩包

```powershell
python scripts/prepare_submission_package.py `
  --pptx "最终答辩稿.pptx" `
  --pdf "最终答辩稿.pdf" `
  --application-form "参赛表.pdf" `
  --output-root "outputs\official_v2" `
  --team-name "团队名称" `
  --project-name "作品名称"
```

成功后记录命令输出的压缩包路径、大小与 SHA256。上传成功后截图，并重新下载核对。

## 常见停止条件

- 官方参考集不是 20 条：停止，确认是否选错目录。
- 有独立目标集时参考集和目标集仍相同：停止，不要绕过保护。
- 官方参考集结果被描述为“独立测试集成绩”：停止，修正文案。
- `SUBMISSION READINESS: FAIL`：停止，不要手工复制结果。
- PPT仍有“待回填”：停止，不要导出最终版。
- 压缩包达到 200 MB：停止，排查是否混入原始数据或重复视频。
