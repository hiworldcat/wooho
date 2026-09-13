# 初赛提交冻结清单

## 正式结果

- [ ] 使用不同目录作为参考集和测试集。
- [ ] 正式结果写入 `outputs/target/v2/`。
- [ ] `run_manifest_v2.json` 中 `dataset_role` 为 `target`，`same_dataset` 为 `false`。
- [ ] 正式报告、异常明细和 Episode 评分来自同一次运行。
- [ ] 执行 `check_submission_readiness.py`，结果为 `PASS`。
- [ ] 执行 `preflight_submission.py`，依赖、数据结构、内存和磁盘检查为 `PASS`。
- [ ] 生成 `submission_work/ppt_metrics/PPT_结果回填表.md`。
- [ ] 人工复核至少三个高置信案例。
- [ ] PPT 中的轨迹号、帧段、异常数和评分与冻结报告一致。

## PPT 与参赛表

- [ ] PPT 展示官方 20 条轨迹的全量结果。
- [ ] 100 条合成轨迹明确标注为工程验证数据。
- [ ] 至少展示一个正式检测案例和一个治理复检案例。
- [ ] 删除“后续复测”“预计效果”等未完成措辞。
- [ ] 执行 `check_ppt_consistency.py`，确认无“待回填”和旧工程指标。
- [ ] 使用常见中文字体并在另一台电脑打开检查。
- [ ] 修复图表外部链接，确保打开时没有修复提示。
- [ ] 参赛表、PPT、团队名、作品名和成员信息完全一致。

## 打包与上传

- [ ] 压缩包按 `赛题名称-团队名称-作品名称.zip` 命名。
- [ ] 必须包含参赛表和最终 PPTX，建议附带 PDF 备用。
- [ ] 不放原始数据、历史结果、缓存和重复视频。
- [ ] 压缩包小于 200 MB。
- [ ] 解压后重新打开全部文件。
- [ ] 平台显示提交成功后截图留证，并重新下载检查。

演示视频为可选材料。若尚未完成，不应挤占正式结果与 PPT 收口时间。

## 最终执行命令

正式数据到位后，先运行并校验冻结流程：

```powershell
python scripts/run_final_submission.py `
  --reference-root "参考集目录" `
  --target-root "官方20条测试集目录"
```

PPT、PDF和参赛表确认后生成最终压缩包：

```powershell
python scripts/prepare_submission_package.py `
  --pptx "最终答辩稿.pptx" `
  --pdf "最终答辩稿.pdf" `
  --application-form "参赛表.pdf" `
  --team-name "团队名称" `
  --project-name "作品名称"
```

也可使用 `scripts/finalize_submission.ps1` 一次完成正式复测、检查与打包。
