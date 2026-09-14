# 初赛提交冻结清单

## 正式结果

- [x] 确认官方本次仅提供 20 条标准参考轨迹，没有独立测试集。
- [x] 冻结验证结果写入 `outputs/official_v2/`。
- [x] `run_manifest_v2.json` 如实记录 `dataset_role` 为 `negative_control`、`same_dataset` 为 `true`。
- [x] 正式报告、异常明细和 Episode 评分来自同一次运行。
- [x] 执行 `check_submission_readiness.py --allow-reference-only`，结果为 `PASS`。
- [x] 报告确认覆盖 20 条轨迹、4142 帧，合并报警数为 0，质量分为 92.76。
- [x] PPT 指标已按冻结报告回填。
- [x] 本次无高置信异常案例，不虚构异常或治理效果。
- [ ] PPT 中的轨迹号、帧段、异常数和评分与冻结报告一致。

## PPT 与参赛表

- [ ] PPT 展示官方 20 条标准参考轨迹的全量验证结果。
- [ ] 100 条合成轨迹明确标注为工程验证数据。
- [ ] 合成异常与治理案例明确标注为工程验证，不冒充官方数据结果。
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

复核本次已经冻结的官方参考集结果：

```powershell
python scripts/check_submission_readiness.py `
  --output-root "outputs\official_v2" `
  --expected-episodes 20 `
  --allow-reference-only
```

PPT、PDF和参赛表确认后生成最终压缩包：

```powershell
python scripts/prepare_submission_package.py `
  --pptx "最终答辩稿.pptx" `
  --pdf "最终答辩稿.pdf" `
  --application-form "参赛表.pdf" `
  --output-root "outputs\official_v2" `
  --team-name "团队名称" `
  --project-name "作品名称"
```

若复赛另发独立目标集，再使用 `scripts/run_final_submission.py` 或
`scripts/finalize_submission.ps1` 完成严格双目录复测、检查与打包。
