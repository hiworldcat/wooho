# 正式复测与提交交接说明

本说明用于拿到官方 20 条数据后直接完成复测。不要使用仓库中的历史
`outputs/v2/` 或 `_upload_package/outputs/` 作为正式结果。

## 1. 准备目录

准备两个不同的数据集目录：

- 参考集：只用于冻结正常数据阈值。
- 官方测试集：必须包含官方 20 条轨迹，只用于检测和评分。

两个目录均应包含 LeRobot V2.1 的 `meta/` 与 `data/` 结构。不要改动原始数据。

## 2. 安装环境

在仓库根目录打开 PowerShell：

```powershell
python --version
python -m pip install -r requirements.txt
```

建议关闭浏览器、IDE和其他占用内存的软件。

## 3. 先做轻量预检

该命令只检查文件结构和资源，不读取图片正文：

```powershell
python scripts/preflight_submission.py `
  --reference-root "D:\data\reference" `
  --target-root "D:\data\official-20"
```

只有显示 `"status": "PASS"` 才继续。若提示缺少 `pyarrow` 或 `cv2`，重新执行依赖安装命令。

## 4. 正式运行

```powershell
python scripts/run_final_submission.py `
  --reference-root "D:\data\reference" `
  --target-root "D:\data\official-20"
```

成功标志：

- 输出 `FINAL TARGET RUN: PASS`。
- 正式结果位于 `outputs/target/v2/`。
- PPT 回填表位于 `submission_work/ppt_metrics/PPT_结果回填表.md`。

不要添加 `--allow-same-dataset`。不要把输出改回 `outputs/v2/`。

## 5. 发回以下内容

将这些文件或文件夹完整发回，不要只发截图：

- `outputs/target/v2/diagnostics/`
- `outputs/target/v2/reports/`
- `submission_work/ppt_metrics/`
- PowerShell 全部运行日志
- 至少一个高置信异常案例对应的原始轨迹编号和帧段

## 6. PPT 回填与最终检查

根据 `PPT_结果回填表.md` 更新第 7、10、11 页。治理后数字必须来自同配置复检，
不得估算。更新后运行：

```powershell
python scripts/check_ppt_consistency.py `
  --pptx "最终答辩稿.pptx" `
  --output-root "outputs\target\v2"
```

只有显示 `PPT CONSISTENCY: PASS` 才能打包。

## 7. 生成最终压缩包

```powershell
python scripts/prepare_submission_package.py `
  --pptx "最终答辩稿.pptx" `
  --pdf "最终答辩稿.pdf" `
  --application-form "参赛表.pdf" `
  --team-name "团队名称" `
  --project-name "作品名称"
```

成功后记录命令输出的压缩包路径、大小与 SHA256。上传成功后截图，并重新下载核对。

## 常见停止条件

- 目标集不是 20 条：停止，确认是否选错目录。
- 参考集和目标集相同：停止，不要绕过保护。
- `SUBMISSION READINESS: FAIL`：停止，不要手工复制结果。
- PPT仍有“待回填”：停止，不要导出最终版。
- 压缩包达到 200 MB：停止，排查是否混入原始数据或重复视频。
