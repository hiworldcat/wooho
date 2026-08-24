# 多模态机器人数据质量检测报告

- 机器人类型：panda
- 轨迹数量：20
- 总帧数：4142
- 采样频率：10 FPS
- 数据集质量分：**98.5/100**

## 评分构成

| 维度 | 满分 | 当前结果 |
|---|---:|---:|
| 时间完整性 | 20 | 20.0 |
| 状态/动作质量 | 25 | 25.0 |
| 图像质量 | 20 | 20.0 |
| 多模态同步 | 20 | 20.0 |
| 任务覆盖与数据价值 | 15 | 13.5 |

## 当前结论

参考集的字段、帧索引、时间戳、状态/动作有限性、图像解码和严重图像质量指标均通过当前规则。
动作与状态之间存在稳定的一帧基线偏移，已作为记录语义基线保留，没有判为异常。

## 输出文件

- `outputs/diagnostics/basic_quality_summary.json`：基础统计
- `outputs/diagnostics/structural_findings.json`：结构和时序异常
- `outputs/diagnostics/image_metrics_summary.json`：图像质量指标
- `outputs/diagnostics/image_findings.json`：图像异常
- `outputs/diagnostics/multimodal_sync_summary.json`：同步分析
- `outputs/diagnostics/detector_validation.json`：故障注入验证
- `outputs/reports/episode_scores.csv`：逐轨迹评分
- `outputs/reports/dataset_quality_report.json`：完整机器可读报告
