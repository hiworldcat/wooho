# V2 多模态机器人数据质量检测报告

- 数据角色: negative_control
- 参考集标识: 鍙傝€冮泦
- 测试集标识: 鍙傝€冮泦
- 输出标识: official_v2
- 数据版本: v2.1
- 机器人类型: panda
- Episode 数: 20
- 总帧数: 4142
- FPS: 10
- 数据集质量分: **92.76/100**
- 正式评分口径: `official_70_20_10_v1` = 检测质量 69.27/70 + 数据价值 16.75/20 + 工程可靠性 6.74/10
- 旧六维均分: 98.95/100（仅保留为 legacy breakdown）

## V2 检测框架

本程序按“基础全覆盖、增强只选一个”的思路实现基础全覆盖部分，并将增强能力收敛为参考增强检测、跨模态一致性和报警归并。

| 维度 | 满分 | 当前均分 |
|---|---:|---:|
| structural | 25 | 25.0 |
| vision_single | 20 | 19.72 |
| vision_vision | 10 | 10.0 |
| state | 15 | 15.0 |
| temporal | 15 | 14.24 |
| cross_modal | 15 | 15.0 |

## 模块状态

状态枚举: `ok` 正常完成，`warning` 核心完成但存在降级或局部不可用，`unavailable` 核心或子模块无法运行，`fail` 模块执行失败。

| 模块 | ok | warning | unavailable | fail |
|---|---:|---:|---:|---:|
| geometry | 0 | 20 | 0 | 0 |
| arms.left | 20 | 0 | 0 | 0 |
| arms.right | 20 | 0 | 0 | 0 |
| bimanual | 0 | 0 | 20 | 0 |
| state_vision.left | 20 | 0 | 0 | 0 |
| state_vision.right | 20 | 0 | 0 | 0 |

## 异常汇总

- 合并后报警数: 0
- 确定异常: 0
- 高置信异常: 0
- 疑似异常: 0
- 分布外样本: 0

## 问题类型 Top 10

| 问题类型 | 数量 |
|---|---:|

## 输出文件

- `diagnostics/findings_v2.json`: 标准化异常明细
- `diagnostics/reference_baselines_v2.json`: 正常参考集阈值与基线
- `diagnostics/problem_standards_v2.json`: 问题定义与判定标准
- `diagnostics/run_manifest_v2.json`: 本次运行的数据角色与版本口径
- `reports/episode_scores_v2.csv`: episode 级评分表
- `reports/dataset_quality_report_v2.json`: 机器可读完整报告
- `reports/dataset_quality_report_v2_zh.md`: 中文摘要报告
