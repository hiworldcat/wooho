# 单一问题消融数据设计

本目录为每一种问题类型生成一个派生 parquet case。每个 case 只注入一种问题，源数据集不做任何修改。

## 问题类型矩阵

| issue_id | 轴向 | 性质 | 目标 | 预期检测族 | 说明 |
|---|---|---|---|---|---|
| black_screen | 非时序 | 硬性 | image | vision_single | 基座相机的一段图像替换为纯黑帧。 |
| white_screen | 非时序 | 硬性 | image | vision_single | 基座相机的一段图像替换为纯白帧。 |
| flower_screen | 非时序 | 硬性 | image | vision_single | 基座相机的一段图像替换为块状随机彩色花屏。 |
| invalid_image_bytes | 非时序 | 硬性 | image | vision_illegal | 图像 bytes 替换为不可解码 payload。 |
| state_nan | 非时序 | 硬性 | state | state_illegal | state 某一维在短片段内写入 NaN。 |
| state_inf | 非时序 | 硬性 | state | state_illegal | state 某一维在短片段内写入 Inf。 |
| action_nan | 非时序 | 硬性 | actions | state_illegal | actions 某一维在短片段内写入 NaN。 |
| state_wrong_dim | 非时序 | 硬性 | state | state_illegal | 单个 state 行被改成非法向量长度。 |
| timestamp_non_monotonic | 时序 | 硬性 | timestamp | temporal_illegal | 单个 timestamp 被改成向后倒退。 |
| frame_index_duplicate | 时序 | 硬性 | frame_index | temporal_illegal | 单个 frame_index 与前一帧重复。 |
| frame_index_skip | 时序 | 硬性 | frame_index | temporal_illegal | frame_index 从局部位置开始整体跳号，但行仍保留。 |
| row_reorder | 时序 | 硬性 | row | temporal_illegal | 交换两个相邻行，破坏时间顺序。 |
| vision_drop_small | 时序 | 软性 | image | vision_temporal | 小面积丢帧：用上一帧图像掩盖短片段缺帧。 |
| vision_drop_large | 时序 | 软性 | image | vision_temporal | 大面积丢帧：用上一帧图像掩盖长片段缺帧。 |
| vision_jump_small | 时序 | 软性 | image | vision_temporal | 小面积跳帧：短片段使用未来视觉帧，造成局部画面跳变。 |
| vision_jump_large | 时序 | 软性 | image | vision_temporal | 大面积跳帧：长片段使用未来视觉帧，造成持续画面跳变。 |
| state_sensor_delay_small | 时序 | 软性 | state | vision_state_temporal | 小延迟传感器问题：state 相对图像延迟 3 帧。 |
| state_sensor_delay_large | 时序 | 软性 | state | vision_state_temporal | 大延迟传感器问题：state 相对图像延迟 10 帧。 |
| video_signal_delay_small | 时序 | 软性 | image | vision_state_temporal | 小延迟视频信号问题：三路相机相对 state 延迟 3 帧。 |
| video_signal_delay_large | 时序 | 软性 | image | vision_state_temporal | 大延迟视频信号问题：三路相机相对 state 延迟 10 帧。 |
| action_response_delay | 时序 | 软性 | actions | state_temporal | 动作响应延迟：actions 相对观测 state 延迟 5 帧。 |
| state_high_freq_jitter | 时序 | 软性 | state | state_temporal | state 连续片段加入交替正负的高频抖动增强。 |
| action_high_freq_jitter | 时序 | 软性 | actions | state_temporal | actions 连续片段加入交替正负的高频抖动增强。 |

## 消融试验设计

1. 原始基线：在未修改参考数据集上运行检测器，记录各维度得分和误报情况。
2. 单问题灵敏度：逐个 case 运行检测器，检查是否命中 `expected_detector_family` 对应检测族。
3. 面积/幅度灵敏度：对比 small/large 的丢帧、跳帧、传感器延迟、视频信号延迟样本，观察分数和置信度是否随问题规模单调变化。
4. 硬性/软性区分：硬性非法样本应主要触发 structural legality；软性样本应更多影响 temporal 或 cross-modal，而不应被误判为解码/结构损坏。
5. 视角泛化：本轮图像局部问题默认注入 base camera，下一轮可把同类问题复制到 left/right wrist camera，验证多视角覆盖。

本轮刻意不生成混合问题，方便后续把检出/漏检准确归因到单一检测能力。
