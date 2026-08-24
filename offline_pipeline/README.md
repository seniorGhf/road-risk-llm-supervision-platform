# 离线模型与证据流水线

本目录保存平台核心研究逻辑的可审计副本，来源于已冻结的“大模型监督/2大模型构建及管理”成果。各脚本保持原始业务逻辑和文件结构，用于说明在线模型、证据包、政策复核和容量约束调度结果如何形成。

执行顺序如下：

1. `01_build_causal_spatiotemporal_dataset.py` 构建只使用决策时刻及以前信息的时空候选样本。
2. `02_train_confirmation_models.py` 训练完整教师、轻量学生和消融模型，并固定监督阈值。
3. `03_build_reasoning_and_distillation.py` 组织事故类型假设、紧凑记忆、政策库和蒸馏数据。
4. `05_realtime_demo.py` 生成单案例的历史因果回放证据包。
5. `14_run_prescriptive_optimization.py` 在预警、巡查、应急容量及空间约束下求解处置组合。
6. `20_evaluate_operational_response_family.py` 对结构化响应类别进行评估。

这些脚本需要项目原始授权数据、路网拓扑和训练依赖，不能在脱离数据所有者许可的环境中直接运行。平台日常启动只需要根目录下的 `server.py`、`web/`、`data/` 和 `models/`。

