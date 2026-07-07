# 基于深度强化学习的两轮移动机器人自平衡控制 —— 程序说明

本程序用深度强化学习（PPO）训练两轮移动机器人的自平衡控制策略，并从
Isaac Lab（PhysX）迁移部署到 MuJoCo 做 sim-to-sim 验证。

---

## 一、运行环境

| 组件 | 版本/说明 |
|---|---|
| 操作系统 | Ubuntu 22.04 |
| GPU | NVIDIA RTX 5070 Ti（CUDA 12.8） |
| 训练框架 | NVIDIA Isaac Lab 2.1.0（物理引擎 PhysX） |
| 强化学习库 | RSL-RL（随 Isaac Lab 提供） |
| Python | 3.10（conda 环境 `env_isaaclab`） |
| PyTorch | 2.12.0+cu128（GPU 版，装在 conda 环境内） |
| 部署/验证 | MuJoCo 3.9.0（纯 numpy 前向，无需 torch） |
| 报告生成 | python-docx 1.2.0（仅生成 .docx 用） |

> 注意：训练/回放/导出需在激活的 conda 环境 `env_isaaclab` 下、经 `isaaclab.sh`
> 运行；MuJoCo 部署/验证用系统 Python 即可。若报
> `Torch not compiled with CUDA enabled`，多半是 `~/.local` 里存在 CPU 版 torch
> 盖住了 conda 的 GPU 版，卸载即可：`python3 -m pip uninstall -y torch torchvision`。

## 二、数据来源

本项目**无外部数据集**：训练数据由 Isaac Lab 仿真在线自采集（4096 个并行环境
交互产生），无需下载或预处理任何数据文件。

## 三、目录结构（scripts/balance_rl/）

```
two_wheel_env_cfg.py     # ★平衡任务环境配置（观测/动作/奖励/域随机化）
ppo_cfg.py               # PPO 超参数配置
train.py                 # 训练入口
play.py                  # 在 Isaac 中回放、观看策略（有画面）
export_policy.py         # 把 .pt 检查点导出为部署用 .npz（自动折叠观测归一化）
verify_balance.py        # 无窗口量化验证（存活率/倾角/漂移），MuJoCo sim-to-sim
deploy_mujoco.py         # 部署到 MuJoCo 并开窗观看（纯 numpy 前向）
validate_env.py          # 只实例化环境、校验配置无误（不训练）
logs/two_wheel_balance/  # 训练模型 model_*.pt（含各阶段备份 *_before/_rewardfix）
policy_actor.npz         # 导出的部署策略（当前为域随机化最终版）
```

模型文件：`../../models/two_wheel_robot/two_wheel.usd`（Isaac 训练用）、
`deploy_chassis.xml`（MuJoCo 部署用，已修正基座惯量）。

## 四、复现步骤

1）训练（约数分钟）：
```bash
conda activate env_isaaclab
cd <IsaacLab 安装目录>
./isaaclab.sh -p <本目录>/train.py --headless --max_iterations 500
```

2）导出策略给 MuJoCo（自动检测并折叠观测归一化）：
```bash
<conda_env>/bin/python <本目录>/export_policy.py
```

3）量化验证 sim-to-sim（系统 Python 即可）：
```bash
python3 <本目录>/verify_balance.py --trials 20 --episode 20
```

4）在 Isaac 中观看（完美平衡+抗推）：
```bash
./isaaclab.sh -p <本目录>/play.py --num_envs 16
```

5）在 MuJoCo 中开窗观看：
```bash
python3 <本目录>/deploy_mujoco.py
```

## 五、关键结果

- Isaac（训练器）：策略无限期保持平衡，并能从随机推力中恢复。
- MuJoCo（sim-to-sim）：经奖励塑形+域随机化+修正部署模型质量缺陷后，平均维持
  时间由 4.8 s 提升至 10.3 s、平均倾角由 7.4° 降至约 4°。

详见报告 `docs/report.pdf`（源 `docs/report.docx`）。
