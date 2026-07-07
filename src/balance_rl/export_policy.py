"""
把 RSL-RL 训练的 .pt checkpoint 导出成 MuJoCo 部署用的 .npz
========================================
背景: deploy_mujoco.py / verify_balance.py 用的是纯 numpy 小网络(w0/b0/w2/b2/w4/b4),
      直接吃"原始观测"。而 RSL-RL 的 .pt 里存的是 PyTorch 的 actor 三层权重,
      且训练若开了 empirical_normalization, 观测进网络前还要先减均值除标准差。

本脚本做两件事:
  1. 从 model_state_dict 取 actor.0/2/4 的 weight/bias (13->64->64->2, ELU)
  2. 若 checkpoint 里带归一化统计量(obs_norm_state_dict), 把 (mean,std) 折进第一层:
        norm(x) = (x-mean)/std
        W0 @ norm(x) + b0 = (W0/std) @ x + (b0 - W0 @ (mean/std))
     折完后 MuJoCo 端照旧喂原始观测即可, deploy/verify 脚本一行都不用改。
     若没有归一化统计量(如早期 empirical_normalization=False 训练的), 就原样导出。

用法:
  python3 export_policy.py                       # 默认导 two_wheel_balance/model_final.pt -> policy_actor.npz
  python3 export_policy.py --ckpt logs/xxx/model_final.pt --out policy_actor.npz
"""

import argparse
import os
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
DEF_CKPT = os.path.join(_HERE, "logs", "two_wheel_balance", "model_final.pt")
DEF_OUT = os.path.join(_HERE, "policy_actor.npz")


def find_normalizer(ckpt):
    """在 checkpoint 里找 empirical normalization 的 mean/std。
    RSL-RL 把它存在顶层 'obs_norm_state_dict'(EmpiricalNormalization 模块),
    里面通常有 mean 和 var(或 std)。找不到就返回 None。"""
    nrm = ckpt.get("obs_norm_state_dict", None)
    if nrm is None:
        return None
    mean = None
    std = None
    for k, v in nrm.items():
        if k.endswith("mean"):
            mean = np.asarray(v, dtype=np.float64).reshape(-1)
        elif k.endswith("var"):
            std = np.sqrt(np.asarray(v, dtype=np.float64).reshape(-1) + 1e-8)
        elif k.endswith("std"):
            std = np.asarray(v, dtype=np.float64).reshape(-1)
    if mean is None or std is None:
        return None
    return mean, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEF_CKPT)
    ap.add_argument("--out", default=DEF_OUT)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]

    # actor 三层 (线性层在索引 0/2/4, 中间 1/3 是 ELU 激活无参数)
    w0 = sd["actor.0.weight"].cpu().numpy().astype(np.float64)
    b0 = sd["actor.0.bias"].cpu().numpy().astype(np.float64)
    w2 = sd["actor.2.weight"].cpu().numpy().astype(np.float64)
    b2 = sd["actor.2.bias"].cpu().numpy().astype(np.float64)
    w4 = sd["actor.4.weight"].cpu().numpy().astype(np.float64)
    b4 = sd["actor.4.bias"].cpu().numpy().astype(np.float64)

    nrm = find_normalizer(ckpt)
    if nrm is not None:
        mean, std = nrm
        std = np.where(std < 1e-6, 1e-6, std)      # 防止除0
        # 折归一化进第一层: W0'=W0/std(按列), b0'=b0 - W0 @ (mean/std)
        b0 = b0 - w0 @ (mean / std)
        w0 = w0 / std[None, :]
        print(f"[INFO] 检测到 empirical normalization, 已把 mean/std 折进第一层 (obs维度={mean.shape[0]})")
    else:
        print("[INFO] checkpoint 无归一化统计量, 原样导出(适用于 empirical_normalization=False 的模型)")

    np.savez(args.out,
             w0=w0.astype(np.float32), b0=b0.astype(np.float32),
             w2=w2.astype(np.float32), b2=b2.astype(np.float32),
             w4=w4.astype(np.float32), b4=b4.astype(np.float32))
    print(f"[INFO] ✅ 导出完成: {args.out}")
    print(f"       网络: {w0.shape[1]} -> {w0.shape[0]} -> {w2.shape[0]} -> {w4.shape[0]} (ELU)")


if __name__ == "__main__":
    main()
