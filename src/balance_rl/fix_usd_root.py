"""
检查并修复 scene_train_full.usd 的双根问题
========================================
MJCF importer 生成的 USD 在 Robot 下有 worldBody + chassis 两个候选根,
导致 Isaac "Found multiple articulation"。
本脚本: 打印 prim 树, 找到 ArticulationRootAPI 在哪, 把它规整到唯一根。

运行:
  cd <IsaacLab 安装目录>
  ./isaaclab.sh -p <仓库>/src/balance_rl/fix_usd_root.py
"""

import argparse
import os
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from pxr import Usd, UsdPhysics, Sdf

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
USD = os.path.join(_ROOT, "models", "combined", "scene_train_full.usd")
stage = Usd.Stage.Open(USD)

print("\n===== prim 树 + ArticulationRoot 标记 =====")
roots = []
for p in stage.Traverse():
    path = str(p.GetPath())
    if path.count("/") <= 3:
        has_art = p.HasAPI(UsdPhysics.ArticulationRootAPI)
        if has_art:
            roots.append(path)
        tag = "  <<< ARTICULATION_ROOT" if has_art else ""
        print(f"  {path}  [{p.GetTypeName()}]{tag}")

print(f"\nArticulationRootAPI 出现在: {roots}")

# 修复策略: 只保留 chassis 上的 root, 移除其它 prim 上的 ArticulationRootAPI
default_prim = stage.GetDefaultPrim()
print(f"defaultPrim = {default_prim.GetPath() if default_prim else None}")

fixed = False
for path in roots:
    if not path.endswith("chassis"):
        prim = stage.GetPrimAtPath(path)
        prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        print(f"移除非chassis根的 ArticulationRootAPI: {path}")
        fixed = True

# 检查是否有把 chassis 固定到世界的 rootJoint(fix base) —— 平衡训练必须浮动!
print("\n===== 检查 root fixed joint(是否被焊死) =====")
to_remove = []
for p in stage.Traverse():
    if p.IsA(UsdPhysics.FixedJoint) and "rootJoint" in str(p.GetPath()):
        print(f"发现固定根关节: {p.GetPath()}")
        to_remove.append(str(p.GetPath()))

# 彻底删除(不是禁用) —— 禁用会留下 disjointed transform 导致物理snap/NaN
for path in to_remove:
    stage.RemovePrim(path)
    print(f"  → 已删除 {path}(chassis浮动)")
    fixed = True

if fixed:
    stage.GetRootLayer().Save()
    print("✅ 已保存修复后的 USD")
else:
    print("ℹ 无需修复")

app.close()
