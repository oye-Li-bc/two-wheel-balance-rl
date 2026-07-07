# -*- coding: utf-8 -*-
"""生成技术报告 (.docx)。排版: 中文宋体小四, 英文 Times New Roman 12pt, 1.5 倍行距。
运行: 在本文件所在目录执行 `python gen_report.py`, 输出 report.docx。"""
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(_HERE, "figures")
OUT = os.path.join(_HERE, "report.docx")

doc = Document()

# ---------- 全局字体/行距 ----------
def _set_cjk(style_or_run):
    rpr = style_or_run
    try:
        el = rpr.element
    except AttributeError:
        el = rpr._element
    rFonts = el.rPr.rFonts if el.rPr is not None and el.rPr.rFonts is not None else None

normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal.font.size = Pt(12)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
normal.paragraph_format.line_spacing = 1.5
normal.paragraph_format.space_after = Pt(0)

def run_font(run, size=12, bold=False, cjk="宋体", color=None):
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    run.font.bold = bold
    if color: run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rf = rpr.get_or_add_rFonts()
    rf.set(qn("w:eastAsia"), cjk)
    rf.set(qn("w:ascii"), "Times New Roman")
    rf.set(qn("w:hAnsi"), "Times New Roman")

def para(text="", size=12, bold=False, align=None, cjk="宋体", space_before=0, space_after=0, indent=True):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    if align: p.alignment = align
    if indent and align is None:
        p.paragraph_format.first_line_indent = Pt(24)  # 首行缩进2字符
    if text:
        r = p.add_run(text); run_font(r, size, bold, cjk)
    return p

def heading(text, level=1):
    sizes = {1: 15, 2: 14, 3: 12}
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_before = Pt(10 if level==1 else 6)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text); run_font(r, sizes.get(level,12), True, "黑体")
    return p

def caption(text):  # 图/表题, 居中, 五号(10.5pt)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing = 1.5; p.paragraph_format.space_before = Pt(2); p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text); run_font(r, 10.5, False, "宋体")

def add_fig(path, width_cm, cap):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.add_run().add_picture(path, width=Cm(width_cm))
    caption(cap)

# ---------- 封面 ----------
for _ in range(5): doc.add_paragraph()
t2 = doc.add_paragraph(); t2.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t2.add_run("基于深度强化学习的两轮移动机器人自平衡控制"); run_font(r, 20, True, "黑体")
t3 = doc.add_paragraph(); t3.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t3.add_run("Deep Reinforcement Learning for Self-Balancing Control of a Two-Wheeled Mobile Robot"); run_font(r, 12, False, "Times New Roman")
for _ in range(6): doc.add_paragraph()
info = [("题　　　目", "基于深度强化学习的两轮移动机器人自平衡控制"),
        ("方　　　法", "深度强化学习（PPO）"),
        ("仿真平台", "NVIDIA Isaac Lab（PhysX） / MuJoCo"),
        ("日　　　期", "2026 年 7 月")]
for k, v in info:
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing = 1.5
    r = p.add_run(f"{k}：{v}"); run_font(r, 14, False, "宋体")
doc.add_page_break()

# ---------- 摘要 ----------
heading("摘　要", 1)
para("两轮移动机器人是典型的欠驱动、强非线性、本质不稳定系统，其自平衡控制传统上依赖精确动力学建模与线性化（如 LQR、MPC），建模成本高、对参数摄动与模型失配较敏感。本文以深度强化学习（Deep Reinforcement Learning, DRL）为核心方法，采用近端策略优化（PPO）算法，在 NVIDIA Isaac Lab 大规模并行仿真环境中端到端地学习仅依靠两轮力矩保持车身直立的控制策略，无需解析动力学模型。针对训练中暴露的“直立前溜”退化解，通过奖励塑形引入水平速度惩罚项予以修正；针对策略过拟合训练器物理引擎的问题，引入域随机化（材质摩擦、连杆质量、关节参数随机化与周期性外力扰动）以提升鲁棒性；并完成从 Isaac（PhysX）到 MuJoCo 的 sim-to-sim 部署验证，过程中定位并修复了部署模型基座惯量缺失导致质量偏差 7.7 倍的缺陷。实验表明：所学策略在训练器中可无限期保持平衡且能从随机推力中恢复；经上述改进后，MuJoCo 迁移下的平均维持时间由 4.8 s 提升至 10.3 s、车身平均倾角由 7.4° 降至约 4°。本工作展示了 DRL 用于欠驱动机器人平衡控制的可行性，所得策略网络极小（13-64-64-2），具备嵌入式实时部署潜力。", indent=True)
kp = doc.add_paragraph(); kp.paragraph_format.line_spacing = 1.5
r = kp.add_run("关键词："); run_font(r, 12, True, "宋体")
r = kp.add_run("深度强化学习；PPO；两轮平衡机器人；奖励塑形；域随机化；sim-to-sim"); run_font(r, 12, False, "宋体")

# ---------- 1 引言 ----------
heading("1　引言", 1)
heading("1.1　问题背景", 2)
para("以两轮/轮腿式为代表的移动机器人（如 Segway、Ascento、各类平衡巡检与物流机器人）具有结构紧凑、机动灵活、占地小的优点，但其本质是一个倒立摆式的欠驱动系统：系统自由度多于独立控制输入，车身姿态本质不稳定，任何微小扰动都会迅速发散。这类系统强非线性、存在轮地接触约束，传统控制方法（LQR、MPC）需要在工作点附近建立精确的动力学模型并线性化，建模与整定成本高，且当负载、地面摩擦、质量分布发生变化时鲁棒性受限。因此，研究一种无需精确解析建模、能自适应非线性动力学的控制方法，具有明确的工程意义。")
heading("1.2　方法优势", 2)
para("强化学习通过智能体与环境的反复交互，以奖惩信号为导向自动学习控制策略，天然适合“难以精确建模、但可大量仿真试错”的控制问题：其一，端到端直接从状态映射到力矩，免去解析动力学建模与线性化；其二，借助深度神经网络作为策略与价值函数的高维非线性逼近器，可处理强非线性动力学；其三，配合大规模并行仿真与域随机化，可在训练阶段主动覆盖参数不确定性，获得对模型失配鲁棒的策略。基于以上考虑，本文选用深度强化学习（PPO 算法）作为两轮机器人自平衡控制的核心方法。")
para("在自主搭建的基础训练管线之上，本文进一步完成了奖励函数塑形、训练稳定性根治、域随机化与 sim-to-sim 部署验证，形成了从建模、训练到跨仿真器部署的完整闭环。")

# ---------- 2 方法设计 ----------
heading("2　方法设计", 1)
heading("2.1　问题的马尔可夫决策过程（MDP）建模", 2)
para("将自平衡控制建模为马尔可夫决策过程 M =（S, A, R, P, γ）。控制频率 50 Hz（物理步长 1/200 s，每 4 个物理步决策一次），单回合最长 20 s。各要素设计如下：")
para("（1）状态/观测空间 S（13 维）：重力方向在机体系的投影（3 维，编码车身姿态）、机体系线速度（3 维）、机体系角速度（3 维）、左右轮关节角速度（2 维）、上一时刻动作（2 维）。观测经在线经验归一化后送入网络。", indent=False)
para("（2）动作空间 A（2 维）：左右轮关节力矩，策略网络输出 [-1, 1] 经缩放系数 20 得到实际力矩（N·m），单轮力矩上限 ±20 N·m。髋关节以高刚度位置伺服锁定于零位，不参与动作。", indent=False)
para("（3）奖励函数 R：以“存活+直立”为主，辅以对晃动、平移、自旋与力矩的惩罚，逐项见表 1。", indent=False)
para("（4）终止条件：回合超时（20 s）判为成功；车身倾角超过约 40°（重力投影阈值 0.7 rad）判为跌倒并提前终止。", indent=False)

# 表1 奖励项
heading("", 3)
caption("表 1　奖励函数各项及权重")
rows = [("奖励项", "数学含义", "权重"),
        ("alive", "每存活一步 +1（鼓励不跌倒）", "+1.0"),
        ("upright", "重力投影水平分量平方（车身越直越小）", "-5.0"),
        ("ang_vel_xy", "翻倒方向角速度平方", "-0.01"),
        ("yaw_spin", "偏航角速度平方（防原地自旋，自定义）", "-0.05"),
        ("lin_vel_z", "竖直方向线速度平方", "-0.1"),
        ("lin_vel_xy", "水平线速度平方（防“直立前溜”，自定义）", "-0.2"),
        ("effort", "轮力矩平方（平滑省力）", "-0.001")]
tb = doc.add_table(rows=len(rows), cols=3); tb.style = "Table Grid"; tb.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, row in enumerate(rows):
    for j, val in enumerate(row):
        c = tb.cell(i, j); c.text = ""
        p = c.paragraphs[0]; p.paragraph_format.line_spacing = 1.0
        r = p.add_run(val); run_font(r, 10.5, i == 0, "宋体")

heading("2.2　核心算法：近端策略优化（PPO）", 2)
para("PPO 是一种基于 actor-critic 框架的策略梯度算法，通过“裁剪替代目标”限制每次更新的策略变化幅度，在稳定性与样本效率间取得平衡。记 πθ 为策略，rt(θ)=πθ(at|st)/πθold(at|st) 为新旧策略概率比，Ât 为广义优势估计（GAE），则其核心优化目标为：")
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.line_spacing = 1.5
r = p.add_run("L^CLIP(θ) = E_t [ min( rt(θ)·Ât , clip(rt(θ), 1-ε, 1+ε)·Ât ) ]"); run_font(r, 12, False, "Times New Roman")
para("其中 ε 为裁剪范围（取 0.2）。优势由 GAE 估计：Ât = Σ (γλ)^l · δ_{t+l}，δt = rt + γV(s_{t+1}) - V(st)，γ=0.99 为折扣因子、λ=0.95。价值网络 V 通过最小化回报的均方误差训练；总损失额外包含价值损失项与鼓励探索的熵正则项。学习率采用基于 KL 散度的自适应调整（目标 KL=0.01）。", indent=False)

heading("2.3　工程实现", 2)
para("仿真与训练平台：NVIDIA Isaac Lab 2.1.0（物理引擎 PhysX）+ 强化学习库 RSL-RL，GPU 为 NVIDIA RTX 5070 Ti。数据来源为仿真自采集：4096 个环境并行采样，每环境每轮采集 24 步，训练 500 轮，累计约 4.9×10⁷ 环境交互步，全程数分钟即可完成。策略与价值网络均为两隐层、每层 64 神经元、ELU 激活的多层感知机（actor：13-64-64-2）。主要超参数见表 2。")
caption("表 2　PPO 训练主要超参数")
hp = [("超参数", "取值", "超参数", "取值"),
      ("并行环境数", "4096", "折扣因子 γ", "0.99"),
      ("每环境步数", "24", "GAE λ", "0.95"),
      ("训练轮数", "500", "裁剪范围 ε", "0.2"),
      ("网络结构", "64-64 (ELU)", "熵系数", "0.005"),
      ("学习率", "1e-3 (自适应)", "更新轮次/批数", "5 / 4"),
      ("初始探索 std", "1.0 (log 参数化)", "目标 KL", "0.01"),
      ("观测归一化", "经验归一化(在线)", "梯度裁剪", "1.0")]
tb2 = doc.add_table(rows=len(hp), cols=4); tb2.style = "Table Grid"; tb2.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, row in enumerate(hp):
    for j, val in enumerate(row):
        c = tb2.cell(i, j); p = c.paragraphs[0]; p.paragraph_format.line_spacing = 1.0
        r = p.add_run(val); run_font(r, 10.5, i == 0, "宋体")

heading("2.4　创新点（对经典方法的改进）", 2)
para("（1）奖励塑形修正退化解。初版策略虽能保持车身直立，却学到“压着微小前倾角匀速滑行”的退化解——在 MuJoCo 中 13 s 内漂移达 10 m 后跌倒。根因在于原奖励仅惩罚竖直速度而未约束水平速度，使“直立滑行”成为合法最优解。为此增设水平速度惩罚项（lin_vel_xy），迫使策略真正原地驻停。", indent=False)
para("（2）训练稳定性根治。四自由度、浮动基座下首轮梯度极大，易导致策略标准差被推为负数、进而 NaN 崩溃。通过（a）对数参数化标准差（std=exp(log_std)，数学上恒正）与（b）在线经验归一化（稳定 critic 回归目标），从根本上消除了训练崩溃。", indent=False)
para("（3）域随机化提升鲁棒性。在训练中随机化材质摩擦（静/动摩擦、弹性）、连杆质量（±15%）、轮关节 armature 与摩擦，并每 5–10 s 施加一次随机水平冲击，促使策略学到“对参数不敏感、且会主动挪轮子接住重心”的鲁棒策略。实践中发现：水平速度惩罚过强（-0.5）会与推力恢复需求冲突，导致训练在“学会↔崩溃”间剧烈震荡（见图 2）；将其调至 -0.2 并适度减弱扰动后训练稳定收敛。", indent=False)
para("（4）sim-to-sim 部署与模型保真度校正。为将 PyTorch 策略部署到 MuJoCo，本文将在线观测归一化的均值/方差解析地折叠进网络首层权重（经数值校验误差 <1e-7），使部署端仅需纯 numpy 前向；对齐了两仿真器的控制周期；并定位修复了部署模型的关键缺陷——底盘 body 缺失惯量定义，导致 MuJoCo 按网格默认密度将基座质量误算为 18 kg（真实 2.3 kg，偏差 7.7 倍），这是 sim-to-sim 迁移失败的主因。", indent=False)

# ---------- 3 实验与分析 ----------
heading("3　实验与分析", 1)
heading("3.1　训练结果", 2)
para("采用最终配置（含域随机化）训练 500 轮，平均奖励在约 120 轮内爬升至 19.7 并保持平稳（后 100 轮标准差仅 0.09），跌倒率稳定趋近于 0（见图 1），表明策略在训练器内已稳定收敛，且能在周期性随机推力下持续保持平衡。")
add_fig(f"{FIG}/fig1_train_curve.png", 12.5, "图 1　最终策略训练曲线：平均奖励（左轴）与跌倒率（右轴，已裁剪）")
para("图 2 对比了域随机化两种配置的训练过程：扰动/惩罚过强时，奖励在冲高后反复崩塌（橙线，最终跌倒率高达 0.87）；按根因调温和后，奖励平稳收敛（绿线）。这印证了奖励项与扰动强度需相互协调的设计经验。")
add_fig(f"{FIG}/fig2_stability.png", 12.5, "图 2　域随机化训练稳定性对比：激进配置（不稳定）vs 温和配置（稳定）")

heading("3.2　平衡性能与 sim-to-sim 迁移分析", 2)
para("在训练器 Isaac 中，最终策略可无限期保持平衡（回合全程不跌倒）并从随机推力中恢复。为检验泛化性，将策略迁移到独立物理引擎 MuJoCo 进行 sim-to-sim 验证（各 20 次随机初始扰动试验）。表 3 给出方法演进各阶段的迁移表现，图 3 对比了修正模型后两种策略的平均维持时间。")
caption("表 3　MuJoCo sim-to-sim 迁移各阶段对比（平均值，20 次试验）")
rr = [("阶段", "部署模型", "维持时间", "平均倾角", "水平漂移"),
      ("初版策略", "缺陷(18kg)", "13.5 s", "7.4°", "10.4 m（前溜）"),
      ("+奖励塑形", "缺陷(18kg)", "4.8 s", "3.4°", "2.2 m"),
      ("+域随机化", "缺陷(18kg)", "7.1 s", "7.4°", "4.6 m"),
      ("+奖励塑形", "修正(2.3kg)", "5.3 s", "3.4°", "1.7 m"),
      ("+域随机化", "修正(2.3kg)", "10.3 s", "4.5°", "5.0 m")]
tb3 = doc.add_table(rows=len(rr), cols=5); tb3.style = "Table Grid"; tb3.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, row in enumerate(rr):
    for j, val in enumerate(row):
        c = tb3.cell(i, j); p = c.paragraphs[0]; p.paragraph_format.line_spacing = 1.0
        r = p.add_run(val); run_font(r, 10.5, i == 0, "宋体")
add_fig(f"{FIG}/fig3_sim2sim.png", 12.5, "图 3　修正部署模型后 MuJoCo 平均维持时间对比（虚线为 Isaac 满回合参考）")
para("由表 3 可见：修正基座质量缺陷带来最大提升（域随机化策略由 7.1 s 提升至 10.3 s）；域随机化相比仅奖励塑形，在正确模型上将维持时间近乎翻倍（5.3 s→10.3 s），验证了其对模型失配的鲁棒作用。")

heading("3.3　局限性分析", 2)
para("（1）残余 sim-to-sim gap：即使修正模型缺陷并对齐控制频率、关节阻尼后，MuJoCo 中仍无法维持满 20 s。这源于 PhysX 与 MuJoCo 在接触求解、数值积分上的固有差异；“原地驻停平衡”作为临界稳定任务对此尤为敏感，域随机化只能缩小而非消除该差异。", indent=False)
para("（2）泛化性有限：训练仅在平地进行，未覆盖斜坡、台阶、变负载等工况。（3）无实物验证：尚未迁移到真实硬件，缺乏对传感器噪声、执行器延迟与时延的建模。（4）实时性：训练依赖 GPU 大规模并行，但推理网络极小，部署侧实时性不成问题。", indent=False)

heading("3.4　工程应用建议", 2)
para("（1）嵌入式部署：最终策略仅为 13-64-64-2 的小型 MLP，前向计算量极低，可经定点量化后部署于 STM32/Jetson 等平台，配合 IMU 与轮编码器以 50–100 Hz 实时运行。（2）缩小 sim-to-real 差距：在仿真中进一步引入传感器噪声、执行器延迟与更宽的动力学随机化，并采用少量真机数据微调（sim-to-real）。（3）严格对齐观测：部署时须逐项核对观测的坐标系、量纲、顺序与归一化，本文的“归一化折叠进权重”方法可简化该流程。（4）扩展任务：在平衡基础上叠加速度指令跟随，实现自主导航与避障。")

# ---------- 4 结论 ----------
heading("4　结论", 1)
for s in [
 "（1）以 PPO 深度强化学习为核心，在 Isaac Lab 大规模并行仿真中端到端学习到了两轮机器人自平衡策略，无需精确解析动力学建模，训练数分钟即稳定收敛并能抵抗随机推力。",
 "（2）奖励塑形是控制类 RL 的关键：通过增设水平速度惩罚项，成功修正了“直立前溜”退化解，使策略真正原地驻停。",
 "（3）对数参数化标准差与在线经验归一化可从根本上根治浮动基座任务的训练崩溃问题。",
 "（4）域随机化显著提升了策略对模型失配的鲁棒性（MuJoCo 维持时间近乎翻倍），但其扰动强度须与奖励项协调，否则会破坏训练稳定性。",
 "（5）sim-to-sim 部署中，模型保真度（尤其惯量参数）比算法本身更易成为瓶颈：一处被忽略的基座惯量缺失即造成 7.7 倍质量偏差，提示工程落地中“模型对齐”与“算法调优”同等重要。",
]:
    para(s, indent=False)

doc.save(OUT)
print("已生成:", OUT)
import os; print("大小:", os.path.getsize(OUT), "字节")
