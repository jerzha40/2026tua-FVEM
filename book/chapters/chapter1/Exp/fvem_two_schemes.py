"""一维二次有限体积元（FVEM）两种对偶网格的收敛阶比较。

本程序是可复现的标准模型实验，尚非已核实的老师作业原题：
    -u''(x) = pi^2*sin(pi*x), x in (0,1), u(0)=u(1)=0.
精确解 u(x)=sin(pi*x)。

每个主单元使用同一组二次 Lagrange 形函数；两种方法仅改变对偶网格分界点：
- original: alpha=1/4, 1-alpha=3/4（相邻节点中点的普通对偶剖分）
- gauss: alpha=(1-1/sqrt(3))/2, 1-alpha（两点 Gauss 节点）

程序打印两张表：n, H1 error, H1 order, L2 error, L2 order。
同时导出两个 CSV 和一个 markdown 报告，便于在报告中引用。

需要：Python >=3.9; numpy。
运行：python fvem_two_schemes.py
"""

from __future__ import annotations
import csv
import math
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss

MESH_SIZES = (8, 16, 32, 64, 128)
OUT_DIR = Path(__file__).resolve().parent


def basis(t):
    """参考单元 [0,1] 的三个二次 Lagrange 形函数。"""
    return np.array([(2*t-1)*(t-1), 4*t*(1-t), t*(2*t-1)])


def grad_basis(t, h):
    """对实际坐标 x 的导数，其中 t=(x-x_left)/h。"""
    return np.array([4*t-3, 4-8*t, 4*t-1]) / h


def u_exact(x):
    return np.sin(np.pi*x)


def du_exact(x):
    return np.pi*np.cos(np.pi*x)


def rhs(x):
    return np.pi**2*np.sin(np.pi*x)


def gauss_integral(function, left, right, order=12):
    """高精度积分：注意这仅用于计算负载/误差，不改变 FVEM 的格式。"""
    t, w = leggauss(order)
    x = (left+right)/2 + (right-left)*t/2
    return float(np.dot(w, function(x))*(right-left)/2)


def solve_fvem(n, alpha):
    """使用对偶控制体积上的通量守恒构造线性系统。"""
    if not (0 < alpha < 0.5):
        raise ValueError('alpha 必须在 (0,1/2) 内')
    h = 1.0/n
    # n个主单元有2n个内部对偶分界点和2n-1个内部自由度
    cuts = np.array([(cell + t)*h for cell in range(n)
                     for t in (alpha, 1-alpha)])
    A = np.zeros((2*n-1, 2*n-1), dtype=float)
    F = np.zeros(2*n-1, dtype=float)

    # 第 i 个内点的对偶体积：[cuts[i],cuts[i+1]]
    for i in range(2*n-1):
        left, right = cuts[i], cuts[i+1]
        # -u''=f => u'(left)-u'(right)=int(left,right)f
        for sign, x in ((1.0, left), (-1.0, right)):
            cell = min(int(n*x), n-1)
            t = x/h - cell
            for local_id, deriv in enumerate(grad_basis(t, h)):
                global_id = 2*cell + local_id
                if 0 < global_id < 2*n:  # u(0)=u(1)=0
                    A[i, global_id-1] += sign*deriv
        F[i] = gauss_integral(rhs, left, right)

    U = np.zeros(2*n+1)
    U[1:-1] = np.linalg.solve(A, F)
    return U


def error_norms(n, U, quadrature_order=12):
    """用独立的高阶求积计算真实 L2 误差、全 H1 误差。"""
    t, w = leggauss(quadrature_order)
    t = (t+1)/2
    w = w/2
    h = 1/n
    sum_l2 = 0.0
    sum_h1_seminorm = 0.0
    for cell in range(n):
        local_u = U[2*cell:2*cell+3]
        x = h*(cell+t)
        values = np.array([basis(tt) @ local_u for tt in t])
        derivatives = np.array([grad_basis(tt,h) @ local_u for tt in t])
        sum_l2 += h*float(np.dot(w, (u_exact(x)-values)**2))
        sum_h1_seminorm += h*float(np.dot(w, (du_exact(x)-derivatives)**2))
    return math.sqrt(sum_h1_seminorm+sum_l2), math.sqrt(sum_l2)


def run_scheme(alpha):
    records = []
    last = None
    for n in MESH_SIZES:
        U = solve_fvem(n, alpha)
        h1, l2 = error_norms(n, U)
        p1 = None if last is None else math.log2(last[0]/h1)
        p0 = None if last is None else math.log2(last[1]/l2)
        records.append({'n':n, 'H1_error':h1, 'H1_order':p1,
                        'L2_error':l2, 'L2_order':p0})
        last = (h1,l2)
    return records


def markdown_table(records):
    result = ['| n | H1误差 | H1阶 | L2误差 | L2阶 |',
              '|---:|---:|---:|---:|---:|']
    for r in records:
        h1_order = '—' if r['H1_order'] is None else f"{r['H1_order']:.4f}"
        l2_order = '—' if r['L2_order'] is None else f"{r['L2_order']:.4f}"
        result.append(f"| {r['n']} | {r['H1_error']:.6e} | {h1_order} | "
                      f"{r['L2_error']:.6e} | {l2_order} |")
    return '\n'.join(result)


def main():
    schemes = (
        ('原格式（普通对偶网格）', 'fvem_original.csv', 0.25),
        ('Gauss 点格式（Gauss 对偶网格）', 'fvem_gauss.csv',
         (1-1/math.sqrt(3))/2),
    )
    report = [
        '# 二次 FVEM 两种格式收敛实验（标准示例，非已核实的作业原题）',
        '',
        r"$-u^{\prime\prime}=\pi^2\sin(\pi x)$，$u(0)=u(1)=0$，精确解 $u=\sin(\pi x)$。",
        '',
        r'收敛阶计算公式：$p=\log_2(E_n/E_{2n})$。',
        '',
    ]
    for title, filename, alpha in schemes:
        records = run_scheme(alpha)
        table = markdown_table(records)
        print(f'\n{title}（alpha={alpha:.9f}）\n{table}\n')
        report.extend([f'## {title}', '', f'对偶分界点：{alpha:.9f}、{1-alpha:.9f}', '', table, ''])
        with (OUT_DIR / filename).open('w', encoding='utf-8-sig', newline='') as fp:
            writer = csv.DictWriter(fp, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
    (OUT_DIR / 'fvem_two_tables.md').write_text('\n'.join(report), encoding='utf-8')
    print('已导出：fvem_original.csv, fvem_gauss.csv, fvem_two_tables.md')

if __name__ == '__main__':
    main()
