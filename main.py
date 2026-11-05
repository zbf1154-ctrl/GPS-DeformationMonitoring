"""项目入口：保持 ``python main.py`` 的用法，内部调用 ``gnss_deformation.cli``。

用法示例::

    python main.py                        # 自动查找 data/ 下的 CSV
    python main.py --show                 # 运行并弹出图形窗口
    python main.py --model step+rate      # 固定模型
    python main.py --weighted             # 按形式中误差加权
"""

from __future__ import annotations

from gnss_deformation.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
