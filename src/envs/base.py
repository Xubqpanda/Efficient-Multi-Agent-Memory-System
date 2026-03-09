# src/envs/base.py
"""
Env：所有环境的统一抽象基类。

从 src/mas/base.py 迁移至此，src/mas/base.py 保留 re-export 保证向后兼容。

子类按任务类型实现：
  - 交互式任务（ALFWorld 等）：实现 step()，设置 max_trials
  - 单轮 QA 任务（HLE 等）：step() 内部调 judge，max_trials=1
"""


class Env:
    """环境交互接口，子类根据具体任务重写。"""

    def __init__(self):
        pass

    def reset(self) -> None:
        pass

    def step(self, action: str) -> tuple[str, float, bool]:
        """
        执行 action，返回 (observation, reward, done)。

        交互式任务：推进环境状态，返回当前 observation。
        QA 任务：内部调 judge，返回 (judge_output, reward, True)。
        """
        raise NotImplementedError

    def feedback(self) -> tuple[float, bool, str]:
        """
        任务结束后的最终反馈，返回 (final_reward, done, feedback_str)。
        所有框架的 run_task 结尾都调用此方法。
        """
        raise NotImplementedError

    def process_action(self, action: str) -> str:
        """对 action 进行预处理（如格式化）。默认返回原始 action。"""
        return action