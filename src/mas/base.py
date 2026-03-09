# src/mas/base.py
"""
MAS 层核心抽象。

MetaMAS : 所有 MAS 框架的统一接口，只暴露 build_system 和 run_task 两个方法。
Agent   : 单个 agent 的通用封装。
Env     : 环境接口（可以适配 ALFWorld 等多步交互任务）。
QAEnv   : 单轮 QA 环境接口（FrontierScience / HLE 等），
          将 judge 逻辑封装在 feedback() 里，使 NoAgentSolver 的
          run_task 结尾与三个框架完全对齐。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Iterable

from src.llm import Message
from src.reasoning import ReasoningBase, ReasoningConfig
from src.memory.base import MASMemoryBase


# ─── Agent ────────────────────────────────────────────────────────────────────

class Agent:
    """单个 agent 的通用封装。"""

    def __init__(
        self,
        name: str,
        role: str,
        system_instruction: str,
        reasoning_module: ReasoningBase,
        memory_module=None,
    ):
        if reasoning_module is None:
            raise ValueError("reasoning_module must not be None.")
        self.name = name
        self.profile = role
        self.system_instruction = system_instruction
        self.reasoning = reasoning_module
        self.memory = memory_module
        self.total_system_instruction = system_instruction

    def add_task_instruction(self, task_instruction: str) -> str:
        self.total_system_instruction = self.system_instruction + "\n" + task_instruction
        return self.total_system_instruction

    def response(self, user_prompt: str, reason_config: ReasoningConfig) -> str:
        messages = [
            Message("system", self.total_system_instruction),
            Message("user", user_prompt),
        ]
        return self.reasoning(messages, reason_config)


# ─── Env ──────────────────────────────────────────────────────────────────────

class Env:
    """
    交互式环境接口，可以适配 ALFWorld 等多步交互任务。
    子类根据具体任务重写。
    """

    def __init__(self):
        pass

    def set_env(self, configs: dict) -> None:
        pass

    def reset(self) -> None:
        pass

    def step(self, action: str) -> tuple[str, float, bool]:
        """执行 action，返回 (observation, reward, done)。"""
        raise NotImplementedError

    def feedback(self) -> tuple[float, bool, str]:
        """任务结束后的最终反馈，返回 (final_reward, done, feedback_str)。"""
        raise NotImplementedError

    def process_action(self, action: str) -> str:
        """对 action 进行预处理（如格式化）。默认返回原始 action。"""
        return action


# ─── QAEnv ────────────────────────────────────────────────────────────────────

class QAEnv(Env):
    """
    单轮 QA 环境接口，适用于 FrontierScience / HLE 等无多步交互的 QA 任务。

    与 Env 的核心区别：
      - 无多步 step 循环，solver 只调用一次 LLM 产生 answer
      - judge 逻辑封装在 feedback() 里，使所有 MAS 框架的 run_task
        结尾保持统一：
            final_reward, final_done, final_feedback = self.env.feedback()
            self.meta_memory.save_task_context(label=final_done, feedback=final_feedback)
            self.meta_memory.backward(final_done)
            return final_reward, final_done

    子类需要实现：
      - set_task(problem, reference, **kwargs) : 每道题开始时注入题目和参考答案
      - submit(answer)                         : solver 提交 answer，触发 judge

    step() 在 QA 场景下不应被调用，调用时直接 raise 避免误用。
    """

    def __init__(self):
        super().__init__()
        self._answer: Optional[str] = None
        self._reward: Optional[float] = None
        self._feedback_str: Optional[str] = None

    def set_task(self, problem: str, reference: str, **kwargs) -> None:
        """
        每道题开始时调用，注入题目内容和参考答案 / rubric。
        子类可通过 **kwargs 接收额外信息（如 subject、task_type 等）。
        """
        self._answer = None
        self._reward = None
        self._feedback_str = None

    def submit(self, answer: str) -> None:
        """
        solver 提交 answer。
        子类在此处调用 judge，计算 reward，存储 feedback_str。
        """
        self._answer = answer

    def step(self, action: str) -> tuple[str, float, bool]:
        """QA 场景无多步交互，不应被调用。"""
        raise NotImplementedError(
            "QAEnv does not support step(). "
            "Use submit(answer) → feedback() instead."
        )

    def feedback(self) -> tuple[float, bool, str]:
        """
        返回 judge 结果，格式与 Env.feedback() 完全一致：
          (final_reward, done, feedback_str)

        final_reward : olympiad → 0.0 / 1.0；research → rubric score（0-10）
        done         : 始终为 True（单轮 QA 完成即结束）
        feedback_str : judge 的详细分析文本，写入 memory 供后续检索
        """
        if self._reward is None:
            raise RuntimeError("Call submit(answer) before feedback().")
        return self._reward, True, self._feedback_str or ""

    def reset(self) -> None:
        self._answer = None
        self._reward = None
        self._feedback_str = None


# ─── MetaMAS ──────────────────────────────────────────────────────────────────

@dataclass
class MetaMAS(ABC):
    """
    所有 MAS 框架的统一抽象基类。

    核心契约：
      - build_system : 注入 reasoning、memory、env 和框架超参，完成框架内部初始化。
      - run_task     : 接受 task_config dict，执行完整 trial，返回 (reward, success)。
    """

    agents_team: Dict[str, Agent] = field(default_factory=dict)
    env: Optional[Env] = None
    meta_memory: Optional[MASMemoryBase] = None

    # ── 工具方法（供子类使用） ─────────────────────────────────────────────────

    def hire(self, agents: Iterable[Agent]) -> None:
        for agent in agents:
            if agent.name not in self.agents_team:
                self.agents_team[agent.name] = agent
            else:
                print(f"[MetaMAS] Agent '{agent.name}' already in team, skipped.")

    def set_env(self, env: Env) -> None:
        self.env = env

    def get_agent(self, agent_name: str) -> Optional[Agent]:
        return self.agents_team.get(agent_name)

    # ── 抽象接口 ───────────────────────────────────────────────────────────────

    @abstractmethod
    def build_system(
        self,
        reasoning: ReasoningBase,
        mas_memory: MASMemoryBase,
        env: Env,
        config: dict,
    ) -> None:
        """完成框架内部组件的初始化与连接。"""
        pass

    @abstractmethod
    def run_task(self, task_config: dict) -> tuple[float, bool]:
        """
        执行单个 trial。

        Args:
            task_config : 任务配置字典，至少包含 task_main 和 task_description。

        Returns:
            (reward, success) : 最终奖励分数和是否成功。
        """
        pass