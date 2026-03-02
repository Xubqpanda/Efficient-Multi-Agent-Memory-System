# src/reasoning/base.py
"""
Reasoning 模块：负责 agent 的推理策略。

ReasoningBase 是抽象接口，ReasoningIO 是最简单的直接 LLM 调用实现。
后续可以添加 CoT、ReAct、ToT 等推理方式。
"""
from dataclasses import dataclass
from typing import Optional

from src.llm import LLMCallable, Message


@dataclass
class ReasoningConfig:
    """控制推理行为的超参配置。None 表示使用 LLM 的默认值。"""
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stop_strs: Optional[list[str]] = None
    num_comps: Optional[int] = None


class ReasoningBase:
    """推理模块抽象基类。"""

    def __init__(self, llm_model: LLMCallable):
        self.llm_model = llm_model

    def __call__(self, prompts: list[Message], config: ReasoningConfig) -> str:
        raise NotImplementedError


class ReasoningIO(ReasoningBase):
    """最简 IO 推理：直接将 prompts 传给 LLM，返回结果。"""

    def __call__(self, prompts: list[Message], config: ReasoningConfig) -> str:
        return self.llm_model(
            prompts,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stop_strs=config.stop_strs,
            num_comps=config.num_comps,
        )