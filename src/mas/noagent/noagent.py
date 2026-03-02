# src/mas/noagent/noagent.py
"""
NoAgentSolver: 无 agent 框架的单 LLM solver，接入完整 memory 接口。

设计目标：
  - memory API 调用顺序与 autogen / dylan / macnet 完全一致
  - 去掉与交互式环境强绑定的 move_memory_state（FrontierScience 无 step reward）
  - label 支持 bool（olympiad）和 float（research 软标签）

调用序列（对齐三个 MAS 框架）：
  init_task_context(task_main, task_description)
      ↓
  retrieve_memory(query_task, ...)
      ↓
  build_prompt → model_caller.call()         ← 单次 LLM call，无 agent 协作
      ↓
  add_agent_node(agent_message, [])          ← 记录这次 LLM 输出
      ↓
  save_task_context(label)                   ← bool 或 float
  backward(label)
"""

from dataclasses import dataclass, field
from typing import Union

from src.common.message import AgentMessage, MASMessage
from src.memory.base import MASMemoryBase
from src.mas.format import format_task_prompt_with_insights, format_task_context


@dataclass
class NoAgentSolver:
    """
    单 LLM solver，完整接入 MASMemoryBase 接口。
    
    Attributes:
        memory:           MASMemoryBase 实例（EmptyMemory / GenerativeMemory / ...）
        model_caller:     任何实现了 .call(prompt) -> {"content": str} 的对象
        successful_topk:  从 memory 检索成功案例数
        failed_topk:      从 memory 检索失败案例数
        insights_topk:    从 memory 检索 insight 数
        threshold:        检索相似度阈值
    """
    memory:          MASMemoryBase
    model_caller:    object                        # ModelCaller or any .call(prompt) compatible
    successful_topk: int   = 1
    failed_topk:     int   = 0
    insights_topk:   int   = 3
    threshold:       float = 0.0

    # ── 对外主接口（evaluator 调用）───────────────────────────────────────────

    def solve(self, task_main: str, task_description: str = None) -> str:
        """
        完整执行一次 trial：init → retrieve → build_prompt → call LLM → record。
        返回 LLM 的原始输出字符串（answer），label 由 evaluator 打分后再调用 record()。

        Args:
            task_main:        题目核心内容（用于 memory 检索的 key）
            task_description: 完整题目描述（含 prompt instruction），默认同 task_main

        Returns:
            str: LLM 的原始输出
        """
        if task_description is None:
            task_description = task_main

        # 1. 初始化本次 trial 的 inside-trial context
        self.memory.init_task_context(task_main, task_description)

        # 2. 从 cross-trial memory 检索历史经验
        successful_trajs, failed_trajs, insights = self.memory.retrieve_memory(
            query_task=task_main,
            successful_topk=self.successful_topk,
            failed_topk=self.failed_topk,
            insight_topk=self.insights_topk,
            threshold=self.threshold,
        )

        # 3. 格式化历史经验用于 prompt 增强（与三个 MAS 框架完全一致）
        memory_few_shots: list[str] = [
            format_task_context(
                traj.task_description,
                traj.task_trajectory,
                traj.get_extra_field("key_steps"),
            )
            for traj in successful_trajs
        ]
        insight_strs: list[str] = list(insights)

        # 4. 构建增强后的 user prompt
        #    EmptyMemory 时三者均为空列表，prompt 退化为纯题目
        augmented_prompt: str = format_task_prompt_with_insights(
            few_shots=[],             # FrontierScience 数据集无 in-context few-shot
            memory_few_shots=memory_few_shots,
            insights=insight_strs,
            task_description=self.memory.summarize(),
        )

        # 5. 调用 LLM（单次，无 agent 协作）
        response = self.model_caller.call(prompt=augmented_prompt)
        answer: str = response["content"]

        # 6. 将本次输出记录到 inside-trial memory（对齐 add_agent_node 调用）
        agent_msg = AgentMessage(
            agent_name="solver",
            user_instruction=augmented_prompt,
            message=answer,
        )
        self.memory.add_agent_node(agent_msg, upstream_agent_ids=[])

        return answer

    def record(self, label: Union[bool, float], feedback: str = None) -> None:
        """
        trial 结束后由 evaluator 调用，将带标签的结果存入 cross-trial memory。

        Args:
            label:    olympiad 传 bool；research 传 float rubric score（软标签）
            feedback: 可选的额外反馈文本
        """
        # 对齐 autogen/dylan/macnet 的 save_task_context + backward
        self.memory.save_task_context(label=label, feedback=feedback)
        self.memory.backward(label)