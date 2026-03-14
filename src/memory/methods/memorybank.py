# src/memory/methods/memorybank.py
import math
import copy
from dataclasses import dataclass, field
from typing import Optional, Union
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.memory.base import MemoryBase, WorkingMemoryContent
from src.memory.prompt import MEMORYBANK
from src.common import MASMessage, AgentMessage
from src.llm import Message


@dataclass
class MemoryForgetter:
    """指数遗忘曲线：exp(-time_interval / (5 * scale))，低于 threshold 的步骤被遗忘。"""

    trajectory_time_pairs: list[tuple[AgentMessage, int]] = field(default_factory=list)
    threshold: float = 0.3

    def add_traj_time_pair(self, agent_message: AgentMessage, time_stemp: int) -> None:
        self.trajectory_time_pairs.append((agent_message, time_stemp))

    def manage_memory(self) -> list[tuple[AgentMessage, int]]:
        if len(self.trajectory_time_pairs) == 0:
            return []

        max_time_stemp: int = self.trajectory_time_pairs[-1][1]
        self.trajectory_time_pairs = [
            pair
            for pair in self.trajectory_time_pairs
            if self._forgetting_function(max_time_stemp - pair[1]) >= self.threshold
        ]
        return copy.deepcopy(self.trajectory_time_pairs)

    def _forgetting_function(self, time_interval: float, scale: float = 1) -> float:
        return math.exp(-time_interval / 5 * scale)

    def clear(self) -> None:
        self.trajectory_time_pairs = []

    @staticmethod
    def format_task_trajectory(agent_steps: list[AgentMessage]) -> str:
        task_trajectory: str = "\n>"
        for agent_step in agent_steps:
            task_trajectory += f" {agent_step.message}\n{agent_step.get_extra_field('observation')}\n>"
        return task_trajectory


@dataclass
class MemoryBankMASMemory(MemoryBase):
    """
    MemoryBank Memory：基于指数遗忘曲线的 Working Memory 管理。

    特色：
      - 任务执行中用 MemoryForgetter 追踪每步的时间戳
      - retrieve_working_memory 时应用遗忘曲线过滤旧步骤
      - 存储时用 LLM 摘要轨迹再写入 Chroma 向量库
      - 经验在 init_working_memory 时预加载
    """

    def __post_init__(self):
        super().__post_init__()

        self.main_memory = Chroma(
            embedding_function=self.embedding_func,
            persist_directory=self.persist_dir,
        )

        self.current_time_stemp: int = 0
        self.memory_forgetter: MemoryForgetter = MemoryForgetter()

        # 检索超参
        self._successful_topk: int = self.global_config.get("successful_topk", 1)
        self._failed_topk: int = self.global_config.get("failed_topk", 1)

        self._preloaded_experience: str = ""

    # ── Working Memory ────────────────────────────────────────────────────────

    def init_working_memory(
        self,
        task_main: str,
        task_description: str = None,
        context_hint: Optional[dict] = None,
    ) -> None:
        super().init_working_memory(task_main, task_description, context_hint)

        # 重置遗忘器
        self.current_time_stemp = 0
        self.memory_forgetter.clear()

        # 预加载经验
        success_list, fail_list = self._retrieve_experience(
            task_main, self._successful_topk, self._failed_topk
        )
        self._preloaded_experience = self._format_experience(success_list, fail_list)

    def add_working_memory(
        self,
        content: WorkingMemoryContent,
        upstream_ids: Optional[list[str]] = None,
        **kwargs,
    ) -> Optional[str]:
        result = super().add_working_memory(content, upstream_ids, **kwargs)

        # 如果是环境反馈 (action, observation)，额外追踪到遗忘器
        if isinstance(content, tuple) and len(content) == 2:
            action, observation = content
            agent_message = AgentMessage(message=action)
            agent_message.add_extra_field("observation", observation)
            self.memory_forgetter.add_traj_time_pair(agent_message, self.current_time_stemp)
            self.current_time_stemp += 1

        return result

    def retrieve_working_memory(self, **kwargs) -> str:
        """应用遗忘曲线过滤，返回遗忘后的轨迹。"""
        trajectory_time_pairs = self.memory_forgetter.manage_memory()
        agent_messages: list[AgentMessage] = [pair[0] for pair in trajectory_time_pairs]
        filtered_traj = MemoryForgetter.format_task_trajectory(agent_messages)

        desc = self.current_task_context.task_description or ""
        base = desc + filtered_traj

        if self._preloaded_experience:
            return self._preloaded_experience + "\n\n" + base
        return base

    # ── Experiential Memory ───────────────────────────────────────────────────

    def add_experiential_memory(
        self,
        label: Union[bool, float],
        feedback: str = None,
    ) -> None:
        super().add_experiential_memory(label, feedback)

        # 重置遗忘器
        self.current_time_stemp = 0
        self.memory_forgetter.clear()

        # 持久化到 Chroma
        self._store_to_memory(self.current_task_context)

    # ── Private: 存储（LLM 摘要 + Chroma） ─────────────────────────────────

    def _store_to_memory(self, mas_message: MASMessage) -> None:
        """用 LLM 将轨迹压缩为摘要，再存入 Chroma 向量库。"""
        prompt: str = MEMORYBANK.task_summary_user_instruction.format(
            task_trajectory=mas_message.task_description + mas_message.task_trajectory
        )
        messages: list[Message] = [
            Message("system", MEMORYBANK.task_summary_system_instruction),
            Message("user", prompt),
        ]
        response: str = self.llm_model(messages, temperature=0.1)
        mas_message.task_main = response

        meta_data: dict = MASMessage.to_dict(mas_message)
        memory_doc = Document(
            page_content=mas_message.task_main,
            metadata=meta_data,
        )
        if mas_message.label is True or mas_message.label is False:
            self.main_memory.add_documents([memory_doc])
        else:
            raise ValueError("The mas_message must have a bool label!")
        self._index_done()

    # ── Private: 检索 ─────────────────────────────────────────────────────────

    def _retrieve_experience(
        self,
        query_task: str,
        successful_topk: int = 1,
        failed_topk: int = 1,
    ) -> tuple[list[MASMessage], list[MASMessage]]:
        """相似度检索成功/失败轨迹。"""
        true_tasks_doc: list[tuple[Document, float]] = []
        false_tasks_doc: list[tuple[Document, float]] = []

        if successful_topk != 0:
            true_tasks_doc = self.main_memory.similarity_search_with_score(
                query=query_task, k=successful_topk, filter={"label": True}
            )
        if failed_topk != 0:
            false_tasks_doc = self.main_memory.similarity_search_with_score(
                query=query_task, k=failed_topk, filter={"label": False}
            )
        true_tasks_doc = sorted(true_tasks_doc, key=lambda x: x[1])
        false_tasks_doc = sorted(false_tasks_doc, key=lambda x: x[1])

        true_task_messages: list[MASMessage] = [
            MASMessage.from_dict(doc[0].metadata) for doc in true_tasks_doc
        ]
        false_task_messages: list[MASMessage] = [
            MASMessage.from_dict(doc[0].metadata) for doc in false_tasks_doc
        ]
        return true_task_messages, false_task_messages

    # ── Private: 格式化 ──────────────────────────────────────────────────────

    @staticmethod
    def _format_experience(
        success_list: list[MASMessage],
        fail_list: list[MASMessage],
    ) -> str:
        """将检索到的经验格式化为可拼入 prompt 的文本。"""
        parts: list[str] = []
        if success_list:
            parts.append("## Relevant Successful Experiences")
            for i, msg in enumerate(success_list, 1):
                parts.append(
                    f"### Success #{i}\n"
                    f"Task: {msg.task_main}\n"
                    f"Trajectory: {msg.task_description}\n{msg.task_trajectory}"
                )
        if fail_list:
            parts.append("## Relevant Failed Experiences")
            for i, msg in enumerate(fail_list, 1):
                parts.append(
                    f"### Failure #{i}\n"
                    f"Task: {msg.task_main}\n"
                    f"Trajectory: {msg.task_description}\n{msg.task_trajectory}"
                )
        return "\n\n".join(parts)