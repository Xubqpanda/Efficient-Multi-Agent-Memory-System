# src/memory/methods/generative.py
from dataclasses import dataclass
from typing import Optional, Union
from langchain_chroma import Chroma
from langchain_core.documents import Document
import re

from src.memory.base import MemoryBase
from src.memory.prompt import GENERATIVE
from src.common import MASMessage
from src.llm import Message


@dataclass
class GenerativeMASMemory(MemoryBase):
    """
    Generative Memory：基于 LLM 重要性评分的经验检索。

    特色：
      - 存储时直接存入 Chroma 向量库（task_main 作为 embedding key）
      - 检索时先 2x 过采样，再用 LLM 对成功轨迹打重要性分数，取 top-k
      - 经验在 init_working_memory 时预加载，拼入 prompt
    """

    def __post_init__(self):
        super().__post_init__()

        self.main_memory = Chroma(
            embedding_function=self.embedding_func,
            persist_directory=self.persist_dir,
        )

        # 检索超参：从 global_config 读取，有默认值
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

        # 预加载经验：检索相关历史轨迹并格式化
        success_list, fail_list = self._retrieve_experience(
            task_main, self._successful_topk, self._failed_topk
        )
        self._preloaded_experience = self._format_experience(success_list, fail_list)

    def retrieve_working_memory(self, **kwargs) -> str:
        base_prompt = super().retrieve_working_memory(**kwargs)
        if self._preloaded_experience:
            return self._preloaded_experience + "\n\n" + base_prompt
        return base_prompt

    # ── Experiential Memory ───────────────────────────────────────────────────

    def add_experiential_memory(
        self,
        label: Union[bool, float],
        feedback: str = None,
    ) -> None:
        super().add_experiential_memory(label, feedback)
        self._store_to_memory(self.current_task_context)

    # ── Private: 存储 ─────────────────────────────────────────────────────────

    def _store_to_memory(self, mas_message: MASMessage) -> None:
        """将已标记的 MASMessage 存入 Chroma 向量库。"""
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

    def _retrieve_memory_raw(
        self,
        query_task: str,
        successful_topk: int = 1,
        failed_topk: int = 1,
    ) -> tuple[list[MASMessage], list[MASMessage]]:
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

    def _retrieve_experience(
        self,
        query_task: str,
        successful_topk: int = 1,
        failed_topk: int = 1,
    ) -> tuple[list[MASMessage], list[MASMessage]]:
        """检索并用 LLM 重要性评分排序，返回 top-k 成功/失败轨迹。"""
        successful_task_trajectories, failed_task_trajectories = self._retrieve_memory_raw(
            query_task, 2 * successful_topk, 2 * failed_topk
        )

        # LLM 重要性评分（仅对成功轨迹）
        importance_score: list[float] = []
        for success_task in successful_task_trajectories:
            prompt: str = GENERATIVE.select_task_user_prompt.format(
                trajectory=success_task.task_description + "\n" + success_task.task_trajectory,
                query_scenario=query_task,
            )
            response: str = self.llm_model(
                messages=[
                    Message("system", GENERATIVE.select_task_system_prompt),
                    Message("user", prompt),
                ]
            )
            match = re.search(r"\d+", response)
            score = int(match.group()) if match else 0
            importance_score.append(score)

        sorted_success_tasks = [
            task
            for _, task in sorted(
                zip(importance_score, successful_task_trajectories),
                key=lambda x: x[0],
                reverse=True,
            )
        ]

        return sorted_success_tasks[:successful_topk], failed_task_trajectories[:failed_topk]

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
