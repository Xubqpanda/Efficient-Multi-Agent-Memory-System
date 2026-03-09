# src/mas/noagent/noagent.py
"""
NoAgentSolver：单 LLM solver，作为无 agent 协作的 baseline。

继承 MetaMAS，run_task 结尾与 autogen / dylan / macnet 三个框架完全对齐：

    final_reward, final_done, final_feedback = self.env.feedback()
    self.meta_memory.save_task_context(label=final_done, feedback=final_feedback)
    self.meta_memory.backward(final_done)
    return final_reward, final_done

judge 逻辑封装在 QAEnv.feedback() 里，NoAgentSolver 不感知评分细节。
"""

from dataclasses import dataclass

from src.mas.base import MetaMAS, QAEnv, Env
from src.reasoning import ReasoningBase, ReasoningConfig
from src.memory.base import MASMemoryBase
from src.common.message import AgentMessage
from src.llm import Message
from src.mas.format import format_task_prompt_with_insights, format_task_context

NOAGENT_SYSTEM_PROMPT = (
    "You are an expert problem solver. "
    "Analyze the problem carefully and provide a clear, well-reasoned answer."
)


@dataclass
class NoAgentSolver(MetaMAS):
    """
    单 LLM solver baseline，继承 MetaMAS。

    与三个框架的核心差异：
      - 无 agents_team，无多步 step 循环
      - env 必须是 QAEnv 子类，judge 逻辑封装在 env.feedback() 里
    """

    def __post_init__(self):
        self.observers = []
        self.reasoning_config = ReasoningConfig(temperature=0, stop_strs=['\n'])

    # ─────────────────────────────────────────────────────────────────────────
    # build_system
    # ─────────────────────────────────────────────────────────────────────────

    def build_system(
        self,
        reasoning: ReasoningBase,
        mas_memory: MASMemoryBase,
        env: Env,
        config: dict,
    ) -> None:
        """
        Args:
            reasoning  : LLM 推理模块。
            mas_memory : memory 实例。
            env        : 必须是 QAEnv 子类（如 HLEEnv / FrontierScienceEnv）。
            config     : 支持以下字段：
                           successful_topk (int)  : 检索成功案例数，默认 1
                           failed_topk     (int)  : 检索失败案例数，默认 0
                           insights_topk   (int)  : 检索 insight 数，默认 3
                           threshold       (float): 检索相似度阈值，默认 0
                           system_prompt   (str)  : 覆盖默认 system prompt
        """
        if not isinstance(reasoning, ReasoningBase):
            raise TypeError("reasoning must be an instance of ReasoningBase")
        if not isinstance(mas_memory, MASMemoryBase):
            raise TypeError("mas_memory must be an instance of MASMemoryBase")
        if not isinstance(env, QAEnv):
            raise TypeError("env must be an instance of QAEnv")

        self._successful_topk: int   = config.get('successful_topk', 1)
        self._failed_topk:     int   = config.get('failed_topk', 0)
        self._insights_topk:   int   = config.get('insights_topk', 3)
        self._threshold:       float = config.get('threshold', 0)
        self._system_prompt:   str   = config.get('system_prompt', NOAGENT_SYSTEM_PROMPT)

        self.notify_observers("Configuration Loaded:")
        self.notify_observers(f"Successful Topk   : {self._successful_topk}")
        self.notify_observers(f"Failed Topk       : {self._failed_topk}")
        self.notify_observers(f"Insights Topk     : {self._insights_topk}")
        self.notify_observers(f"Retrieve Threshold: {self._threshold}")

        self._reasoning = reasoning
        self.meta_memory = mas_memory
        self.set_env(env)

    # ─────────────────────────────────────────────────────────────────────────
    # run_task
    # ─────────────────────────────────────────────────────────────────────────

    def run_task(self, task_config: dict) -> tuple[float, bool]:
        """
        执行单次 trial，结尾与三个框架完全对齐。

        task_config 支持以下字段：
          task_main        (str)  : 题目核心内容（必填）
          task_description (str)  : 完整题目描述，默认同 task_main
          few_shots        (list) : in-context few-shot，默认空列表
          context_hint     (dict) : 可选任务元信息，传给 memory
          problem          (str)  : 传给 env.set_task() 的题目文本，默认同 task_main
          reference        (str)  : 传给 env.set_task() 的参考答案 / rubric
        """
        if task_config.get('task_main') is None:
            raise ValueError("Missing required key 'task_main' in task_config")

        task_main:        str  = task_config['task_main']
        task_description: str  = task_config.get('task_description', task_main)
        few_shots:        list = task_config.get('few_shots', [])
        context_hint:     dict = task_config.get('context_hint', {})

        env: QAEnv = self.env
        env.reset()
        env.set_task(
            problem=task_config.get('problem', task_main),
            reference=task_config.get('reference', ''),
            **{k: v for k, v in task_config.items()
               if k not in ('task_main', 'task_description', 'few_shots',
                            'context_hint', 'problem', 'reference')},
        )

        # ── 初始化 inside-trial context ────────────────────────────────────
        self.meta_memory.init_task_context(
            task_main=task_main,
            task_description=task_description,
            context_hint=context_hint,
        )

        # ── 检索跨任务历史经验 ─────────────────────────────────────────────
        successful_trajs, _, insights = self.meta_memory.retrieve_memory(
            query_task=task_main,
            successful_topk=self._successful_topk,
            failed_topk=self._failed_topk,
            insight_topk=self._insights_topk,
            threshold=self._threshold,
        )

        # ── 构建 prompt ────────────────────────────────────────────────────
        memory_few_shots: list[str] = [
            format_task_context(
                traj.task_description,
                traj.task_trajectory,
                traj.get_extra_field('key_steps'),
            )
            for traj in successful_trajs
        ]

        user_prompt: str = format_task_prompt_with_insights(
            few_shots=few_shots,
            memory_few_shots=memory_few_shots,
            insights=list(insights),
            task_description=self.meta_memory.summarize(),
        )
        self.notify_observers(user_prompt)

        messages = [
            Message('system', self._system_prompt),
            Message('user',   user_prompt),
        ]

        # ── 单次 LLM call ──────────────────────────────────────────────────
        answer: str = self._reasoning(messages, self.reasoning_config)
        self.notify_observers(f"Answer: {answer}")

        # ── 记录到 inside-trial StateChain ─────────────────────────────────
        agent_msg = AgentMessage(
            agent_name='solver',
            user_instruction=user_prompt,
            message=answer,
        )
        self.meta_memory.add_agent_node(agent_msg, upstream_agent_ids=[])

        # ── 提交 answer，触发 env 内部 judge ───────────────────────────────
        env.submit(answer)

        # ── 与三个框架完全对齐的结尾 ───────────────────────────────────────
        final_reward, final_done, final_feedback = self.env.feedback()
        self.notify_observers(final_feedback)
        self.meta_memory.save_task_context(label=final_done, feedback=final_feedback)
        self.meta_memory.backward(final_done)

        return final_reward, final_done

    # ─────────────────────────────────────────────────────────────────────────
    # Observer 日志机制（与三个框架保持一致）
    # ─────────────────────────────────────────────────────────────────────────

    def add_observer(self, observer) -> None:
        self.observers.append(observer)

    def notify_observers(self, message: str) -> None:
        for observer in self.observers:
            observer.log(message)