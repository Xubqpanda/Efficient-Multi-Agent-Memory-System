# src/mas/single_agent/single_agent.py
"""
SingleAgentSolver：单 agent solver，同时支持交互式任务（ALFWorld 等）和单轮 QA 任务（HLE 等）。

继承 MetaMAS，run_task 主循环与三个框架完全对齐：

    for i in range(max_trials):
        answer = reasoning(prompt)
        add_working_memory(AgentMessage)
        observation, reward, done = env.step(answer)
        add_working_memory((answer, observation), reward=reward)
        if done: break

    final_reward, final_done, final_feedback = env.feedback()
    meta_memory.add_experiential_memory(label=final_done, feedback=final_feedback)
    return final_reward, final_done

QA 任务（HLE 等）：env.step() 内部调 judge，max_trials=1，循环自然只跑一次。
交互式任务（ALFWorld 等）：env.step() 推进环境状态，max_trials 由 task_config 或 env 决定。
"""

from dataclasses import dataclass

from src.envs.base import Env
from src.mas.base import MetaMAS
from src.reasoning import ReasoningBase, ReasoningConfig
from src.memory.base import MASMemoryBase
from src.common.message import AgentMessage
from src.llm import Message
from src.mas.format import format_task_prompt_with_insights, format_task_context

SINGLE_AGENT_SYSTEM_PROMPT = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your answer choice}\n"
    "Answer: {your answer}"
)


@dataclass
class SingleAgentSolver(MetaMAS):
    """
    单 agent solver baseline，继承 MetaMAS。

    与三个框架的唯一差异：无 agents_team 拓扑，只有一个隐式 solver agent。
    交互式任务和 QA 任务通过 env 的行为自然区分，SingleAgentSolver 不感知任务类型。
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
            config 支持以下字段：
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
        if not isinstance(env, Env):
            raise TypeError("env must be an instance of Env")

        self._successful_topk: int   = config.get('successful_topk', 1)
        self._failed_topk:     int   = config.get('failed_topk', 0)
        self._insights_topk:   int   = config.get('insights_topk', 3)
        self._threshold:       float = config.get('threshold', 0)
        self._system_prompt:   str   = config.get('system_prompt', SINGLE_AGENT_SYSTEM_PROMPT)

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
        执行单次 trial，主循环与三个框架完全对齐。

        task_config 字段：
          task_main        (str)  : 题目/任务核心内容（必填，memory 检索 key）
          task_description (str)  : 完整任务描述，默认同 task_main
          few_shots        (list) : in-context few-shot，默认空列表
          context_hint     (dict) : 可选任务元信息，传给 memory
          max_trials       (int)  : 最大交互步数；
                                    未指定时优先读 env.max_trials，
                                    env 也没有则默认 1（QA 场景）
        """
        if task_config.get('task_main') is None:
            raise ValueError("Missing required key 'task_main' in task_config")

        task_main:        str  = task_config['task_main']
        task_description: str  = task_config.get('task_description', task_main)
        few_shots:        list = task_config.get('few_shots', [])
        context_hint:     dict = task_config.get('context_hint', {})

        # max_trials 优先级：task_config > env.max_trials > 1
        max_trials: int = task_config.get(
            'max_trials',
            getattr(self.env, 'max_trials', 1)
        )

        env = self.env
        env.reset()

        # ── 初始化 working memory ──────────────────────────────────────────
        self.meta_memory.init_working_memory(
            task_main=task_main,
            task_description=task_description,
            context_hint=context_hint,
        )

        # ── 检索 experiential memory ───────────────────────────────────────
        successful_trajs, _, insights = self.meta_memory.retrieve_experiential_memory(
            query_task=task_main,
            successful_topk=self._successful_topk,
            failed_topk=self._failed_topk,
            insight_topk=self._insights_topk,
            threshold=self._threshold,
        )

        memory_few_shots: list[str] = [
            format_task_context(
                traj.task_description,
                traj.task_trajectory,
                traj.get_extra_field('key_steps'),
            )
            for traj in successful_trajs
        ]
        raw_insights: list[str] = list(insights)

        # ── 主循环（与三个框架对齐）────────────────────────────────────────
        for i in range(max_trials):

            user_prompt: str = format_task_prompt_with_insights(
                few_shots=few_shots,
                memory_few_shots=memory_few_shots,
                insights=raw_insights,
                task_description=self.meta_memory.retrieve_working_memory(),
            )
            self.notify_observers(user_prompt)

            messages = [
                Message('system', self._system_prompt),
                Message('user',   user_prompt),
            ]

            answer: str = self._reasoning(messages, self.reasoning_config)
            self.notify_observers(f"Step {i+1} Answer: {answer}")

            # agent 输出写入 working memory
            self.meta_memory.add_working_memory(
                AgentMessage(
                    agent_name='solver',
                    user_instruction=user_prompt,
                    message=answer,
                ),
                upstream_ids=[],
            )

            # 与 env 交互
            observation, reward, done = env.step(answer)
            step_message = f'Act {i+1}: {answer}\nObs {i+1}: {observation}'
            self.notify_observers(step_message)

            # env 反馈写入 working memory，推进状态链
            self.meta_memory.add_working_memory(
                (answer, observation),
                reward=reward,
            )

            if done:
                break

        # ── 与三个框架完全对齐的结尾 ───────────────────────────────────────
        final_reward, final_done, final_feedback = env.feedback()
        self.notify_observers(final_feedback)
        self.meta_memory.add_experiential_memory(
            label=final_done,
            feedback=final_feedback,
        )

        return final_reward, final_done

    # ─────────────────────────────────────────────────────────────────────────
    # Observer
    # ─────────────────────────────────────────────────────────────────────────

    def add_observer(self, observer) -> None:
        self.observers.append(observer)

    def notify_observers(self, message: str) -> None:
        for observer in self.observers:
            observer.log(message)