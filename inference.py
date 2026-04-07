"""
Baseline inference script for Claude Code for Health.

Runs an LLM agent against all 3 task difficulties (easy, medium, hard).
Emits [START], [STEP], [END] stdout lines per the OpenEnv spec.

Required env vars:
    API_BASE_URL  — LLM endpoint (default: HF router)
    MODEL_NAME    — model identifier
    HF_TOKEN      — API key
"""

import asyncio
import os
import re
import textwrap
from typing import Optional

from openai import OpenAI

from claude_code_for_health import ClaudeCodeForHealthEnv, MedAction

IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = "claude_code_for_health"
MAX_STEPS = 30
TEMPERATURE = 0.3
MAX_TOKENS = 200

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a clinical AI assistant interacting with a medical environment via CLI commands.
    Each turn, respond with EXACTLY ONE command — no explanation, no markdown, just the command.

    DIAGNOSIS TASKS — commands:
      chart.history          View past medical history, meds, allergies, social, family
      chart.vitals           View vital signs
      chart.labs             List available lab panels
      chart.labs <panel>     View specific lab panel results
      chart.imaging          List available imaging studies
      chart.imaging <type>   View specific imaging findings
      chart.exam             List available physical exam systems
      chart.exam <system>    View specific exam findings
      chart.medications      View current medications
      chart.allergies        View known allergies
      ddx.add <diagnosis>    Add diagnosis to differential
      ddx.remove <diagnosis> Remove from differential
      ddx.list               Show current differential
      ddx.confirm <diagnosis> Submit final diagnosis (ends episode)
      help                   List commands

    CALCULATION TASKS — commands:
      case.read              Read the full patient note
      calculate <name>       Declare which calculator you're using
      submit <number>        Submit numeric answer (ends episode)
      help                   List commands

    NOTE REVIEW TASKS — commands:
      note.read                              Read the clinical note
      note.correct <sentence_id> <text>      Correct an error in a sentence
      note.approve                           Approve note / submit corrections (ends episode)
      help                                   List commands

    Strategy:
    - Always read available data before making decisions
    - For diagnosis: review history, vitals, labs, then form differential before confirming
    - For calculations: read the case, identify the calculator, compute, submit
    - For note review: read the note carefully, correct errors if any, then approve
""")


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_clean = action.replace("\n", " ").strip()
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def clean_llm_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```\w*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    text = text.strip("`").strip()
    if text.startswith("$ "):
        text = text[2:]
    lines = text.strip().split("\n")
    return lines[0].strip()


def get_agent_command(client: OpenAI, messages: list[dict]) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        return clean_llm_output(raw) if raw else "help"
    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)
        return "help"


async def run_task(client: OpenAI, env, difficulty: str) -> float:
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=difficulty, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(options={"task": difficulty})
        observation_text = result.observation.output
        task_type = result.observation.task_type

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Task type: {task_type}\n\nEnvironment output:\n{observation_text}"},
        ]

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            command = get_agent_command(client, messages)

            messages.append({"role": "assistant", "content": command})

            result = await env.step(MedAction(command=command))

            reward = result.reward or 0.0
            done = result.done
            error = result.observation.error or None
            observation_text = result.observation.output

            rewards.append(reward)
            steps_taken = step

            messages.append({"role": "user", "content": f"Environment output:\n{observation_text}"})

            log_step(step=step, action=command, reward=reward, done=done, error=error)

            if done:
                break

        score = sum(rewards)
        score = min(max(score, 0.0), 1.0)
        success = score >= 0.1

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for difficulty in ["easy", "medium", "hard"]:
        if IMAGE_NAME:
            env = await ClaudeCodeForHealthEnv.from_docker_image(IMAGE_NAME)
        else:
            env = ClaudeCodeForHealthEnv(base_url="http://localhost:8000")
        await run_task(client, env, difficulty)


if __name__ == "__main__":
    asyncio.run(main())
