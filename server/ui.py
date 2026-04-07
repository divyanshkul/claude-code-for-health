"""Custom Gradio dashboard — plugs into OpenEnv's ``gradio_builder`` hook at /web."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import gradio as gr

_CSS = """
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    .term-bar {
        background: #1a2133;
        border: 1px solid rgba(255,255,255,0.12);
        border-bottom: none;
        border-radius: 12px 12px 0 0;
        padding: 11px 16px;
        display: flex;
        align-items: center;
        gap: 12px;
        margin-top: 6px;
        position: relative;
        z-index: 2;
        box-shadow: 0 -4px 20px rgba(0,0,0,0.3);
    }
    .term-dots { display: flex; gap: 7px; }
    .term-dot { width: 11px; height: 11px; border-radius: 50%; }
    .term-dot.r { background: #ff5f57; }
    .term-dot.y { background: #febc2e; }
    .term-dot.g { background: #28c840; }
    .term-title {
        font-family: 'JetBrains Mono', ui-monospace, monospace;
        font-size: 11px;
        color: #4a5568;
        letter-spacing: 0.02em;
    }

    .terminal-area {
        margin-top: 0 !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-top: 1px solid rgba(255,255,255,0.05) !important;
        border-radius: 0 0 12px 12px !important;
        overflow: hidden;
        position: relative;
        z-index: 1;
        box-shadow:
            0 8px 32px rgba(0,0,0,0.5),
            0 2px 8px rgba(0,0,0,0.3),
            inset 0 1px 0 rgba(255,255,255,0.03);
        max-height: 720px !important;
    }
    .terminal-area .cm-scroller,
    .terminal-area .code-block,
    .terminal-area pre { max-height: 680px !important; overflow-y: auto !important; }
    .terminal-area label { display: none !important; }
    .terminal-area pre, .terminal-area code, .terminal-area textarea {
        font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code',
                    ui-monospace, monospace !important;
        font-size: 13px !important;
        line-height: 1.7 !important;
        background: #0a0f18 !important;
        color: #c9d1d9 !important;
        letter-spacing: 0.01em !important;
    }

    .cmd-input input, .cmd-input textarea {
        font-family: 'JetBrains Mono', ui-monospace, monospace !important;
        font-size: 13px !important;
        background: #151c28 !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 8px !important;
        padding: 11px 14px !important;
    }
    .cmd-input input::placeholder, .cmd-input textarea::placeholder {
        color: #64748b !important;
    }

    .sidebar-panel > div { padding: 0 !important; }
    .execute-btn { min-width: 110px !important; }
"""


def _header_html() -> str:
    return (
        '<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:'
        'wght@400;500;600;700&display=swap" rel="stylesheet">'
        '<div style="padding:12px 0 4px;display:flex;align-items:baseline;gap:10px;">'
        '<span style="font-family:\'JetBrains Mono\',monospace;font-size:18px;'
        'font-weight:700;color:#e2e8f0;letter-spacing:-0.03em;">'
        '\U0001f3e5 Clinical Terminal</span>'
        '<span style="font-family:\'JetBrains Mono\',monospace;font-size:10px;'
        'color:#3d4a5c;letter-spacing:0.08em;padding:2px 8px;'
        'border:1px solid rgba(255,255,255,0.06);border-radius:4px;">v1.0</span>'
        '</div>'
    )


def _terminal_bar_html() -> str:
    return (
        '<div class="term-bar">'
        '<div class="term-dots">'
        '<span class="term-dot r"></span>'
        '<span class="term-dot y"></span>'
        '<span class="term-dot g"></span>'
        '</div>'
        '<span class="term-title">claude code for healthcare</span>'
        '</div>'
    )


def _score_html(score: float) -> str:
    if score > 0:
        color, glow, bg = "#4ade80", "rgba(74,222,128,0.3)", "#0c1f14"
    elif score < 0:
        color, glow, bg = "#f87171", "rgba(248,113,113,0.3)", "#1f0c0c"
    else:
        color, glow, bg = "#94a3b8", "rgba(148,163,184,0.1)", "#151c28"

    return (
        f'<div style="background:{bg};border:1px solid rgba(255,255,255,0.1);'
        'border-radius:10px;padding:20px;text-align:center;margin-bottom:10px;">'
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;'
        'color:#8b949e;text-transform:uppercase;letter-spacing:2px;'
        'margin-bottom:8px;">Episode Score</div>'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:38px;'
        f'font-weight:700;color:{color};font-variant-numeric:tabular-nums;'
        f'text-shadow:0 0 30px {glow},0 0 60px {glow};'
        f'letter-spacing:-0.02em;">{score:.2f}</div></div>'
    )


def _status_html(
    task_type: str,
    difficulty: str,
    step: int,
    max_steps: int,
) -> str:
    pct = int(step / max_steps * 100) if max_steps else 0
    bar_color = "#3b82f6" if pct < 75 else "#f59e0b" if pct < 95 else "#ef4444"

    badge_bg, badge_fg = "rgba(96,165,250,0.15)", "#7db8f7"
    if task_type == "diagnosis":
        badge_bg, badge_fg = "rgba(251,191,36,0.15)", "#fcd34d"
    elif task_type == "calculation":
        badge_bg, badge_fg = "rgba(167,139,250,0.15)", "#c4b5fd"
    elif task_type == "note_review":
        badge_bg, badge_fg = "rgba(52,211,153,0.15)", "#6ee7b7"

    lbl = ("font-size:10px;color:#6b7d94;text-transform:uppercase;"
           "letter-spacing:1px;font-family:'JetBrains Mono',monospace;")

    return (
        '<div style="background:#151c28;border:1px solid rgba(255,255,255,0.1);'
        'border-radius:10px;padding:16px;margin-bottom:10px;">'
        f'<div style="{lbl}margin-bottom:14px;font-weight:600;">Status</div>'
        f'<div style="margin-bottom:14px;"><span style="{lbl}">Task</span><br/>'
        f'<span style="display:inline-block;background:{badge_bg};'
        f'color:{badge_fg};padding:3px 10px;border-radius:5px;'
        'font-family:\'JetBrains Mono\',monospace;font-size:12px;'
        f'font-weight:600;margin-top:4px;">{task_type or "\u2014"}</span></div>'
        f'<div style="margin-bottom:14px;"><span style="{lbl}">Difficulty</span><br/>'
        '<span style="font-family:\'JetBrains Mono\',monospace;font-size:13px;'
        f'color:#c9d1d9;margin-top:2px;display:inline-block;">'
        f'{difficulty or "\u2014"}</span></div>'
        f'<div><span style="{lbl}">Progress</span>'
        '<div style="display:flex;align-items:center;gap:8px;margin-top:6px;">'
        '<div style="flex:1;height:4px;background:rgba(255,255,255,0.08);'
        'border-radius:2px;overflow:hidden;">'
        f'<div style="width:{pct}%;height:100%;background:{bar_color};'
        'border-radius:2px;transition:width .4s ease;"></div></div>'
        '<span style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
        f'color:#8b949e;font-weight:600;">{step}/{max_steps}</span>'
        '</div></div></div>'
    )


def _commands_html(cmds: List[str]) -> str:
    lbl = ("font-family:'JetBrains Mono',monospace;font-size:10px;color:#6b7d94;"
           "text-transform:uppercase;letter-spacing:1px;font-weight:600;")
    if not cmds:
        return (
            '<div style="background:#151c28;border:1px solid rgba(255,255,255,0.1);'
            'border-radius:10px;padding:16px;">'
            f'<div style="{lbl}margin-bottom:8px;">Commands</div>'
            '<p style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
            'color:#6b7d94;margin:0;font-style:italic;">awaiting reset\u2026</p></div>'
        )

    items = "".join(
        f'<div style="padding:4px 0;font-family:\'JetBrains Mono\',monospace;'
        f'font-size:12px;color:#c9d1d9;border-bottom:1px solid rgba(255,255,255,0.05);">'
        f'<span style="color:#58a6ff;margin-right:6px;">\u203a</span>{c}</div>'
        for c in cmds
    )
    return (
        '<div style="background:#151c28;border:1px solid rgba(255,255,255,0.1);'
        'border-radius:10px;padding:16px;">'
        f'<div style="{lbl}margin-bottom:10px;">Commands</div>'
        f'{items}</div>'
    )

    items = "".join(
        f'<div style="padding:3px 0;font-family:\'JetBrains Mono\',monospace;'
        f'font-size:11px;color:#8b949e;border-bottom:1px solid rgba(255,255,255,0.03);">'
        f'<span style="color:#3d4a5c;margin-right:4px;">\u203a</span> {c}</div>'
        for c in cmds
    )
    return (
        '<div style="background:#0d1117;border:1px solid rgba(255,255,255,0.06);'
        'border-radius:10px;padding:16px;">'
        f'<div style="{lbl}font-size:10px;color:#4a5568;margin-bottom:10px;">Commands</div>'
        f'{items}</div>'
    )


_TASK_OPTIONS = [
    "Easy \u2014 Note Review",
    "Medium \u2014 Calculation",
    "Hard \u2014 Diagnosis",
]
_TASK_KEY = {
    _TASK_OPTIONS[0]: "easy",
    _TASK_OPTIONS[1]: "medium",
    _TASK_OPTIONS[2]: "hard",
}


def build_custom_dashboard(
    web_manager: Any,
    action_fields: List[Dict[str, Any]],
    metadata: Any,
    is_chat_env: bool,
    title: str,
    quick_start_md: Optional[str],
) -> gr.Blocks:
    """Return a ``gr.Blocks`` app for the Custom tab at /web."""

    async def on_reset(difficulty: str):
        task_key = _TASK_KEY.get(difficulty, "easy")
        try:
            data = await web_manager.reset_environment(
                {"options": {"task": task_key}}
            )
        except Exception as exc:
            return (
                f"ERROR: {exc}",
                _status_html("\u2014", "\u2014", 0, 50),
                _commands_html([]),
                _score_html(0.0),
                "",
            )

        obs = data.get("observation", {})
        output = obs.get("output", "")
        task_type = obs.get("task_type", "")
        step = obs.get("step_number", 0)
        max_steps = obs.get("max_steps", 50)
        cmds = obs.get("available_commands", [])

        terminal = (
            f" \u250c\u2500 {task_type.upper()} \u2500\u2500 new episode\n"
            f" \u2502\n"
            f" \u2502  {output.replace(chr(10), chr(10) + ' \u2502  ')}\n"
            f" \u2502\n"
            f" \u2514\u2500\u2500\u2500\n"
        )

        return (
            terminal,
            _status_html(task_type, task_key, step, max_steps),
            _commands_html(cmds),
            _score_html(0.0),
            "",
        )

    async def on_step(command: str, history: str):
        if not command or not command.strip():
            return (
                history or "",
                "",
                gr.update(),
                gr.update(),
                gr.update(),
            )

        try:
            data = await web_manager.step_environment(
                {"command": command.strip()}
            )
        except Exception as exc:
            return (
                (history or "") + f"\n\u276f {command}\n  \u2718 {exc}\n",
                "",
                gr.update(),
                gr.update(),
                gr.update(),
            )

        obs = data.get("observation", {})
        output = obs.get("output", "")
        error = obs.get("error", "")
        reward = data.get("reward", 0.0)
        done = data.get("done", False)
        step = obs.get("step_number", 0)
        max_steps = obs.get("max_steps", 50)
        task_type = obs.get("task_type", "")
        cmds = obs.get("available_commands", [])

        entry = f"\n\u276f {command}\n"
        if error:
            entry += f"  \u2718 {error}\n"
        entry += f"  {output.replace(chr(10), chr(10) + '  ')}\n"
        if reward != 0:
            sign = "+" if reward > 0 else ""
            entry += f"  \u2500\u2500 reward: {sign}{reward:.4f}\n"
        if done:
            entry += "\n  \u2588\u2588 EPISODE COMPLETE \u2588\u2588\n"

        full = (history or "") + entry

        try:
            state = web_manager.get_state()
            score = state.get("total_score", 0.0)
            difficulty = state.get("difficulty", "")
        except Exception:
            score = 0.0
            difficulty = ""

        return (
            full,
            "",
            _status_html(task_type, difficulty, step, max_steps),
            _commands_html(cmds),
            _score_html(score),
        )

    _SCROLL_JS = """
    () => {
        setTimeout(() => {
            const s = document.querySelector('.terminal-area .cm-scroller')
                   || document.querySelector('.terminal-area pre');
            if (s) s.scrollTop = s.scrollHeight;
        }, 150);
    }
    """

    with gr.Blocks() as blocks:
        gr.HTML(f"<style>{_CSS}</style>" + _header_html())

        with gr.Row(equal_height=False):
            with gr.Column(scale=7, min_width=480):
                with gr.Row():
                    difficulty = gr.Dropdown(
                        choices=_TASK_OPTIONS,
                        value=_TASK_OPTIONS[0],
                        label="Task",
                        scale=3,
                        interactive=True,
                    )
                    reset_btn = gr.Button(
                        "Start Episode",
                        variant="primary",
                        scale=1,
                    )

                gr.HTML(_terminal_bar_html())

                terminal = gr.Code(
                    value=(
                        "  Welcome to Claude Code for Healthcare.\n"
                        "  Select a task and press Start Episode.\n"
                        + "\n" * 18
                    ),
                    label="Terminal",
                    language=None,
                    lines=20,
                    interactive=False,
                    elem_classes=["terminal-area"],
                )

            with gr.Column(scale=3, min_width=250, elem_classes=["sidebar-panel"]):
                score_md = gr.HTML(_score_html(0.0))
                cmd_input = gr.Textbox(
                    placeholder="\u276f type a command\u2026",
                    label="Command",
                    elem_classes=["cmd-input"],
                )
                send_btn = gr.Button(
                    "Execute \u21b5",
                    variant="primary",
                    elem_classes=["execute-btn"],
                )
                status_md = gr.HTML(_status_html("\u2014", "\u2014", 0, 50))
                commands_md = gr.HTML(_commands_html([]))

        reset_outputs = [terminal, status_md, commands_md, score_md, cmd_input]
        step_outputs = [terminal, cmd_input, status_md, commands_md, score_md]

        reset_btn.click(
            fn=on_reset,
            inputs=[difficulty],
            outputs=reset_outputs,
        ).then(fn=None, js=_SCROLL_JS)
        send_btn.click(
            fn=on_step,
            inputs=[cmd_input, terminal],
            outputs=step_outputs,
        ).then(fn=None, js=_SCROLL_JS)
        cmd_input.submit(
            fn=on_step,
            inputs=[cmd_input, terminal],
            outputs=step_outputs,
        ).then(fn=None, js=_SCROLL_JS)

    return blocks
