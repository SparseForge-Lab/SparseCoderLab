# Agentic coding infrastructure

The tiny language model is not expected to become a competent coding agent. `RepoSandbox` copies a licensed toy repository under project-local `work/`. `TaskSpec` describes goal, allowed edits, verifier command and license. `ToolCall` supports inspect, hypothesize/replan, edit, shell/test and diff. Verifiers record pass/fail, bounded output and action/time caps. Only the reviewed task command is allowed in the initial shell interface.

`tools.agent_demo` records inspect -> hypothesize -> wrong edit -> failed test -> repeated failure -> replan -> repair -> passing test -> diff. It records hashes before/after each action and flags repeated action/result with unchanged state. The fixture is scripted infrastructure evidence, not a model-generated successful trajectory. Failed and looping events remain in the recording.

This disposable copy is not an OS security boundary. Future untrusted model shell/code execution requires a container/VM with network/resource isolation. Path escapes and non-allowlisted edits/commands are rejected now. There is no anti-loop neural head; initially learn/evaluate loop behavior from trajectories and state progress.
