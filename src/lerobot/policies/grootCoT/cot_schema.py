"""CoT schema, prompt builders, JSON parser, and session state for GR00T System-2.

This module is self-contained and has no heavy dependencies beyond the standard library.
It is imported lazily during CoT generation (not on every training step).
"""

from __future__ import annotations

import json
import re
import random
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Token IDs (Qwen3-VL tokenizer constants, stable across 8B Instruct/Thinking)
# ---------------------------------------------------------------------------
VISION_START_ID: int = 151652  # <|vision_start|>
VISION_END_ID: int = 151653    # <|vision_end|>
IMAGE_PAD_ID: int = 151655     # <|image_pad|>

SCHEMA_VERSION = "vla.plan.recap.v1"

# ---------------------------------------------------------------------------
# Task library hints (kept short so they fit in the context alongside images)
# ---------------------------------------------------------------------------
_DEX3_HINTS = """\
Task primitives (use in plan.steps[].skill):
  perceive | reach | grasp | regrasp | lift | transport | align | place | release | wait | handover | inspect | abort
Dex3 task recipes:
  BlockStacking: perceive→pick red→align on tape→place red→verify→pick yellow→place on red→verify→pick blue→place on yellow→verify tower stable
  ObjectPlacement: perceive→for each object: pick→transport→align→place in container→verify
  Pouring: pick bottle (right hand)→align above cup→tilt pour→return upright→place bottle
  CameraPackaging: open case→pick camera→insert→close lid→inspect closure
  ToastedBread: pick bread→insert into toaster→wait→retrieve toast→handover to human
Replanning triggers: tower_unstable | dropped_object | wrong_object | spill | no_progress (>5 retries)"""

_HUMANOID_HINTS = """\
Task primitives (use in plan.steps[].skill):
  perceive | navigate | reach | grasp | regrasp | lift | transport | align | place | release | wait | handover | inspect | abort
Humanoid-Everyday: use depth/LiDAR for navigation, tactile for grasp confirm, IMU for balance.
Replanning triggers: obstacle_detected | fall_risk | grasp_slip | no_progress | wrong_target"""

_TASK_HINTS: dict[str, str] = {
    "dex3": _DEX3_HINTS,
    "humanoid_everyday": _HUMANOID_HINTS,
}

# ---------------------------------------------------------------------------
# System prompt (shared for INIT and TICK)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are System-2 (planner+critic) for a humanoid robot.
Output ONE valid JSON object following schema_version "vla.plan.recap.v1". No prose, no markdown fences — raw JSON only.
Rules:
1. Put ALL reasoning in cot.thoughts[] (array of strings). Think explicitly step by step.
2. INIT mode: produce a full plan (3–12 steps). TICK mode: keep existing plan unless accidents force replanning.
3. RECAP labels are mandatory every call: recap.reward_label.r_t, recap.value_target.v_hat, recap.advantage.indicator_I.
4. indicator_I must be exactly "POS", "NEG", or "DROPPED". Set "DROPPED" with ~30% probability to simulate dropout.
5. Estimate v_hat as -(remaining_steps_estimate) / max_episode_steps, clipped to [-1, 0].
6. next.system1_subtask_text must be a short imperative instruction (≤15 words) for the low-level action policy."""

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def detect_dataset_family(dataset_name: str) -> str:
    """Heuristically detect dex3 vs humanoid_everyday from dataset name."""
    dex3_kw = {"blockstacking", "objectplacement", "pouring", "camerapackaging",
                "toastedbread", "dex3", "g1", "unitreerobotics"}
    nl = dataset_name.lower().replace("-", "").replace("_", "")
    if any(kw in nl for kw in dex3_kw):
        return "dex3"
    return "humanoid_everyday"


def _meta_block(meta: dict) -> str:
    lines = [
        f'DATASET: {meta.get("dataset_name", "unknown")} (family: {meta.get("dataset_family", "dex3")})',
        f'INSTRUCTION: {meta.get("instruction", "Perform the task.")}',
        f'TIME_INDEX: {meta.get("time_index", 0)}',
        f'SYSTEM2_PERIOD: {meta.get("system2_period", 4)} steps',
        f'CAMERAS: {", ".join(meta.get("cameras_present", ["head_left", "head_right", "wrist_left", "wrist_right"]))}',
        f'EXTRA_MODALITIES: {", ".join(meta.get("extra_modalities_present", ["none"]))}',
    ]
    return "\n".join(lines)


def build_init_prompt(meta: dict) -> tuple[str, str]:
    """Return (system_text, user_text) for INIT mode.

    meta keys: dataset_name, dataset_family, instruction, time_index,
               system2_period, cameras_present, extra_modalities_present,
               time_horizon_s, terminal_failure_cost.
    """
    family = meta.get("dataset_family", "dex3")
    hints = _TASK_HINTS.get(family, _DEX3_HINTS)
    c_fail = meta.get("terminal_failure_cost", 100)
    max_steps = int(meta.get("time_horizon_s", 40) * 30)  # 30 Hz

    user = f"""\
MODE: INIT
{_meta_block(meta)}
TIME_HORIZON_S: {meta.get("time_horizon_s", 40)}
MAX_EPISODE_STEPS: {max_steps}

{hints}

Current robot observations are shown in the images above.

Output ONE JSON object with these fields filled:
  schema_version = "{SCHEMA_VERSION}"
  mode = "INIT"
  meta: fill dataset_family, dataset_name, instruction, time_index, system2_period, cameras_present, extra_modalities_present
  cot.thoughts: array of explicit reasoning strings (think step by step about what you see and what to do)
  world_model: list visible objects, agents, risks with conf scores
  plan_update = "replace"
  plan: steps (3–12), replan_triggers, termination
      - plan.steps[].skill must be from: perceive|navigate|reach|grasp|regrasp|lift|transport|align|place|release|wait|handover|inspect|abort
      - plan.steps[].args.hand: "left"|"right"|"both"|"either"
  execution_state: active_step_id = plan.steps[0].step_id, active_step_status = "ongoing", retries_used = 0
  next.decision = "continue"
  next.system1_subtask_text: short first instruction for System-1
  next.expected_horizon_steps: integer estimate
  events: all false initially
  progress.phi_total: 0.0
  recap.episode_done = false
  recap.episode_success = null
  recap.reward_spec: step_penalty=-1.0, terminal_success=0.0, terminal_failure={-abs(c_fail)}
  recap.reward_label.r_t = -1.0
  recap.value_target.v_hat: estimate -(remaining steps)/{max_steps}, clipped [-1,0]
  recap.advantage.indicator_I: "POS" (first step, assuming plan is good); roll 30% chance of "DROPPED"
"""
    return SYSTEM_PROMPT, user


def build_tick_prompt(meta: dict, last_plan: dict | None, exec_state: dict | None) -> tuple[str, str]:
    """Return (system_text, user_text) for TICK mode."""
    family = meta.get("dataset_family", "dex3")
    hints = _TASK_HINTS.get(family, _DEX3_HINTS)
    max_steps = int(meta.get("time_horizon_s", 40) * 30)
    c_fail = meta.get("terminal_failure_cost", 100)

    plan_json = json.dumps(last_plan, indent=None) if last_plan else "null"
    exec_json = json.dumps(exec_state, indent=None) if exec_state else "null"

    user = f"""\
MODE: TICK
{_meta_block(meta)}
TIME_HORIZON_S: {meta.get("time_horizon_s", 40)}
MAX_EPISODE_STEPS: {max_steps}

{hints}

CURRENT_PLAN (from previous System-2 call):
{plan_json}

EXECUTION_STATE (from previous System-2 call):
{exec_json}

Current robot observations are shown in the images above.

Output ONE JSON object:
  schema_version = "{SCHEMA_VERSION}"
  mode = "TICK"
  meta: same as before, update time_index
  cot.thoughts: reason about what has changed since last call, whether the plan is on track, any accidents
  world_model: update object states based on current images
  plan_update: "keep" (default) | "patch" (minor fix) | "replace" (major accident/failure)
  plan: if plan_update="keep", copy previous plan unchanged; if "patch"/"replace", update accordingly
  execution_state: update active_step_id, active_step_status, retries_used, stall_counter
  next.decision: "continue"|"advance"|"recover"|"replan"|"abort"
  next.system1_subtask_text: current short instruction for System-1 (≤15 words)
  next.expected_horizon_steps: remaining steps estimate
  events: detect dropped_object, wrong_object, spill, tower_unstable (true/false + conf)
  progress.phi_total: fraction of plan completed [0,1]
  recap.episode_done: true if task finished
  recap.episode_success: true/false/null
  recap.reward_spec: step_penalty=-1.0, terminal_success=0.0, terminal_failure={-abs(c_fail)}
  recap.reward_label.r_t: -1.0 if ongoing; 0.0 if success; {-abs(c_fail)} if failure
  recap.value_target.v_hat: -(remaining_steps_estimate)/{max_steps}, clipped [-1,0]
  recap.advantage.indicator_I: "POS" if action improves progress, "NEG" if worsening, "DROPPED" with ~30% probability
"""
    return SYSTEM_PROMPT, user


# ---------------------------------------------------------------------------
# JSON parser (robust to LLM formatting quirks)
# ---------------------------------------------------------------------------

def parse_cot_json(raw_text: str) -> tuple[dict | None, str | None]:
    """Try to parse a JSON dict from LLM output.

    Returns (parsed_dict, None) on success, (None, error_msg) on failure.
    Tries multiple strategies to handle common LLM formatting issues.
    """
    if not raw_text:
        return None, "empty output"

    text = raw_text.strip()

    # Strategy 1: direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj, None
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown code fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if fence_match:
        try:
            obj = json.loads(fence_match.group(1))
            if isinstance(obj, dict):
                return obj, None
        except json.JSONDecodeError:
            pass

    # Strategy 3: find outermost { ... }
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        candidate = text[brace_start : brace_end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj, None
        except json.JSONDecodeError:
            pass

    # Strategy 4: try to repair common issues (trailing commas, single quotes)
    if brace_start != -1 and brace_end > brace_start:
        candidate = text[brace_start : brace_end + 1]
        # Remove trailing commas before } or ]
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        # Replace Python None/True/False with JSON null/true/false
        candidate = candidate.replace(": None", ": null").replace(":None", ":null")
        candidate = candidate.replace(": True", ": true").replace(":True", ":true")
        candidate = candidate.replace(": False", ": false").replace(":False", ":false")
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj, None
        except json.JSONDecodeError as e:
            return None, f"parse_failed_after_repair: {e}"

    return None, f"no_json_found (len={len(text)})"


# ---------------------------------------------------------------------------
# Session state (tracks INIT/TICK across System-2 calls per episode)
# ---------------------------------------------------------------------------

@dataclass
class CoTSessionState:
    """Tracks mode and plan across System-2 calls within one episode.

    step_count == 0 → emit INIT prompt
    step_count > 0  → emit TICK prompt with last_plan embedded
    """
    episode_index: int
    instruction: str
    dataset_name: str
    dataset_family: str
    time_horizon_s: float = 40.0
    terminal_failure_cost: float = 100.0
    system2_period: int = 4
    cameras_present: list[str] = field(default_factory=lambda: [
        "head_left", "head_right", "wrist_left", "wrist_right"
    ])
    extra_modalities_present: list[str] = field(default_factory=lambda: ["none"])

    step_count: int = 0
    last_plan: dict[str, Any] | None = None
    last_execution_state: dict[str, Any] | None = None
    last_json: dict[str, Any] | None = None

    @property
    def mode(self) -> str:
        return "INIT" if self.step_count == 0 else "TICK"

    def build_meta(self, time_index: int = 0) -> dict:
        return {
            "dataset_family": self.dataset_family,
            "dataset_name": self.dataset_name,
            "instruction": self.instruction,
            "time_index": time_index,
            "system2_period": self.system2_period,
            "cameras_present": self.cameras_present,
            "extra_modalities_present": self.extra_modalities_present,
            "time_horizon_s": self.time_horizon_s,
            "terminal_failure_cost": self.terminal_failure_cost,
        }

    def get_prompts(self, time_index: int = 0) -> tuple[str, str]:
        """Return (system_text, user_text) for the current mode."""
        meta = self.build_meta(time_index)
        if self.mode == "INIT":
            return build_init_prompt(meta)
        return build_tick_prompt(meta, self.last_plan, self.last_execution_state)

    def advance(self, parsed_json: dict | None) -> None:
        """Update session from the latest parsed JSON output."""
        self.step_count += 1
        if parsed_json is None:
            return
        self.last_json = parsed_json
        # Extract plan
        if "plan" in parsed_json and isinstance(parsed_json["plan"], dict):
            self.last_plan = parsed_json["plan"]
        # Extract execution_state
        if "execution_state" in parsed_json and isinstance(parsed_json["execution_state"], dict):
            self.last_execution_state = parsed_json["execution_state"]

    def reset(self, episode_index: int, instruction: str, dataset_name: str | None = None) -> None:
        """Reset for a new episode."""
        self.episode_index = episode_index
        self.instruction = instruction
        if dataset_name is not None:
            self.dataset_name = dataset_name
            self.dataset_family = detect_dataset_family(dataset_name)
        self.step_count = 0
        self.last_plan = None
        self.last_execution_state = None
        self.last_json = None


# ---------------------------------------------------------------------------
# Session registry (maps episode_index → CoTSessionState)
# ---------------------------------------------------------------------------

class CoTSessionRegistry:
    """Manages per-episode CoTSessionState objects during training."""

    def __init__(
        self,
        *,
        dataset_name: str = "unknown",
        dataset_family: str | None = None,
        time_horizon_s: float = 40.0,
        terminal_failure_cost: float = 100.0,
        system2_period: int = 4,
        cameras_present: list[str] | None = None,
        extra_modalities_present: list[str] | None = None,
    ):
        self._default_dataset_name = dataset_name
        self._default_dataset_family = (
            dataset_family if dataset_family is not None else detect_dataset_family(dataset_name)
        )
        self._time_horizon_s = time_horizon_s
        self._terminal_failure_cost = terminal_failure_cost
        self._system2_period = system2_period
        self._cameras_present = cameras_present or ["head_left", "head_right", "wrist_left", "wrist_right"]
        self._extra_modalities_present = extra_modalities_present or ["none"]
        self._sessions: dict[int, CoTSessionState] = {}

    def get_or_create(self, episode_index: int, instruction: str) -> CoTSessionState:
        """Return existing session for this episode, or create a fresh INIT session."""
        if episode_index not in self._sessions:
            self._sessions[episode_index] = CoTSessionState(
                episode_index=episode_index,
                instruction=instruction,
                dataset_name=self._default_dataset_name,
                dataset_family=self._default_dataset_family,
                time_horizon_s=self._time_horizon_s,
                terminal_failure_cost=self._terminal_failure_cost,
                system2_period=self._system2_period,
                cameras_present=list(self._cameras_present),
                extra_modalities_present=list(self._extra_modalities_present),
            )
        return self._sessions[episode_index]

    def evict_old(self, keep_recent: int = 64) -> None:
        """Evict oldest sessions to avoid unbounded memory growth."""
        if len(self._sessions) > keep_recent * 2:
            sorted_keys = sorted(self._sessions.keys())
            for k in sorted_keys[: len(sorted_keys) - keep_recent]:
                del self._sessions[k]
