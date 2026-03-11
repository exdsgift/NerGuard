"""
NerGuard Interactive REPL — persistent test session with model kept hot in memory.

Usage:
    ./test.sh
    uv run python -m src.scripts.repl
"""

import json
import os
import readline
import signal
import sys
import time
import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from src.core.constants import DEFAULT_MODEL_PATH

# ── Color palette — hacker: bright blue + white scale ────────────────────────
#
#   B   bright blue    \033[1;94m  — logo, labels, prompt ❯, separators
#   W   bright white   \033[1;97m  — redacted output (dominant)
#   G   dim gray       \033[38;5;240m — input text (muted)
#   DW  dim gray-blue  \033[2;34m  — metadata, status bar, key labels
#   R   bright red     \033[1;91m  — errors
#
B  = "\033[1;38;5;27m"   # bold deep royal/electric blue — accent, logo, labels
W  = "\033[1;97m"        # bold bright white — redacted (dominant foreground)
G  = "\033[38;5;252m"    # light gray        — input text (muted but legible on dark bg)
DW = "\033[38;5;245m"    # medium gray       — separators, metadata, hints
GR = "\033[1;92m"        # bright green      — local LLM privacy (positive)
Y  = "\033[1;33m"        # amber/yellow      — cloud privacy warning
R  = "\033[1;91m"        # bright red        — errors
RS = "\033[0m"           # reset

SEP = f"{B}  {'─' * 53}{RS}"

_VERSION = "1.0.0"

BANNER = f"""\n\n\
{B}   ▄   ▄███▄   █▄▄▄▄   ▄▀    ▄   ██   █▄▄▄▄ ██▄   {RS}
{B}    █  █▀   ▀  █  ▄▀ ▄▀       █  █ █  █  ▄▀ █  █  {RS}
{B}██   █ ██▄▄    █▀▀▌  █ ▀▄  █   █ █▄▄█ █▀▀▌  █   █ {RS}
{B}█ █  █ █▄   ▄▀ █  █  █   █ █   █ █  █ █  █  █  █  {RS}
{B}█  █ █ ▀███▀     █    ███  █▄ ▄█    █   █   ███▀   {RS}
{B}█   ██          ▀           ▀▀▀    █   ▀            {RS}
{B}                                  ▀                 {RS}\n"""

_MODES = ["human", "rag", "json", "generic"]


@dataclass
class Session:
    mode: str = "human"
    llm: bool = False
    backend: str = "openai"
    model_name: str = "gpt-4o"
    labels: bool = True    # show entity label rows
    mapping: bool = False  # show entity→value map (rag/generic)
    model_path: str = DEFAULT_MODEL_PATH
    # populated after warmup — used to reprint header on /clear
    model_base: str = ""
    model_arch: str = ""
    model_params: str = ""
    model_labels: str = ""
    model_device: str = ""
    # cached RAG instances keyed by (model_path, llm_routing, backend, model_name, typed)
    _rag_cache: dict = None

    def __post_init__(self):
        self._rag_cache = {}


# ── Model caching — loaded once, reused across all queries ────────────────────
_model_cache: dict = {}


def _make_cached_loader(original_fn):
    def cached(model_path=DEFAULT_MODEL_PATH, device=None, eval_mode=True, use_fast=True):
        key = model_path
        if key not in _model_cache:
            _model_cache[key] = original_fn(
                model_path, device=device, eval_mode=eval_mode, use_fast=use_fast
            )
        return _model_cache[key]
    return cached


def _install_model_cache():
    from src.core.model_loader import load_model_and_tokenizer as original
    cached = _make_cached_loader(original)
    import src.scripts.redact as _redact_mod
    _redact_mod.load_model_and_tokenizer = cached


# ── Ollama helpers ────────────────────────────────────────────────────────────
def _ollama_list() -> tuple[bool, str, list[tuple[float, str]]]:
    """Single subprocess call: return (ok, error_msg, [(size_mb, name), ...])."""
    import shutil
    import subprocess
    if shutil.which("ollama") is None:
        return False, "ollama is not installed  (brew install ollama / apt install ollama)", []
    try:
        r = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=3)
        if r.returncode != 0:
            return False, "ollama daemon is not running  (run: ollama serve)", []
    except Exception:
        return False, "ollama daemon is not running  (run: ollama serve)", []

    models = []
    for line in r.stdout.strip().splitlines()[1:]:
        parts = line.split()
        if not parts:
            continue
        name = parts[0]
        size_val = None
        for i, p in enumerate(parts):
            if p in ("GB", "GiB") and i > 0:
                try:
                    size_val = float(parts[i - 1]) * 1024
                except ValueError:
                    pass
            elif p in ("MB", "MiB") and i > 0:
                try:
                    size_val = float(parts[i - 1])
                except ValueError:
                    pass
        models.append((size_val if size_val is not None else float("inf"), name))
    models.sort()
    return True, "", models


# ── Privacy helpers ───────────────────────────────────────────────────────────
def _backend_privacy(backend: str) -> tuple[str, str]:
    """Privacy level implied by a backend choice (LLM-on perspective)."""
    if backend == "ollama":
        return "local LLM", f"{GR}local LLM{RS}"
    return "cloud", f"{Y}cloud  ·  data sent to OpenAI{RS}"


def _privacy(s: Session) -> tuple[str, str]:
    """Return (label, colored_string) for current effective privacy level."""
    if not s.llm:
        return "full local", f"{B}full local{RS}"
    return _backend_privacy(s.backend)


# ── Display helpers ───────────────────────────────────────────────────────────
def _print_status(s: Session):
    llm_str  = f"llm {B}on{RS}  {DW}{s.backend}/{s.model_name}" if s.llm else f"llm {DW}off"
    lbl_str  = f"labels {B}on{RS}" if s.labels else f"labels {DW}off"
    map_str  = f"mapping {B}on{RS}" if s.mapping else f"mapping {DW}off"
    _, priv  = _privacy(s)
    print(f"\n{DW}  mode {B}{s.mode}{RS}   {DW}{llm_str}{RS}   {DW}{lbl_str}{RS}   {DW}{map_str}{RS}   {DW}privacy {RS}{priv}")


def _prompt_str(s: Session) -> str:
    """Return readline-safe prompt (ANSI codes wrapped in \\001/\\002)."""
    rl_dw = "\001\033[38;5;245m\002"
    rl_b  = "\001\033[1;38;5;27m\002"
    rl_rs = "\001\033[0m\002"
    return f"{rl_dw}{s.mode}{rl_rs} {rl_b}❯{rl_rs} "


def _print_hint():
    """Hint line printed above the prompt — stays clean, no cursor tricks."""
    print(f"\n\n\n{DW}  enter text to redact, or /help{RS}\n")


def _print_result(text: str, entities, redacted: str, show_labels: bool, mapping: dict = None):
    """Unified result block:  input (gray) → redacted (white) → [labels + mapping]."""
    print()
    print(f"  {DW}input    {RS}{G}{text}{RS}")
    print()
    print(f"  {DW}redacted {RS}{W}{redacted}{RS}")

    show_bottom = (show_labels and entities) or mapping
    if show_bottom:
        print()
        print(SEP)
        if show_labels and entities:
            max_lbl = max(len(e["label"]) for e in entities)
            max_val = max(len(e["text"]) for e in entities)
            for e in entities:
                lbl  = f"{B}{e['label']:<{max_lbl}}{RS}"
                val  = f"{W}{e['text']}{RS}"
                pad  = " " * (max_val - len(e["text"]))
                meta = f"{DW}{e['source']:<16s}  conf {e['confidence']:.3f}{RS}"
                print(f"  {lbl}   {val}{pad}   {meta}")
        if mapping:
            if show_labels and entities:
                print()
            pairs = "  ·  ".join(f"{DW}{k}{RS} → {W}{v}{RS}" for k, v in mapping.items())
            print(f"  {DW}mapping  {RS}{pairs}")
        print(SEP)

    print()


# ── Inference runners ─────────────────────────────────────────────────────────
def _run(text: str, s: Session):
    if s.mode == "human":
        from src.scripts.redact import redact_pipeline
        entities, redacted = redact_pipeline(
            text=text, model_path=s.model_path,
            llm_routing=s.llm, llm_source=s.backend, llm_model=s.model_name,
        )
        _print_result(text, entities, redacted, show_labels=s.labels)

    elif s.mode in ("rag", "generic"):
        from src.rag.redactor import nerguard as NerGuardRAG
        typed = s.mode == "rag"
        rag_key = (s.model_path, s.llm, s.backend, s.model_name, typed)
        if rag_key not in s._rag_cache:
            s._rag_cache[rag_key] = NerGuardRAG(
                model_path=s.model_path, llm_routing=s.llm,
                llm_source=s.backend, llm_model=s.model_name,
                typed=typed,
            )
        result = s._rag_cache[rag_key].redact(text)
        mapping = result.mapping if s.mapping else None
        _print_result(text, result.entities, result.text, show_labels=s.labels, mapping=mapping)

    elif s.mode == "json":
        from src.scripts.redact import redact_pipeline
        entities, redacted = redact_pipeline(
            text=text, model_path=s.model_path,
            llm_routing=s.llm, llm_source=s.backend, llm_model=s.model_name,
        )
        data = {"input": text, "entities": entities, "redacted": redacted}
        print()
        print(SEP)
        print(json.dumps(data, indent=2, ensure_ascii=False))
        print(SEP)
        print()


# ── Command handlers ──────────────────────────────────────────────────────────
def _handle_command(cmd: str, s: Session) -> bool:
    """Returns False to quit, True to continue."""
    parts = cmd.strip().split(None, 1)
    name = parts[0].lower()
    arg  = parts[1].strip() if len(parts) > 1 else ""

    if name == "/quit":
        return False

    elif name == "/help":
        c = B    # command color
        a = DW   # arg color
        d = DW   # description color
        rows = [
            # (command,                   args,                   description)
            ("/mode",    "[human|rag|json|generic]", "output mode  (no arg → cycle)"),
            ("/llm",     "",                          "toggle LLM routing on / off"),
            ("/backend", "[openai|ollama]",           "switch LLM backend"),
            ("/model",   "NAME",                      "set model  (e.g. qwen2.5:7b)"),
            ("/labels",  "",                          "toggle entity label rows on / off"),
            ("/mapping", "",                          "toggle entity→value map  (rag/generic)"),
            ("/file",    "PATH",                      "redact an entire file"),
            ("/clear",   "",                          "clear the screen"),
            ("/help",    "",                          "show this help"),
            ("/quit",    "",                          "exit  (or Ctrl+D)"),
        ]
        col_cmd  = max(len(r[0]) for r in rows)
        col_arg  = max(len(r[1]) for r in rows)
        print(f"\n{SEP}")
        for cmd, arg, desc in rows:
            cmd_s = f"{c}{cmd:<{col_cmd}}{RS}"
            arg_s = f"{a}{arg:<{col_arg}}{RS}" if arg else f"{' ' * col_arg}"
            print(f"  {cmd_s}  {arg_s}   {d}{desc}{RS}")
        print(f"{SEP}\n")

    elif name == "/clear":
        os.system("clear")
        _print_header(s)

    elif name == "/mode":
        if arg:
            if arg not in _MODES:
                print(f"  {R}unknown mode '{arg}'  —  choose: {', '.join(_MODES)}{RS}\n")
            else:
                s.mode = arg
                print(f"  {DW}mode → {B}{s.mode}{RS}\n")
        else:
            s.mode = _MODES[(_MODES.index(s.mode) + 1) % len(_MODES)]
            print(f"  {DW}mode → {B}{s.mode}{RS}\n")

    elif name == "/llm":
        s.llm = not s.llm
        state = f"{B}on{RS}" if s.llm else f"{DW}off{RS}"
        _, priv = _privacy(s)
        suffix = f"   {DW}(backend: {s.backend}){RS}" if s.llm else ""
        print(f"  {DW}LLM routing → {state}   privacy → {priv}{suffix}\n")

    elif name == "/backend":
        if arg not in ("openai", "ollama"):
            print(f"  {R}unknown backend '{arg}'  —  choose: openai, ollama{RS}\n")
        elif arg == "ollama":
            ok, err, models = _ollama_list()
            if not ok:
                print(f"  {R}{err}{RS}\n")
            else:
                s.backend = "ollama"
                _, priv = _backend_privacy("ollama")
                if models:
                    s.model_name = models[0][1]
                    print(f"  {DW}backend → {GR}ollama{RS}   {DW}model → {B}{s.model_name}{RS}  {DW}(smallest)   when LLM on: privacy → {priv}{RS}\n")
                else:
                    print(f"  {DW}backend → {GR}ollama{RS}   {R}no models found — run: ollama pull <model>{RS}\n")
        else:
            s.backend = arg
            _, priv = _backend_privacy(arg)
            print(f"  {DW}backend → {B}{s.backend}{RS}   {DW}when LLM on: privacy → {priv}{RS}\n")

    elif name == "/model":
        if not arg:
            print(f"  {DW}model: {B}{s.model_name}{RS}\n")
        else:
            s.model_name = arg
            print(f"  {DW}model → {B}{s.model_name}{RS}\n")

    elif name == "/labels":
        s.labels = not s.labels
        state = f"{B}on{RS}" if s.labels else f"{DW}off{RS}"
        print(f"  {DW}labels → {state}\n")

    elif name == "/mapping":
        s.mapping = not s.mapping
        state = f"{B}on{RS}" if s.mapping else f"{DW}off{RS}"
        print(f"  {DW}mapping → {state}\n")

    elif name == "/file":
        if not arg:
            print(f"  {R}usage: /file PATH{RS}\n")
        elif not os.path.isfile(arg):
            print(f"  {R}file not found: {arg}{RS}\n")
        else:
            with open(arg) as f:
                text = f.read().strip()
            if not text:
                print(f"  {R}file is empty{RS}\n")
            else:
                _run(text, s)

    else:
        print(f"  {R}unknown command '{name}'  —  type /help{RS}\n")

    return True


# ── Clean exit ───────────────────────────────────────────────────────────────
def _exit_clean():
    os.system("clear")
    # Reset handlers to default before sending SIGTERM to process group
    # to avoid re-entering this handler recursively.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    try:
        os.killpg(os.getpgid(0), signal.SIGTERM)
    except Exception:
        pass
    sys.exit(0)


def _install_signal_handlers():
    """Catch SIGTERM and SIGINT so _exit_clean() always runs on forced kills."""
    def _handler(sig, frame):  # noqa: ARG001
        _exit_clean()

    signal.signal(signal.SIGTERM, _handler)
    # SIGINT (Ctrl+C) is handled as KeyboardInterrupt inside the loop;
    # install a fallback here so a Ctrl+C *outside* the loop (e.g. during
    # model load or inference) still triggers a clean kill.
    signal.signal(signal.SIGINT, _handler)


# ── readline setup ────────────────────────────────────────────────────────────
def _setup_readline():
    history_file = os.path.expanduser("~/.nerguard_history")
    try:
        readline.read_history_file(history_file)
    except FileNotFoundError:
        pass
    readline.set_history_length(500)
    import atexit
    atexit.register(readline.write_history_file, history_file)


# ── Header (banner + model info + status) — reusable on /clear ───────────────
def _print_header(s: Session, show_hint: bool = False):
    print()
    print(BANNER)
    print()
    print(f"{DW}  NerGuard Interactive REPL  ·  v{_VERSION}{RS}")
    print()
    if s.model_base:
        print(f"{DW}  model    {RS}{W}{s.model_base}{RS}")
        print(f"{DW}  arch     {RS}{G}{s.model_arch}  ·  {s.model_params}  ·  {s.model_labels}  ·  {s.model_device}{RS}")
    _print_status(s)
    print(f"{B}  {'─' * 53}{RS}")
    if show_hint:
        _print_hint()


# ── Startup model warm-up ─────────────────────────────────────────────────────
def _warmup(s: Session):
    sys.stdout.write(f"{DW}  loading model ···{RS}")
    sys.stdout.flush()
    t0 = time.time()
    import torch
    import src.scripts.redact as _redact_mod  # already patched with cached loader
    from src.core.model_loader import get_device
    device = get_device()
    try:
        model, _ = _redact_mod.load_model_and_tokenizer(s.model_path, device=str(device), eval_mode=True)
    except torch.cuda.OutOfMemoryError:
        sys.stdout.write(f"\r{DW}  loading model ···  {Y}CUDA OOM — falling back to CPU{RS}  \n")
        sys.stdout.flush()
        device = torch.device("cpu")
        model, _ = _redact_mod.load_model_and_tokenizer(s.model_path, device="cpu", eval_mode=True)
    elapsed = time.time() - t0
    sys.stdout.write(f"\r{DW}  loading model ···  {B}✓{RS}  {DW}{elapsed:.1f}s{RS}          \n")
    sys.stdout.flush()

    cfg = model.config
    raw_base = getattr(cfg, "_name_or_path", s.model_path) or s.model_path
    # Local path → use friendly name
    s.model_base = "NerGuard-0.3B  (local)" if raw_base == s.model_path else raw_base
    s.model_arch   = type(model).__name__
    s.model_params = f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params"
    s.model_labels = f"{len(cfg.id2label) if hasattr(cfg, 'id2label') else '?'} labels"
    s.model_device = str(device).upper()


# ── Main REPL loop ────────────────────────────────────────────────────────────
def main():
    from dotenv import load_dotenv
    load_dotenv()

    _install_signal_handlers()
    _install_model_cache()
    _setup_readline()

    s = Session()

    os.system("clear")

    # Banner visible immediately while model loads
    print(BANNER)
    print()
    print(f"{DW}  NerGuard Interactive REPL  ·  v{_VERSION}{RS}")
    print()
    try:
        _warmup(s)   # loads model, stores info in s, prints spinner
    except Exception:
        import traceback
        print(f"\n  {R}failed to load model:{RS}\n")
        print(traceback.format_exc())
        sys.exit(1)

    # Model info + status + separator + hint
    print(f"{DW}  model    {RS}{W}{s.model_base}{RS}")
    print(f"{DW}  arch     {RS}{G}{s.model_arch}  ·  {s.model_params}  ·  {s.model_labels}  ·  {s.model_device}{RS}")
    _print_status(s)
    print(f"{B}  {'─' * 53}{RS}")
    _print_hint()

    while True:
        try:
            text = input(_prompt_str(s)).strip()
        except KeyboardInterrupt:
            print()
            continue
        except EOFError:
            _exit_clean()

        if not text:
            continue

        if text.startswith("/"):
            keep = _handle_command(text, s)
            if not keep:
                _exit_clean()
        else:
            try:
                _run(text, s)
            except Exception as exc:
                print(f"\n  {R}error: {exc}{RS}\n")


if __name__ == "__main__":
    main()
