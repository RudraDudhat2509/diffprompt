"""Run diffprompt's own pipeline (diffprompt.cli._run_diff, unmodified) on the 25 labeled pairs, 3 repeats each.

Only the MODEL BACKENDS are redirected at runtime (no diffprompt source is edited), because this machine has no
Ollama qwen2.5:7b and the Groq key no longer has the hard-coded Llama models:
  runner               groq/gpt-4o-mini-runner   -> OpenAI gpt-4o-mini (T=0.7)
  first-pass judge     llama-3.1-8b-instant      -> OpenAI gpt-4o-mini (T=0)
  escalation judge     llama-3.3-70b-versatile   -> OpenAI gpt-4o      (T=0)
(Groq was tried first: its free tier is 1,000 requests/day and 8,000 tokens/min, too small for ~3,000 calls.)
  local Ollama                                   -> disabled (returns None immediately)
Usage: python benchmarks/run_pairs.py [reps] [name-prefix]
"""
import asyncio, json, os, pathlib, sys, time
HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
for l in (pathlib.Path(os.environ["BENCH_ENV_FILE"]).read_text().splitlines() if os.environ.get("BENCH_ENV_FILE") else []):  # keys: OPEN_API_KEY (and GROQ_API_KEY); or just export them
    if "=" in l and not l.startswith("#"):
        a, _, b = l.partition("="); os.environ[a.strip()] = b.strip().strip('"').strip("'")

from openai import AsyncOpenAI
import diffprompt.models.cascade as cascade
import diffprompt.core.judge as judge_mod

oai = AsyncOpenAI(api_key=os.environ["OPEN_API_KEY"])
SEM_G = asyncio.Semaphore(8)
C = {"runner_calls": 0, "small_judge_calls": 0, "escalations": 0, "other_groq_calls": 0, "backend_errors": 0}

async def fake_ollama(model, prompt, system=None):
    return None

async def routed_groq(model, prompt, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    target, temp = {"gpt-4o-mini-runner": ("gpt-4o-mini", 0.7), "llama-3.1-8b-instant": ("gpt-4o-mini", 0.0),
                    "llama-3.3-70b-versatile": ("gpt-4o", 0.0)}.get(model, ("gpt-4o-mini", 0.0))
    C["runner_calls" if model == "gpt-4o-mini-runner" else "small_judge_calls" if model == "llama-3.1-8b-instant" else "other_groq_calls"] += 1
    for attempt in range(4):
        try:
            async with SEM_G:
                r = await oai.chat.completions.create(model=target, messages=msgs, temperature=temp, max_tokens=400)
            return r.choices[0].message.content
        except Exception:
            C["backend_errors"] += 1
            await asyncio.sleep(2 * (attempt + 1))
    return None

_orig_groq_only = judge_mod.call_groq_only
async def counting_groq_only(*a, **k):
    C["escalations"] += 1
    return await _orig_groq_only(*a, **k)

_orig_cascade_j = judge_mod.call_cascade
J = {"judge_calls": 0}
async def counting_cascade(*a, **k):
    J["judge_calls"] += 1
    return await _orig_cascade_j(*a, **k)

cascade.call_ollama = fake_ollama
cascade.call_groq = routed_groq
judge_mod.call_groq_only = counting_groq_only
judge_mod.call_cascade = counting_cascade

from diffprompt.cli import _run_diff
from pairs import PAIRS

OUT = HERE / "raw" / "runs"; OUT.mkdir(parents=True, exist_ok=True)

async def one(name, v1, v2, rep, test_file=None, n=10, auto=False, tag=None):
    tag = tag or f"{name.split()[0]}_r{rep}"
    path = OUT / f"{tag}.json"
    if path.exists():
        return json.loads(path.read_text())["summary"]
    for k in C: C[k] = 0
    J["judge_calls"] = 0
    t0 = time.time()
    await _run_diff(prompt_v1=v1, prompt_v2=v2, auto_generate=auto, n=n, test_file=test_file,
                    model="groq/gpt-4o-mini-runner", judge="local/qwen2.5:7b", local_only=False, no_judge=False,
                    output_format="json", save=str(path.with_suffix(".report.json")), top_n=3, quiet=True,
                    verbose=False, ci=False, threshold=75)
    rep_json = json.loads(path.with_suffix(".report.json").read_text(encoding="utf-8"))
    s = dict(pair=name, rep=rep, verdict=rep_json["verdict"], score=rep_json["regression_score"],
             improved=rep_json["n_improved"], regressed=rep_json["n_regressed"], neutral=rep_json["n_neutral"],
             clusters=len(rep_json.get("clusters", [])), seconds=round(time.time() - t0, 1), judge_calls=J["judge_calls"], **C)
    path.write_text(json.dumps({"summary": s}))
    return s

async def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    prefixes = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("",)
    tf = str(HERE / "inputs.jsonl")
    for rep in range(reps):  # all pairs once before any repeat, so partial runs still cover every pair
        for name, v1, v2, label in PAIRS:
            if not name.startswith(prefixes): continue
            s = await one(name, v1, v2, rep, test_file=tf)
            print(f"{name:24s} [{label:7s}] rep{rep}: {s['verdict']:11s} score {s['score']:5.1f}  +{s['improved']}/-{s['regressed']}/={s['neutral']}  "
                  f"judge {s['judge_calls']} esc {s['escalations']} errs {s['backend_errors']}  {s['seconds']}s", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
