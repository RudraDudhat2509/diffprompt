"""Fast path: diffprompt's own verdict pipeline (run_both -> batch_similarity -> judge_all -> regression_score -> _compute_verdict),
unmodified, on the 25 labeled pairs x 3 repeats, all concurrently. Skips ontology anchors, clustering, slicing and key-example
selection, which do not feed the overall verdict. Same runtime-only OpenAI model redirect as run_pairs.py."""
import asyncio, json, sys, time, pathlib
HERE = pathlib.Path(__file__).parent
sys.argv = ["x"]; sys.path.insert(0, str(HERE))
import run_pairs as rp
rp.SEM_G = asyncio.Semaphore(16)
from diffprompt.cli import _load_test_file, _compute_verdict
from diffprompt.core.runner import run_both
from diffprompt.core.embedder import batch_similarity
from diffprompt.core.judge import judge_all
from diffprompt.core.scorer import regression_score
from diffprompt.models import DiffResult, Verdict
from pairs import PAIRS

TF = str(HERE / "inputs.jsonl")
OUT = HERE / "raw" / "fast_results.json"

async def one(name, v1, v2, label, rep, gate):
    async with gate:
        tcs = _load_test_file(TF)
        r1, r2 = await run_both(tcs, v1, v2, model="groq/gpt-4o-mini-runner")
        a = [r1[t.id].output for t in tcs]; b = [r2[t.id].output for t in tcs]
        sims = batch_similarity(list(zip(a, b)))
        js = await judge_all(tcs, a, b, sims)
        diffs = [DiffResult(test_case=t, v1_output=a[i], v2_output=b[i], similarity=sims[i], divergence=1 - sims[i],
                            verdict=js[i][0], reason=js[i][1], judge_confidence=js[i][2], v1_latency_ms=0, v2_latency_ms=0)
                 for i, t in enumerate(tcs)]
        ni = sum(d.verdict == Verdict.IMPROVEMENT for d in diffs); nr = sum(d.verdict == Verdict.REGRESSION for d in diffs)
        nn = len(diffs) - ni - nr
        res = dict(pair=name, label=label, rep=rep, verdict=_compute_verdict(ni, nr, nn).value, score=regression_score(diffs),
                   improved=ni, regressed=nr, neutral=nn)
        print(f"{name:22s} [{label:7s}] rep{rep}: {res['verdict']:11s} score {res['score']:5.1f} +{ni}/-{nr}/={nn}", flush=True)
        return res

async def main():
    t0 = time.time(); gate = asyncio.Semaphore(6)
    results = await asyncio.gather(*[one(n, v1, v2, l, r, gate) for r in range(3) for n, v1, v2, l in PAIRS])
    stats = dict(seconds=round(time.time() - t0, 1), runner_calls=rp.C["runner_calls"], judge_calls=rp.J["judge_calls"],
                 escalations=rp.C["escalations"], backend_errors=rp.C["backend_errors"], model_calls_total=rp.C["runner_calls"] + rp.C["small_judge_calls"] + rp.C["other_groq_calls"])
    OUT.write_text(json.dumps(dict(results=results, stats=stats), indent=1))
    print("DONE", stats, flush=True)

asyncio.run(main())
