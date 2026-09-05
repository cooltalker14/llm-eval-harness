"""Run a model over the golden dataset and cache the raw outputs.

Caching is keyed on (model, prompt, sampling params). Re-scoring should never
require re-inference: scorers change often, model outputs do not. This also
makes the eval reproducible, since the cached outputs can be committed.
"""

import asyncio
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import aiohttp

from .dataset import Item


@dataclass
class Generation:
    item_id: str
    output: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    error: str = ""


@dataclass
class RunResult:
    tag: str
    model: str
    generations: list[Generation] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    wall_time_s: float = 0.0
    n_errors: int = 0
    env: dict = field(default_factory=dict)


def _cache_key(model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    h = hashlib.sha256()
    h.update(f"{model}|{temperature}|{max_tokens}|{prompt}".encode())
    return h.hexdigest()[:16]


async def _one(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    item: Item,
    temperature: float,
    max_tokens: int,
) -> Generation:
    payload = {
        "model": model,
        "prompt": item.prompt(),
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    t0 = time.perf_counter()
    try:
        async with session.post(
            f"{base_url}/v1/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                return Generation(item.id, "", error=f"HTTP {resp.status}: {body[:200]}")
            data = await resp.json()
    except Exception as e:  # noqa: BLE001 - any failure is a failed generation
        return Generation(item.id, "", error=f"{type(e).__name__}: {e}")

    usage = data.get("usage") or {}
    return Generation(
        item_id=item.id,
        output=(data.get("choices") or [{}])[0].get("text", ""),
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        latency_s=time.perf_counter() - t0,
    )


async def run(
    items: list[Item],
    base_url: str,
    model: str,
    tag: str,
    temperature: float = 0.0,
    max_tokens: int = 200,
    concurrency: int = 8,
    cache_dir: Path | None = None,
) -> RunResult:
    """Generate outputs for every item, using cache where available.

    temperature=0 by default: an eval comparing two configurations should not
    also be measuring sampling noise. Non-zero temperature requires multiple
    samples per item to be meaningful, which is a different experiment.
    """
    cache: dict[str, str] = {}
    cache_file = None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{tag}.json"
        if cache_file.exists():
            cache = json.loads(cache_file.read_text())

    gens: dict[str, Generation] = {}
    todo: list[Item] = []
    for it in items:
        key = _cache_key(model, it.prompt(), temperature, max_tokens)
        if key in cache:
            gens[it.id] = Generation(it.id, cache[key])
        else:
            todo.append(it)

    if todo:
        print(f"[run:{tag}] {len(gens)} cached, generating {len(todo)}", flush=True)
    else:
        print(f"[run:{tag}] all {len(gens)} items cached", flush=True)

    t0 = time.perf_counter()
    if todo:
        sem = asyncio.Semaphore(concurrency)
        conn = aiohttp.TCPConnector(limit=concurrency * 2)
        async with aiohttp.ClientSession(connector=conn) as session:

            async def guarded(it: Item):
                async with sem:
                    return await _one(session, base_url, model, it, temperature, max_tokens)

            for g in await asyncio.gather(*[guarded(it) for it in todo]):
                gens[g.item_id] = g
                if not g.error:
                    it = next(x for x in todo if x.id == g.item_id)
                    cache[_cache_key(model, it.prompt(), temperature, max_tokens)] = g.output

    wall = time.perf_counter() - t0

    if cache_file:
        cache_file.write_text(json.dumps(cache, indent=2))

    ordered = [gens[it.id] for it in items]
    return RunResult(
        tag=tag,
        model=model,
        generations=ordered,
        total_prompt_tokens=sum(g.prompt_tokens for g in ordered),
        total_completion_tokens=sum(g.completion_tokens for g in ordered),
        wall_time_s=wall,
        n_errors=sum(1 for g in ordered if g.error),
    )


def save(result: RunResult, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), indent=2))


def load(path: Path) -> RunResult:
    d = json.loads(Path(path).read_text())
    d["generations"] = [Generation(**g) for g in d["generations"]]
    return RunResult(**d)
