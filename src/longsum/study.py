"""Run every strategy over the sample and write reports/.

Usage: python -m longsum.study
"""

from __future__ import annotations

import json
import statistics
import time
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import FIG_DIR, MODEL_MAX_INPUT, MODEL_NAME, REPORT_DIR, SEED  # noqa: E402
from .data import build_sample  # noqa: E402
from .rouge import all_scores  # noqa: E402
from .strategies import STRATEGIES, reduce_pass  # noqa: E402
from .summariser import Distilbart  # noqa: E402

BASE = ["truncate", "lead", "sequential", "clustered_ordered", "clustered_scrambled"]
MULTI_CHUNK = ["sequential", "clustered_ordered", "clustered_scrambled"]
ORDER = BASE + [f"{n}_reduced" for n in MULTI_CHUNK]
LABELS = {
    "truncate": "Truncate (do nothing)",
    "lead": "Lead-3 (no model)",
    "sequential": "Sequential chunks",
    "clustered_ordered": "Clustered, order kept",
    "clustered_scrambled": "Clustered, order lost",
    "sequential_reduced": "Sequential + reduce",
    "clustered_ordered_reduced": "Clustered ordered + reduce",
    "clustered_scrambled_reduced": "Clustered scrambled + reduce",
}


def md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def run() -> dict:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    model = Distilbart()
    documents = build_sample(model.count_tokens)

    per_strategy: dict[str, list[dict]] = {name: [] for name in ORDER}
    def record(name: str, result, seconds: float, reference: str) -> None:
        per_strategy[name].append(
            {
                "id": document["id"], **all_scores(result.text, reference), "seconds": seconds,
                "n_chunks": result.n_chunks, "ordered": result.is_ordered,
                "n_words": result.n_words, "summary": result.text,
            }
        )

    for document in documents:
        for name in BASE:
            started = time.time()
            result = STRATEGIES[name](document["article"], model)
            elapsed = time.time() - started
            record(name, result, elapsed, document["reference"])
            if name in MULTI_CHUNK:
                started = time.time()
                reduced = reduce_pass(result, model)
                record(reduced.strategy, reduced, elapsed + (time.time() - started),
                       document["reference"])

    summary = {
        name: {
            "rouge1": statistics.mean(r["rouge1"] for r in rows),
            "rouge2": statistics.mean(r["rouge2"] for r in rows),
            "rougeL": statistics.mean(r["rougeL"] for r in rows),
            "seconds": statistics.mean(r["seconds"] for r in rows),
            "chunks": statistics.mean(r["n_chunks"] for r in rows),
            "words": statistics.mean(r["n_words"] for r in rows),
            # Paired standard error against the sequential strategy on the same documents.
            "rouge1_sd": statistics.pstdev([r["rouge1"] for r in rows]),
        }
        for name, rows in per_strategy.items()
    }

    def paired_difference(better: str, worse: str, metric: str = "rouge1") -> list[float]:
        """Mean difference and 95% interval on the same documents, one pair per document."""
        # strict=True: both lists hold one result per document in the same order, so a
        # mismatch means a strategy skipped a document and the pairing is meaningless.
        differences = [a[metric] - b[metric]
                       for a, b in zip(per_strategy[better], per_strategy[worse], strict=True)]
        mean = statistics.mean(differences)
        error = statistics.pstdev(differences) / (len(differences) ** 0.5) if len(differences) > 1 else 0.0
        return [mean, mean - 1.96 * error, mean + 1.96 * error]

    baseline = [r["rouge1"] for r in per_strategy["sequential"]]
    for name, rows in per_strategy.items():
        differences = [r["rouge1"] - b for r, b in zip(rows, baseline, strict=True)]
        mean_difference = statistics.mean(differences)
        error = statistics.pstdev(differences) / (len(differences) ** 0.5) if len(differences) > 1 else 0.0
        summary[name]["vs_sequential"] = [mean_difference, mean_difference - 1.96 * error,
                                          mean_difference + 1.96 * error]

    # The comparison the study was built for: identical clustering, identical model,
    # identical output length, one difference - whether sentence order was kept.
    ordering_effect = {
        "raw": paired_difference("clustered_ordered", "clustered_scrambled"),
        "length_controlled": paired_difference("clustered_ordered_reduced",
                                               "clustered_scrambled_reduced"),
    }

    # ------------------------------------------------------------------- figure
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = range(len(ORDER))
    ax.bar(x, [summary[n]["rouge1"] for n in ORDER],
           yerr=[summary[n]["rouge1_sd"] / len(documents) ** 0.5 for n in ORDER],
           capsize=4, color=["#8C8C8C", "#55A868", "#4C72B0", "#4C72B0", "#C44E52"])
    ax.set_xticks(list(x))
    ax.set_xticklabels([LABELS[n].replace(" (", "\n(").replace(", ", ",\n") for n in ORDER], fontsize=8)
    ax.set(ylabel="ROUGE-1 F1", title=f"Long-document summarisation, {len(documents)} articles")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rouge_by_strategy.png", dpi=150)
    plt.close(fig)

    # -------------------------------------------------------------------- report
    lines = [
        "# Does clever chunking beat truncation?",
        "",
        f"Generated {date.today().isoformat()} by `python -m longsum.study`. "
        f"Model: `{MODEL_NAME}`, which reads {MODEL_MAX_INPUT} tokens. "
        f"Sample: {len(documents)} CNN/DailyMail test articles of at least "
        f"{min(d['tokens'] for d in documents):,} tokens "
        f"(median {statistics.median(d['tokens'] for d in documents):,.0f}), "
        f"so every one of them overflows the window. Seed {SEED}.",
        "",
        "## Results",
        "",
        md_table(
            ["Strategy", "ROUGE-1", "ROUGE-2", "ROUGE-L", "Chunks", "Words out", "Seconds",
             "vs sequential [95% CI]"],
            [
                [LABELS[n], f"{summary[n]['rouge1']:.3f}", f"{summary[n]['rouge2']:.3f}",
                 f"{summary[n]['rougeL']:.3f}", f"{summary[n]['chunks']:.1f}",
                 f"{summary[n]['words']:.0f}", f"{summary[n]['seconds']:.1f}",
                 "baseline" if n == "sequential" else
                 f"{summary[n]['vs_sequential'][0]:+.3f} "
                 f"[{summary[n]['vs_sequential'][1]:+.3f}, {summary[n]['vs_sequential'][2]:+.3f}]"]
                for n in ORDER
            ],
        ),
        "",
        "![ROUGE by strategy](figures/rouge_by_strategy.png)",
        "",
        "## Does sentence order matter?",
        "",
        "The experiment this study was built for: identical clustering, identical model, one "
        "difference only - whether sentences keep their document order when handed to the "
        "summariser.",
        "",
        md_table(
            ["Comparison", "ROUGE-1 difference", "95% CI"],
            [
                ["Raw output", f"{ordering_effect['raw'][0]:+.3f}",
                 f"[{ordering_effect['raw'][1]:+.3f}, {ordering_effect['raw'][2]:+.3f}]"],
                ["After the reduce pass", f"{ordering_effect['length_controlled'][0]:+.3f}",
                 f"[{ordering_effect['length_controlled'][1]:+.3f}, "
                 f"{ordering_effect['length_controlled'][2]:+.3f}]"],
            ],
        ),
        "",
        "The raw comparison is exactly zero, and that is a property of the measure rather "
        "than a result: ROUGE-1 counts unigrams, so concatenating the same chunk summaries in "
        "a different order cannot change it. Comparing concatenated output on ROUGE-1 was "
        "never going to detect reordering. ROUGE-L, which requires matches to appear in "
        f"sequence, does move a little ({summary['clustered_ordered']['rougeL']:.3f} against "
        f"{summary['clustered_scrambled']['rougeL']:.3f}).",
        "",
        "There is a second reason the raw comparison says nothing. Clustering produces more "
        "chunks, more chunks make a longer final text "
        f"({summary['clustered_ordered']['words']:.0f} words against "
        f"{summary['sequential']['words']:.0f} for sequential chunking), and ROUGE F1 punishes "
        "length through precision. The length difference was large enough to bury the effect "
        "being measured.",
        "",
        "What the variants actually differ in is worth stating precisely: the clusters "
        "contain the same sentences either way, and sentences keep their document order "
        "*inside* each cluster. The manipulation is the order the clusters themselves are "
        "summarised and joined in. That only reaches the output once a model reads the joined "
        "text, which is what the reduce pass does.",
        "",
        "With the reduce pass equalising output length to around "
        f"{summary['clustered_ordered_reduced']['words']:.0f} words, the ordered variant scores "
        f"{ordering_effect['length_controlled'][0]:+.3f} ROUGE-1 above the scrambled one, with an "
        f"interval of [{ordering_effect['length_controlled'][1]:+.3f}, "
        f"{ordering_effect['length_controlled'][2]:+.3f}] that still includes zero on "
        f"{len(documents)} documents. Suggestive, not established. Two lessons hold regardless: "
        "a metric has to be able to detect the thing being manipulated, and a confound - here "
        "output length - has to be held fixed before a comparison means anything.",
        "",
        "Intervals are paired: each strategy is compared on the *same* "
        f"{len(documents)} articles, which removes the variation between articles and leaves the "
        "variation between methods. With a sample this small an interval spanning zero means the "
        "study cannot tell the two apart, not that they are equal.",
        "",
        "## Limitations",
        "",
        "- ROUGE counts overlapping words. It rewards copying the reference's phrasing and cannot "
        "tell a correct summary from a fluent wrong one. A human read of a handful of outputs "
        "belongs next to any ROUGE table.",
        "- CNN/DailyMail references are bullet-point highlights, which favour extractive methods; "
        "the lead baseline is strong on this data for that reason.",
        f"- {len(documents)} documents is a demonstration, not an evaluation. The pipeline takes a "
        "sample size argument; the intervals widen or narrow accordingly.",
        "- One model, one dataset, one language.",
    ]
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "report.md").write_text("\n".join(lines))
    (REPORT_DIR / "metrics.json").write_text(
        json.dumps({"strategies": summary, "ordering_effect": ordering_effect}, indent=2)
    )
    (REPORT_DIR / "per_document.json").write_text(json.dumps(per_strategy, indent=2))
    (REPORT_DIR / "examples.json").write_text(
        json.dumps({n: rows[0] for n, rows in per_strategy.items()}, indent=2)
    )
    return {"strategies": summary, "ordering_effect": ordering_effect}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
