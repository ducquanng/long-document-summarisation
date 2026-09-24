# Does clever chunking beat truncation?

Generated 2026-09-23 by `python -m longsum.study`. Model: `sshleifer/distilbart-cnn-6-6`, which reads 1024 tokens. Sample: 12 CNN/DailyMail test articles of at least 1,222 tokens (median 1,457), so every one of them overflows the window. Seed 42.

## Results

| Strategy | ROUGE-1 | ROUGE-2 | ROUGE-L | Chunks | Words out | Seconds | vs sequential [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Truncate (do nothing) | 0.279 | 0.104 | 0.213 | 1.0 | 44 | 2.6 | +0.028 [+0.009, +0.047] |
| Lead-3 (no model) | 0.222 | 0.063 | 0.168 | 0.0 | 80 | 0.0 | -0.029 [-0.081, +0.023] |
| Sequential chunks | 0.251 | 0.083 | 0.177 | 2.0 | 91 | 4.8 | baseline |
| Clustered, order kept | 0.158 | 0.045 | 0.107 | 5.2 | 227 | 10.5 | -0.094 [-0.137, -0.050] |
| Clustered, order lost | 0.158 | 0.045 | 0.108 | 5.2 | 227 | 10.2 | -0.094 [-0.137, -0.050] |
| Sequential + reduce | 0.283 | 0.099 | 0.206 | 2.0 | 46 | 6.5 | +0.032 [+0.002, +0.061] |
| Clustered ordered + reduce | 0.217 | 0.041 | 0.147 | 5.2 | 51 | 12.5 | -0.034 [-0.075, +0.007] |
| Clustered scrambled + reduce | 0.182 | 0.037 | 0.140 | 5.2 | 50 | 12.3 | -0.069 [-0.111, -0.027] |

![ROUGE by strategy](figures/rouge_by_strategy.png)

## Does sentence order matter?

The experiment this study was built for: identical clustering, identical model, one difference only - whether sentences keep their document order when handed to the summariser.

| Comparison | ROUGE-1 difference | 95% CI |
| --- | --- | --- |
| Raw output | +0.000 | [+0.000, +0.000] |
| After the reduce pass | +0.035 | [-0.012, +0.082] |

The raw comparison is exactly zero, and that is a property of the measure rather than a result: ROUGE-1 counts unigrams, so concatenating the same chunk summaries in a different order cannot change it. Comparing concatenated output on ROUGE-1 was never going to detect reordering. ROUGE-L, which requires matches to appear in sequence, does move a little (0.107 against 0.108).

There is a second reason the raw comparison says nothing. Clustering produces more chunks, more chunks make a longer final text (227 words against 91 for sequential chunking), and ROUGE F1 punishes length through precision. The length difference was large enough to bury the effect being measured.

What the variants actually differ in is worth stating precisely: the clusters contain the same sentences either way, and sentences keep their document order *inside* each cluster. The manipulation is the order the clusters themselves are summarised and joined in. That only reaches the output once a model reads the joined text, which is what the reduce pass does.

With the reduce pass equalising output length to around 51 words, the ordered variant scores +0.035 ROUGE-1 above the scrambled one, with an interval of [-0.012, +0.082] that still includes zero on 12 documents. Suggestive, not established. Two lessons hold regardless: a metric has to be able to detect the thing being manipulated, and a confound - here output length - has to be held fixed before a comparison means anything.

Intervals are paired: each strategy is compared on the *same* 12 articles, which removes the variation between articles and leaves the variation between methods. With a sample this small an interval spanning zero means the study cannot tell the two apart, not that they are equal.

## Limitations

- ROUGE counts overlapping words. It rewards copying the reference's phrasing and cannot tell a correct summary from a fluent wrong one. A human read of a handful of outputs belongs next to any ROUGE table.
- CNN/DailyMail references are bullet-point highlights, which favour extractive methods; the lead baseline is strong on this data for that reason.
- 12 documents is a demonstration, not an evaluation. The pipeline takes a sample size argument; the intervals widen or narrow accordingly.
- One model, one dataset, one language.