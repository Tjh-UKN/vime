# Qwen3.8-27B: historical 100-round Ascend GRPO evidence

This bundle supports [Vime PR #512](https://github.com/vllm-project/vime/pull/512).
It contains the saved 2026-10-30/31 integration run, rather than a new run of the PR head.
The PR additionally fixes shared input-layernorm TP-gradient synchronization.
The saved run used vLLM 0.30.0 / Ascend 0.30.0rc1 and an unpacked HCCL compatibility
adapter. Current-head NPU coverage is provided separately by PR CI.

## Files

- `training-dashboard.png` / `.svg`: twelve native training/runtime metric panels,
  following the presentation in PR #403; raw values, trailing 5-step means, and full-run means.
- `train.metrics.jsonl`: all 100 rounds' original scalar dictionaries plus native Timer
  values and original log line numbers. Console prefixes, prompts and machine paths are omitted.
- `train.metrics.csv`: the same numeric data in tabular form.
- `summary.json`: configuration, environment, completion evidence, timing definitions and interpretation.
- `SHA256SUMS`: bundle-file checksums.

Source `train.log`: 18,183,197 bytes, 132,386 lines.
SHA256: `0b150d6ea1f13f61d2c4644b435b1575bdb0b8769a6c2b5c215d7be300754a87`. Its bytes were checked against the immutable original evidence archive.
Each exported round was checked against its native rollout/train dictionaries and Timer,
and against the previously verified per-round CSV. All scalar values are finite.

## Reading the curves

The 100 rounds use train steps 0–99, 32 prompts x 8 samples per round, a response
limit of 4096, TP4/PP2/SP/CP1 GDN head TP, V2 and no MTP. Sixteen logical Ascend910
devices are split 8 actor / 8 rollout, on eight physical dual-chip cards.
The Ray job succeeded with driver exit 0; there are 101 weight syncs including
initialization, and 202 HTTP 200 responses each for update/finish/resume across two engines.

DAPO scores +1 for correct / -1 for incorrect; `(raw_reward + 1) / 2` is the
training-batch judged-correct fraction. Compare windows rather than single batches.
First/last ten-round raw reward: -0.154688 / 0.642188.
Mean response length: 3124.65 / 1675.95.
Training batches change, so these curves do not establish held-out generalization or convergence.

The reference-model KL penalty was disabled: `train/kl_loss` is absent. The chart
shows the actually logged `train/ppo_kl` instead. Near-zero group-normalized PG loss
should be interpreted together with nonzero gradient norms and the optimizer steps.

The plotted `perf/step_time` is the native metric (mean 990.37s).
Continuous end-to-end step wall time is 990.16s on average,
measured between successive post-training synchronization ends and including checkpoint/phase overhead.
The two series have different recording boundaries and should be reported separately.
`perf/update_weights_time` is retained verbatim but can refer to the preceding sync;
the post-training sync plot instead uses the correct native Timer (mean 3.609s, last 3.6s).
Responses became shorter; changes in timing alone do not isolate a kernel speedup.
The last few rounds also overlapped environment-backup CPU/disk activity, without an isolation control.
The native generic FLOPs field is not used to claim hybrid-GDN MFU.
