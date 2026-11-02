# sglang/ — configs for Ampere+ GPUs (UNTESTED on T4)

**SGLang officially requires sm_80+ (Ampere).** The Kaggle T4 plan (sm_75)
cannot run it, so these configs are provided for future GPUs and are
explicitly **NOT verified**. Do not cite any sglang number from this project
unless it was actually measured on Ampere+ hardware.

Per the build brief (section 1): SGLang is handled as config + documentation
only; the T4 experiments run on vLLM.

## Intended use (when an A10/A100/H100 node is available)

```bash
# Serve the same model + task set; AgentFlight's client is OpenAI-compatible
# and works unchanged against sglang's server.
pip install "sglang[all]"
python -m sglang.launch_server \
  --model Qwen/Qwen3-8B \
  --port 8000 \
  --context-length 32768 \
  --enable-radix-cache          # sglang's prefix-cache analog
  # --disable-radix-cache       # for the "caching off" arm

# E2 replay against sglang (single-node analog of the routing study):
python experiments/e2_replay.py --traces results/ --results results-sglang/ \
  --model Qwen/Qwen3-8B
```

Open questions to verify on Ampere+ before trusting results:
- sglang tool-calling flags for Qwen3 (`--tool-call-parser` equivalent)
- reasoning-content exposure for Qwen3 thinking mode
- radix-cache eviction behavior under our round-robin interleaving (this is
  the interesting comparison vs vLLM prefix caching)
