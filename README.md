# bevy_laya

A Bevy 0.20 adapter for [burn-laya](https://crates.io/crates/burn-laya).
Bevy uses no default features. The adapter needs no window or renderer.

## Interfaces

- `laya` re-exports the dependency's public API.
- `model_dir()` returns the checkpoint directory prepared by `build.rs`, or the runtime `LAYA_MODEL_DIR` override.
- `LayaPlugin::new(agent)` accepts a loaded `laya::Agent`. The caller selects the checkpoint and compute device.
- `LayaRequest { id, request }` carries a caller-selected `u64` identifier and an owned `laya::Request`.
- `LayaResponse { id, result }` returns that identifier and the original `anyhow::Result<laya::Prediction>`.
- `LayaSet` orders producers and consumers within `Update`. Put producers before this set and consumers after it.

## Examples

Run these examples from the repository root.
The build script downloads and converts the pinned checkpoint automatically on the first build.
Release builds optimize the CPU tensor kernels.

| Example | Demonstrates |
| --- | --- |
| [headless](examples/headless.rs) | Read JSON from standard input, submit an owned request, and receive a complete prediction. |
| [typed_questions](examples/typed_questions.rs) | Construct ordered choice, score, and noul questions through the Rust API. |
| [ecs_decision](examples/ecs_decision.rs) | Submit from an ECS system before `LayaSet`, then update entity components from responses after it. |
| [queue_recovery](examples/queue_recovery.rs) | Preserve FIFO order across valid, invalid, and valid requests without stopping after a failure. |

```bash
cargo run --release --locked --example headless <<'JSON'
{
  "state": "An invoice charged us twice for the same purchase.",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which department should handle this?",
      "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages"}
    }
  }
}
JSON

cargo run --release --locked --example typed_questions
cargo run --release --locked --example ecs_decision
cargo run --release --locked --example queue_recovery
```

Each example runs without a window and stops after receiving its expected responses.
The [shared helpers](examples/common/mod.rs) load the CPU agent, poll with a deadline, and write complete predictions as JSON lines.
The ECS example also prints the resulting component values. Its entity-based request identifiers are valid only within the current world.
The queue example reports the expected rejected request on standard error and still completes successfully.
Other loading, inference, timeout, or output failures return an error.
The examples do not establish confidence calibration or gameplay accuracy.

## Execution contract

The plugin drains request messages into a FIFO queue on each update.
Only one request runs at a time on `AsyncComputeTaskPool`.
The plugin adds `TaskPoolPlugin` if the app does not contain it.
On native targets, the frame thread polls completion without waiting for inference.
Each completed request produces one response, including failed requests.
A failed request does not stop later requests.

Requests retain their typed questions, criteria order, and state.
Predictions retain all answer and usage fields.
The adapter does not select actions or interpret confidence values.

The queue has no size limit. Producers must control their request rate.
Responses use Bevy's normal message retention. Consumers must read them promptly.
Dropping the app discards queued work and the task handle.
Dropping the handle cannot interrupt a synchronous model call that already started.

## Checkpoint and validation boundary

`build.rs` downloads and prepares the pinned checkpoint into the user cache directory (`$XDG_CACHE_HOME/bevy_laya/<revision>`, `$HOME/.cache/bevy_laya/<revision>`, `$LOCALAPPDATA/bevy_laya/<revision>`, or `OUT_DIR/laya`).
The build script invokes `uv` to fetch the pinned files, verify the weight digest, and export ONNX, then converts and validates `laya.bpk` with `burn-laya`'s `prepare` feature.
Prepared artifacts stay outside the repository and are reused across builds.
Set `LAYA_MODEL_DIR` at build time or runtime to select another prepared checkpoint directory.
Runtime loading needs `laya.bpk`, `rl_agent_config.json`, and `tokenizer/`.
See the dependency's [architecture document](https://github.com/ParkSnoopy/burn-laya/blob/main/docs/ARCHITECTURE.md) for artifact details and supported devices.
Checkpoint loading can warn about temperature adjustment. Treat affected confidence values as uncalibrated.

The base checkpoint is [convaiinnovations/laya at revision 1c5edc17a7acd8701df6fc341c0d179f1c62c982](https://huggingface.co/convaiinnovations/laya/tree/1c5edc17a7acd8701df6fc341c0d179f1c62c982).
Its `model.safetensors` SHA-256 is `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`.
The build script verifies that digest, exports ONNX with published `laya==0.4.6`, and validates the generated Burn graph against `burn-laya`.

Run `cargo test` from this repository.
The integration test uses the real CPU checkpoint prepared by `build.rs`, not initialized or mock weights.
It compares every serialized prediction field with direct inference on the same agent.
It also checks FIFO order, request identifiers, error propagation, and recovery after failure.
The optimized test profile makes Flex tensor kernels practical.
