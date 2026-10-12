# bevy_laya

A Bevy 0.19 adapter for the Git dependency [laya-rs](https://github.com/ParkSnoopy/laya-rs).
Bevy uses no default features. The adapter needs no window or renderer.

## Interfaces

- `laya` re-exports the dependency's public API.
- `LayaPlugin::new(agent)` accepts a loaded `laya::Agent`. The caller selects the checkpoint and compute device.
- `LayaRequest { id, request }` carries a caller-selected `u64` identifier and an owned `laya::Request`.
- `LayaResponse { id, result }` returns that identifier and the original `anyhow::Result<laya::Prediction>`.
- `LayaSet` orders producers and consumers within `Update`. Put producers before this set and consumers after it.

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

The dependency generates Burn code from ONNX during its Cargo build.
`LAYA_MODEL_DIR` must identify a prepared checkpoint directory through an absolute path.
The directory needs `laya.onnx` and its external tensor data for compilation.
Runtime loading also needs `laya.bpk`, configuration, and tokenizer files.
See the dependency's [architecture document](https://github.com/ParkSnoopy/laya-rs/blob/main/docs/ARCHITECTURE.md) for artifact preparation and supported devices.
Checkpoint loading can warn about temperature adjustment. Treat affected confidence values as uncalibrated.

Run `cargo test` with `LAYA_MODEL_DIR` set to that directory.
The integration test uses the real CPU checkpoint, not initialized or mock weights.
It compares every serialized prediction field with direct inference on the same agent.
It also checks FIFO order, request identifiers, error propagation, and recovery after failure.
The test fails if required artifacts are absent.
The optimized test profile makes Flex tensor kernels practical.
