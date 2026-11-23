//! Checkpoint loading, bounded headless polling, and JSON output for the examples.

use std::{
    io::Write,
    time::{Duration, Instant},
};

use anyhow::{Result, ensure};
use bevy::prelude::*;
use bevy_laya::{LayaResponse, laya};
use burn::tensor::Device;

/// Load the build-prepared checkpoint or the runtime `LAYA_MODEL_DIR` override.
///
/// # Errors
/// Returns an error for a missing directory or incompatible checkpoint assets.
pub fn load_agent() -> Result<laya::Agent> {
    laya::Agent::load(&bevy_laya::model_dir(), &Device::flex())
}

/// Update a headless app until the requested responses arrive.
///
/// Response consumers run inside `Update` before this function drains messages.
/// Inference errors remain in each response. A timeout returns an error.
///
/// # Errors
/// Returns an error if the requested responses do not arrive within 300 seconds.
pub fn run(app: &mut App, count: usize) -> Result<Vec<LayaResponse>> {
    let deadline = Instant::now() + Duration::from_secs(300);
    let mut responses = Vec::with_capacity(count);
    while responses.len() < count {
        ensure!(
            Instant::now() < deadline,
            "inference timed out after 300 seconds"
        );
        app.update();
        responses.extend(
            app.world_mut()
                .resource_mut::<Messages<LayaResponse>>()
                .drain(),
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    Ok(responses)
}

/// Write the request identifier and complete prediction as one JSON line.
///
/// # Errors
/// Returns an error if serialization or standard output fails.
pub fn write_prediction(id: u64, prediction: &laya::Prediction) -> Result<()> {
    let mut output = std::io::stdout().lock();
    serde_json::to_writer(
        &mut output,
        &serde_json::json!({"id": id, "prediction": prediction}),
    )?;
    writeln!(output)?;
    Ok(())
}
