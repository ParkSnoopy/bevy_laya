//! Exercise the message adapter with a complete local checkpoint.

use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, ensure};
use bevy::prelude::*;
use bevy_laya::{LayaPlugin, LayaRequest, LayaResponse, LayaSet, laya};
use burn::tensor::Device;
use serde_json::Value;

const REQUEST: &str = r#"{
    "state": "We were billed twice. Please refund the duplicate or we will cancel our plan.",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this?",
            "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages"}
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["not urgent", "soon", "blocking"]
        },
        "churn_risk": {
            "type": "noul",
            "instructions": "Does the user threaten to cancel or leave?"
        }
    }
}"#;

const EMPTY_REQUEST: &str = r#"{"state":"","questions":{}}"#;

#[derive(Default, Resource)]
struct Responses(Vec<(u64, Result<Value, String>)>);

fn collect(mut messages: MessageReader<LayaResponse>, mut responses: ResMut<Responses>) {
    for response in messages.read() {
        let result = response
            .result
            .as_ref()
            .map_err(ToString::to_string)
            .and_then(|prediction| {
                serde_json::to_value(prediction).map_err(|error| error.to_string())
            });
        responses.0.push((response.id, result));
    }
}

#[test]
fn queued_requests_preserve_predictions_and_errors() -> Result<()> {
    let directory = PathBuf::from(
        std::env::var_os("LAYA_MODEL_DIR")
            .context("Set LAYA_MODEL_DIR to a prepared checkpoint directory")?,
    );
    let agent = laya::Agent::load(&directory, &Device::flex())?;
    let request: laya::Request = serde_json::from_str(REQUEST)?;
    let expected = serde_json::to_value(agent.predict(&request)?)?;

    let mut app = App::new();
    app.add_plugins(LayaPlugin::new(agent))
        .init_resource::<Responses>()
        .add_systems(Update, collect.after(LayaSet));
    for (id, input) in [(0, EMPTY_REQUEST), (u64::MAX, REQUEST), (1, EMPTY_REQUEST)] {
        let request = serde_json::from_str(input)?;
        app.world_mut().write_message(LayaRequest { id, request });
    }

    let deadline = Instant::now() + Duration::from_secs(300);
    let mut updates = 0;
    while app.world().resource::<Responses>().0.len() < 3 {
        ensure!(
            Instant::now() < deadline,
            "inference did not finish within 300 seconds"
        );
        app.update();
        updates += 1;
        std::thread::yield_now();
    }
    ensure!(
        updates > 2,
        "queued requests must survive message maintenance"
    );
    let responses = &app.world().resource::<Responses>().0;
    ensure!(
        responses.as_slice()
            == [
                (0, Err("request needs at least one question".to_owned())),
                (u64::MAX, Ok(expected)),
                (1, Err("request needs at least one question".to_owned())),
            ],
        "responses differ from direct inference or request order: {responses:?}"
    );

    app.update();
    ensure!(
        app.world().resource::<Responses>().0.len() == 3,
        "request ran more than once"
    );
    Ok(())
}
