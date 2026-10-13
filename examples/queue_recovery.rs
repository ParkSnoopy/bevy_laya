//! Queue valid, invalid, and valid requests and handle each result in FIFO order.

mod common;

use std::io::Write;

use anyhow::{Result, bail, ensure};
use bevy::prelude::*;
use bevy_laya::{LayaPlugin, LayaRequest};

const VALID: &str = r#"{
    "state": "An invoice charged us twice for the same purchase.",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this?",
            "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages"}
        }
    }
}"#;

fn main() -> Result<()> {
    let mut app = App::new();
    app.add_plugins(LayaPlugin::new(common::load_agent()?));
    for (id, input) in [
        (10, VALID),
        (11, r#"{"state":"","questions":{}}"#),
        (12, VALID),
    ] {
        app.world_mut().write_message(LayaRequest {
            id,
            request: serde_json::from_str(input)?,
        });
    }

    let responses = common::run(&mut app, 3)?;
    ensure!(
        responses
            .iter()
            .map(|response| response.id)
            .eq([10, 11, 12]),
        "responses arrived out of order"
    );
    for response in responses {
        if response.id == 11 {
            let Err(error) = response.result else {
                bail!("invalid request was accepted");
            };
            writeln!(
                std::io::stderr().lock(),
                "request {} rejected: {error}",
                response.id
            )?;
        } else {
            common::write_prediction(response.id, &response.result?)?;
        }
    }
    Ok(())
}
