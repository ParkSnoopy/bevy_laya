//! Construct choice, score, and noul questions directly through the Rust API.

mod common;

use anyhow::Result;
use bevy::prelude::*;
use bevy_laya::{LayaPlugin, LayaRequest, laya};

fn main() -> Result<()> {
    let request = laya::Request {
        state: "We were billed twice. Please refund the duplicate or we will cancel our plan."
            .to_owned(),
        questions: [
            (
                "department".to_owned(),
                laya::Question::Choice {
                    instructions: "Which department should handle this?".to_owned(),
                    criteria: [
                        (
                            "billing".to_owned(),
                            Some("invoices, payments, refunds".to_owned()),
                        ),
                        ("technical".to_owned(), Some("bugs, outages".to_owned())),
                    ]
                    .into_iter()
                    .collect(),
                },
            ),
            (
                "urgency".to_owned(),
                laya::Question::Score {
                    instructions: "How urgent is this?".to_owned(),
                    criteria: ["not urgent", "soon", "blocking"]
                        .into_iter()
                        .map(str::to_owned)
                        .collect(),
                },
            ),
            (
                "churn_risk".to_owned(),
                laya::Question::Noul {
                    instructions: "Does the user threaten to cancel or leave?".to_owned(),
                    criteria: std::iter::empty().collect(),
                },
            ),
        ]
        .into_iter()
        .collect(),
    };
    let mut app = App::new();
    app.add_plugins(LayaPlugin::new(common::load_agent()?));
    app.world_mut()
        .write_message(LayaRequest { id: 42, request });

    for response in common::run(&mut app, 1)? {
        common::write_prediction(response.id, &response.result?)?;
    }
    Ok(())
}
