//! Submit a JSON request from standard input to a manually updated headless app.

mod common;

use anyhow::Result;
use bevy::prelude::*;
use bevy_laya::{LayaPlugin, LayaRequest, laya};

fn main() -> Result<()> {
    let request: laya::Request = serde_json::from_reader(std::io::stdin().lock())?;
    let mut app = App::new();
    app.add_plugins(LayaPlugin::new(common::load_agent()?));
    app.world_mut()
        .write_message(LayaRequest { id: 0, request });

    for response in common::run(&mut app, 1)? {
        common::write_prediction(response.id, &response.result?)?;
    }
    Ok(())
}
