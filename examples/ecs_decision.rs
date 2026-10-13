//! Route asynchronous movement decisions to their originating entities.

mod common;

use std::io::Write;

use anyhow::{Result, bail};
use bevy::prelude::*;
use bevy_laya::{LayaPlugin, LayaRequest, LayaResponse, LayaSet, laya};

#[derive(Debug, Clone, Copy, Component)]
struct Situation(&'static str);

#[derive(Debug, Component)]
enum Decision {
    Waiting,
    Pending,
    Chosen(String),
    Failed(String),
}

fn submit(
    mut requests: MessageWriter<'_, LayaRequest>,
    mut actors: Query<'_, '_, (Entity, &Situation, &mut Decision)>,
) {
    for (entity, situation, mut decision) in &mut actors {
        if !matches!(*decision, Decision::Waiting) {
            continue;
        }
        requests.write(LayaRequest {
            // Entity bits include the generation. These IDs are runtime-only.
            id: entity.to_bits(),
            request: laya::Request {
                state: situation.0.to_owned(),
                questions: std::iter::once((
                    "movement".to_owned(),
                    laya::Question::Choice {
                        instructions: "Which movement is safest for this unit?".to_owned(),
                        criteria: [
                            (
                                "advance".to_owned(),
                                Some("move toward the enemy".to_owned()),
                            ),
                            ("hold".to_owned(), Some("remain in cover".to_owned())),
                            (
                                "retreat".to_owned(),
                                Some("move away from the enemy".to_owned()),
                            ),
                        ]
                        .into_iter()
                        .collect(),
                    },
                ))
                .collect(),
            },
        });
        *decision = Decision::Pending;
    }
}

fn apply(
    mut responses: MessageReader<'_, '_, LayaResponse>,
    mut decisions: Query<'_, '_, &mut Decision>,
) {
    for response in responses.read() {
        let Some(entity) = Entity::try_from_bits(response.id) else {
            continue;
        };
        // Ignore a response if the originating entity was removed.
        let Ok(mut decision) = decisions.get_mut(entity) else {
            continue;
        };
        *decision = match &response.result {
            Ok(prediction) => {
                match prediction
                    .answers
                    .get("movement")
                    .map(|answer| &answer.value)
                {
                    Some(laya::AnswerValue::Choice { choice, .. }) => {
                        Decision::Chosen(choice.clone())
                    }
                    _ => Decision::Failed("missing movement choice".to_owned()),
                }
            }
            Err(error) => Decision::Failed(error.to_string()),
        };
    }
}

fn main() -> Result<()> {
    let mut app = App::new();
    app.add_plugins(LayaPlugin::new(common::load_agent()?))
        .add_systems(Update, submit.before(LayaSet))
        .add_systems(Update, apply.after(LayaSet));
    for state in [
        "Our unit has low health, no ammunition, and an escape route. The enemy is approaching.",
        "Our healthy unit has ammunition and good cover. Reinforcements will arrive soon.",
    ] {
        app.world_mut().spawn((Situation(state), Decision::Waiting));
    }

    for response in common::run(&mut app, 2)? {
        common::write_prediction(response.id, &response.result?)?;
    }
    let mut actors = app.world_mut().query::<(&Situation, &Decision)>();
    let mut output = std::io::stdout().lock();
    for (situation, decision) in actors.iter(app.world()) {
        match decision {
            Decision::Chosen(choice) => writeln!(output, "{} -> {choice}", situation.0)?,
            Decision::Failed(error) => bail!("movement decision failed: {error}"),
            Decision::Waiting | Decision::Pending => bail!("movement decision is incomplete"),
        }
    }
    Ok(())
}
