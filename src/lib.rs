//! Asynchronous Laya inference for Bevy 0.19.

use std::{collections::VecDeque, sync::Arc};

use anyhow::Result;
use bevy::{
    app::TaskPoolPlugin,
    prelude::*,
    tasks::{AsyncComputeTaskPool, Task, futures::check_ready},
};

pub use laya_rs as laya;

/// Runs requests sequentially on Bevy's asynchronous compute pool.
///
/// Load the agent before adding the plugin. The caller selects its checkpoint
/// and device through Laya's API.
#[derive(Debug)]
pub struct LayaPlugin {
    agent: Arc<laya::Agent>,
}

impl LayaPlugin {
    /// Use a loaded agent without copying its model weights.
    pub fn new(agent: laya::Agent) -> Self {
        Self {
            agent: Arc::new(agent),
        }
    }
}

impl Plugin for LayaPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<TaskPoolPlugin>() {
            app.add_plugins(TaskPoolPlugin::default());
        }
        app.add_message::<LayaRequest>()
            .add_message::<LayaResponse>()
            .insert_resource(Worker {
                agent: Arc::clone(&self.agent),
                pending: VecDeque::new(),
                task: None,
            })
            .add_systems(Update, infer.in_set(LayaSet));
    }
}

/// Order request producers before this set and response consumers after it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, SystemSet)]
pub struct LayaSet;

/// An owned request. The plugin consumes this message and retains queued work.
#[derive(Debug, Message)]
pub struct LayaRequest {
    /// Caller-selected identifier, copied unchanged to the response.
    pub id: u64,
    /// State and typed questions passed unchanged to Laya.
    pub request: laya::Request,
}

/// A completed request, including any validation or inference error.
#[derive(Debug, Message)]
pub struct LayaResponse {
    /// Identifier from the corresponding request.
    pub id: u64,
    /// Original Laya prediction or error.
    pub result: Result<laya::Prediction>,
}

#[derive(Resource)]
struct Worker {
    agent: Arc<laya::Agent>,
    pending: VecDeque<LayaRequest>,
    task: Option<Task<LayaResponse>>,
}

fn infer(
    mut requests: ResMut<Messages<LayaRequest>>,
    mut worker: ResMut<Worker>,
    mut responses: MessageWriter<LayaResponse>,
) {
    worker.pending.extend(requests.drain());
    if let Some(response) = worker.task.as_mut().and_then(check_ready) {
        worker.task = None;
        responses.write(response);
    }
    if worker.task.is_none()
        && let Some(request) = worker.pending.pop_front()
    {
        let agent = Arc::clone(&worker.agent);
        worker.task = Some(AsyncComputeTaskPool::get().spawn(async move {
            LayaResponse {
                id: request.id,
                result: agent.predict(&request.request),
            }
        }));
    }
}
