//! Download and prepare the pinned Laya checkpoint during the build.

use std::{
    env,
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use anyhow::{Context, Result, bail, ensure};

const REVISION: &str = "1c5edc17a7acd8701df6fc341c0d179f1c62c982";
const WEIGHT_SHA256: &str = "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c";
const READY_MARKER: &str = ".bevy_laya_ready";
const PYTORCH_CPU_INDEX: &str = "https://download.pytorch.org/whl/cpu";

const DEPLOYMENT_FILES: [&str; 4] = [
    "laya.bpk",
    "rl_agent_config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
];

const EXPORT_SCRIPT: &str = include_str!("scripts/export_laya.py");

struct StagingGuard {
    path: PathBuf,
    active: bool,
}

impl StagingGuard {
    const fn new(path: PathBuf) -> Self {
        Self { path, active: true }
    }

    const fn disarm(&mut self) {
        self.active = false;
    }
}

impl Drop for StagingGuard {
    fn drop(&mut self) {
        if self.active {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

fn emit_directives(model_dir: &Path) -> Result<()> {
    let model_dir = model_dir
        .to_str()
        .context("prepared model directory path must be valid UTF-8")?;
    let mut stdout = std::io::stdout().lock();
    writeln!(stdout, "cargo::rerun-if-changed=build.rs")?;
    writeln!(stdout, "cargo::rerun-if-changed=scripts/export_laya.py")?;
    writeln!(stdout, "cargo::rerun-if-env-changed=LAYA_MODEL_DIR")?;
    writeln!(stdout, "cargo::rerun-if-env-changed=XDG_CACHE_HOME")?;
    writeln!(stdout, "cargo::rerun-if-env-changed=LOCALAPPDATA")?;
    writeln!(stdout, "cargo::rerun-if-env-changed=DOCS_RS")?;
    writeln!(stdout, "cargo::rustc-env=LAYA_MODEL_DIR={model_dir}")?;
    Ok(())
}

fn non_empty_env(name: &str) -> Option<PathBuf> {
    env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn resolve_cache_dir() -> Result<PathBuf> {
    let cache_root = non_empty_env("XDG_CACHE_HOME")
        .or_else(|| non_empty_env("HOME").map(|home| home.join(".cache")))
        .or_else(|| non_empty_env("LOCALAPPDATA"))
        .map(|root| root.join("bevy_laya").join(REVISION));
    match cache_root {
        Some(path) => Ok(path),
        None => Ok(PathBuf::from(env::var_os("OUT_DIR").context("OUT_DIR is unset")?).join("laya")),
    }
}

fn has_deployment_files(dir: &Path) -> bool {
    DEPLOYMENT_FILES
        .iter()
        .all(|relative| dir.join(relative).is_file())
}

fn expected_marker() -> String {
    format!("{REVISION}:{WEIGHT_SHA256}\n")
}

fn is_cache_ready(dir: &Path) -> bool {
    has_deployment_files(dir)
        && fs::read_to_string(dir.join(READY_MARKER))
            .is_ok_and(|content| content == expected_marker())
}

fn export_onnx(staging_dir: &Path) -> Result<()> {
    let mut child = Command::new("uv")
        .args([
            "run",
            "--no-project",
            "--python",
            "3.13",
            "--index",
            PYTORCH_CPU_INDEX,
            "--index-strategy",
            "unsafe-best-match",
            "--with",
            "torch",
            "--with",
            "laya[onnx]==0.4.6",
            "python",
            "-",
            REVISION,
            WEIGHT_SHA256,
        ])
        .arg(staging_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to execute `uv`; install uv to download and export Laya")?;

    let mut stdin = child
        .stdin
        .take()
        .context("failed to open stdin for uv export process")?;
    stdin.write_all(EXPORT_SCRIPT.as_bytes())?;
    drop(stdin);

    let output = child.wait_with_output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("uv checkpoint export failed ({}):\n{stderr}", output.status);
    }
    Ok(())
}

fn prune_intermediates(dir: &Path) -> Result<()> {
    for file in [
        "model.safetensors",
        "laya.onnx",
        "laya.onnx.data",
        "laya.rs",
    ] {
        let path = dir.join(file);
        if path.is_file() {
            fs::remove_file(path)?;
        }
    }
    for subdir in ["encoder", ".cache"] {
        let path = dir.join(subdir);
        if path.is_dir() {
            fs::remove_dir_all(path)?;
        }
    }
    Ok(())
}

fn publish_staging(staging: &mut StagingGuard, target_dir: &Path) -> Result<()> {
    if target_dir.exists() {
        fs::remove_dir_all(target_dir)?;
    }
    fs::rename(&staging.path, target_dir)?;
    staging.disarm();
    Ok(())
}

fn prepare_checkpoint(target_dir: &Path) -> Result<()> {
    let parent = target_dir
        .parent()
        .context("model directory must have a parent directory")?;
    fs::create_dir_all(parent)?;

    let lock_path = parent.join(format!(".bevy_laya-{REVISION}.lock"));
    let lock_file: File = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&lock_path)?;
    lock_file.lock()?;

    if is_cache_ready(target_dir) {
        return Ok(());
    }

    let staging_path = parent.join(format!(".staging-{REVISION}-{}", std::process::id()));
    if staging_path.exists() {
        fs::remove_dir_all(&staging_path)?;
    }
    fs::create_dir_all(&staging_path)?;
    let mut staging = StagingGuard::new(staging_path);

    export_onnx(&staging.path)?;
    burn_laya::inference::prepare::generate(&staging.path)?;
    prune_intermediates(&staging.path)?;
    ensure!(
        has_deployment_files(&staging.path),
        "prepared checkpoint is missing required deployment files"
    );
    fs::write(staging.path.join(READY_MARKER), expected_marker())?;
    publish_staging(&mut staging, target_dir)?;
    Ok(())
}

fn main() -> Result<()> {
    if env::var_os("DOCS_RS").is_some() {
        return emit_directives(Path::new(""));
    }
    if let Some(override_dir) = non_empty_env("LAYA_MODEL_DIR") {
        let override_dir = std::path::absolute(override_dir)?;
        ensure!(
            has_deployment_files(&override_dir),
            "LAYA_MODEL_DIR ({}) is missing required deployment files",
            override_dir.display()
        );
        return emit_directives(&override_dir);
    }
    let target_dir = resolve_cache_dir()?;
    if !is_cache_ready(&target_dir) {
        prepare_checkpoint(&target_dir)?;
    }
    emit_directives(&target_dir)
}
