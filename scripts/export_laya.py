"""Download the pinned Laya checkpoint and export its ONNX graph."""

import argparse
import hashlib
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from laya.agent import Agent
from laya.common import collate_items
from onnxscript import BOOL, FLOAT, opset23, optimizer

REPO = "convaiinnovations/laya"
FILES = (
    "model.safetensors",
    "rl_agent_config.json",
    "encoder/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
)
INPUTS = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
SAMPLE_STATE = (
    "We were billed twice. Please refund the duplicate or we will cancel our plan."
)
SAMPLE_QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["not urgent", "soon", "blocking"],
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?",
    },
}


def sha256_file(path: Path) -> str:
    """Return the lowercase hexadecimal SHA-256 digest of a file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export_attention(
    query: FLOAT,
    key: FLOAT,
    value: FLOAT,
    attn_mask: FLOAT | BOOL | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> FLOAT:
    """Translate PyTorch scaled dot-product attention into opset-23 Attention."""
    if dropout_p or enable_gqa:
        msg = "export requires inference-only multi-head attention"
        raise ValueError(msg)
    if attn_mask is not None:
        # ORT's Attention kernel requires the query axis explicitly,
        # even for a broadcast padding mask.
        mask_shape = opset23.Concat(
            opset23.Shape(query, start=0, end=1),
            opset23.Shape(attn_mask, start=1, end=2),
            opset23.Shape(query, start=2, end=3),
            opset23.Shape(key, start=2, end=3),
            axis=0,
        )
        attn_mask = opset23.Expand(attn_mask, mask_shape)
    return opset23.Attention(
        query,
        key,
        value,
        attn_mask,
        is_causal=int(is_causal),
        scale=scale,
    )[0]


def download_checkpoint(directory: Path, revision: str, expected_sha256: str) -> None:
    """Download pinned Hugging Face checkpoint files and verify the weight digest."""
    for name in FILES:
        hf_hub_download(
            repo_id=REPO,
            revision=revision,
            filename=name,
            local_dir=str(directory),
        )
    actual = sha256_file(directory / "model.safetensors")
    if actual != expected_sha256:
        msg = f"checkpoint digest mismatch: expected {expected_sha256}, got {actual}"
        raise ValueError(msg)


def export_onnx(directory: Path) -> None:
    """Export the local checkpoint to opset-23 ONNX with external tensor data."""
    agent = Agent(str(directory), device="cpu", compile=False)
    agent.model.eval().float()
    capacity = agent.model.encoder.config.max_position_embeddings
    if agent.amp_enabled:
        msg = "ONNX export requires FP32 without autocast"
        raise ValueError(msg)

    internal = {qid: agent._to_internal(defn) for qid, defn in SAMPLE_QUESTIONS.items()}
    encoded_rows = agent._encode_state(
        SAMPLE_STATE,
        list(SAMPLE_QUESTIONS),
        internal,
        max_len=capacity,
    )
    batch = collate_items([encoded_rows], agent.tok.pad_token_id)

    attention = agent.model.encoder.config._attn_implementation
    fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        agent.model.encoder.config._attn_implementation = "eager"
        torch.backends.mha.set_fastpath_enabled(False)
        batch_dim = torch.export.Dim("batch", min=1)
        sequence_dim = torch.export.Dim("sequence", min=1, max=capacity)
        marker_dim = torch.export.Dim("markers", min=2)
        sdpa = torch.ops.aten.scaled_dot_product_attention.default
        with torch.inference_mode():
            program = torch.onnx.export(
                agent.model,
                tuple(batch[key] for key in INPUTS),
                None,
                input_names=list(INPUTS),
                output_names=["logits", "act_logits"],
                dynamic_shapes=(
                    {0: batch_dim, 1: sequence_dim},
                    {0: batch_dim, 1: sequence_dim},
                    {0: batch_dim, 1: marker_dim},
                    {0: batch_dim, 1: marker_dim},
                    {0: batch_dim},
                ),
                opset_version=23,
                custom_translation_table={sdpa: export_attention},
                dynamo=True,
                external_data=True,
                optimize=False,
            )
            # Constant-fold without the opset-23 RotaryEmbedding rewrite,
            # which loses ModernBERT's broadcast cache semantics.
            program.model = optimizer.optimize(program.model)
            program.save(str(directory / "laya.onnx"), external_data=True)
    finally:
        agent.model.encoder.config._attn_implementation = attention
        torch.backends.mha.set_fastpath_enabled(fastpath)


def main(argv: list[str] | None = None) -> None:
    """Download the pinned checkpoint and export its ONNX graph."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision", help="Pinned Hugging Face commit revision")
    parser.add_argument("expected_sha256", help="Expected SHA-256 of model.safetensors")
    parser.add_argument("directory", type=Path, help="Output staging directory")
    args = parser.parse_args(argv)
    download_checkpoint(args.directory, args.revision, args.expected_sha256)
    export_onnx(args.directory)


if __name__ == "__main__":
    main(sys.argv[1:])
