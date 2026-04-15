"""PathClaw CLI — onboarding and management."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


PATHCLAW_DIR = Path(os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()
CONFIG_PATH = PATHCLAW_DIR / "config.json"


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {}


def _save_config(config: dict):
    PATHCLAW_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2))


def onboard():
    """First-run onboarding wizard."""
    print()
    print("🔬 Welcome to PathClaw — Computational Pathology Research Platform")
    print("=" * 60)
    print()
    print("PathClaw helps you run MIL/MAMMOTH training pipelines for")
    print("whole-slide image classification.")
    print()

    config = _load_config()

    # Step 0: Research-tool disclaimer
    DISCLAIMER_VERSION = 1
    if (not config.get("disclaimer_acknowledged")
            or int(config.get("disclaimer_version", 0)) < DISCLAIMER_VERSION):
        print("Step 0: Research-Tool Disclaimer")
        print("-" * 40)
        print("PathClaw is a RESEARCH TOOL. Any AI model trained, evaluated,")
        print("or deployed through this platform requires independent clinical")
        print("and statistical validation before use in patient care. You are")
        print("responsible for regulatory compliance in your jurisdiction")
        print("(FDA, CE-IVDR, CDSCO, or equivalent).")
        print()
        ack = input("  Do you understand and accept? [y/N]: ").strip().lower()
        if ack != "y":
            print("  Onboarding aborted.")
            sys.exit(1)
        from datetime import datetime, timezone
        config["disclaimer_acknowledged"] = True
        config["disclaimer_version"] = DISCLAIMER_VERSION
        config["disclaimer_at"] = datetime.now(timezone.utc).isoformat()
        _save_config(config)
        print("  ✅ Acknowledged.")
        print()

    # Step 1: HuggingFace Token
    print("Step 1: HuggingFace Token")
    print("-" * 40)
    print("PathClaw uses foundation model backbones (UNI, CONCH, etc.)")
    print("for feature extraction. These require a HuggingFace account.")
    print()

    existing_hf = config.get("huggingface_token") or os.environ.get("HUGGINGFACE_TOKEN", "")
    if existing_hf:
        print(f"  ✅ HuggingFace token already configured (***{existing_hf[-4:]})")
        update = input("  Update it? [y/N]: ").strip().lower()
        if update == "y":
            existing_hf = ""

    if not existing_hf:
        hf_token = input("  Enter your HuggingFace token (hf_...): ").strip()
        if hf_token:
            config["huggingface_token"] = hf_token
            print("  ✅ Token saved.")
        else:
            print("  ⏭  Skipped. You can set HUGGINGFACE_TOKEN env var later.")
            print("     Some features (backbone download) will not work without it.")

    print()

    # Step 2: GDC Token (optional)
    print("Step 2: GDC Token (optional)")
    print("-" * 40)
    print("For controlled-access TCGA data, you need a GDC token.")
    print("Open-access data (diagnostic slides) works without it.")
    print()

    existing_gdc = config.get("gdc_token", "")
    if existing_gdc:
        print("  ✅ GDC token already configured.")
    else:
        gdc_token = input("  Enter GDC token (or press Enter to skip): ").strip()
        if gdc_token:
            config["gdc_token"] = gdc_token
            print("  ✅ Token saved.")
        else:
            print("  ⏭  Skipped. You can add it later when needed.")

    print()

    # Step 3: Data directory
    print("Step 3: Data Directory")
    print("-" * 40)
    data_dir = config.get("data_dir", str(PATHCLAW_DIR))
    print(f"  Current data directory: {data_dir}")
    new_dir = input("  Change data directory? (Enter to keep): ").strip()
    if new_dir:
        config["data_dir"] = new_dir
        Path(new_dir).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Data directory set to: {new_dir}")

    print()

    # Step 4: GPU check
    print("Step 4: GPU Check")
    print("-" * 40)
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            print(f"  ✅ GPU available: {gpu_name} (×{gpu_count})")
        else:
            print("  ⚠️  No GPU detected. Training will run on CPU (slow).")
    except ImportError:
        print("  ⚠️  PyTorch not installed. Run: pip install -e backend/")

    # Save config
    _save_config(config)

    print()
    print("=" * 60)
    print("✅ PathClaw onboarding complete!")
    print()
    print("Start the backend:")
    print("  cd backend && uvicorn pathclaw.api.app:app --port 8100")
    print()
    print("Open the UI:")
    print("  Open http://localhost:8100 in your browser")
    print()
    print("Or use the agent workspace:")
    print("  openclaw gateway  # Start OpenClaw gateway")
    print()


def status():
    """Show PathClaw system status."""
    config = _load_config()

    print("🔬 PathClaw Status")
    print("=" * 40)
    print(f"  Data dir: {config.get('data_dir', str(PATHCLAW_DIR))}")
    print(f"  HF token: {'✅ configured' if config.get('huggingface_token') else '❌ not set'}")
    print(f"  GDC token: {'✅ configured' if config.get('gdc_token') else '⏭ not set (optional)'}")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"  GPU: ✅ {torch.cuda.get_device_name(0)}")
        else:
            print("  GPU: ❌ not available")
    except ImportError:
        print("  GPU: ❓ PyTorch not installed")


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: pathclaw <command>")
        print("  onboard  — First-run setup")
        print("  status   — System status")
        return

    cmd = sys.argv[1]
    if cmd == "onboard":
        onboard()
    elif cmd == "status":
        status()
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
