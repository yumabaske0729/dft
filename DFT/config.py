
# gaussian/config.py
from __future__ import annotations
from pathlib import Path
import os

ENV_OUTDIR = "DFT_OUTPUT_DIR"

def resolve_output_root(cli_outdir: str | None) -> Path:
    """
    出力ルートディレクトリを決定する。
    優先順位: CLI --outdir > 環境変数 DFT_OUTPUT_DIR > ./output
    """
    if cli_outdir and cli_outdir.strip():
        root = Path(cli_outdir).expanduser()
    else:
        env = os.getenv(ENV_OUTDIR)
        root = Path(env).expanduser() if env else (Path.cwd() / "output")
    root.mkdir(parents=True, exist_ok=True)
    return root
