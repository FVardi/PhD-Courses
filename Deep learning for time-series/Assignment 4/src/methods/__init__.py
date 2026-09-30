"""Adapters around the external checkouts, plus the one helper both drivers need.

Nothing here alters a method's learning objective; see src/methods/tloss.py and
src/methods/ts2vec.py for what each adapter does and why.
"""

from pathlib import Path

from src.methods import tloss, ts2vec

METHODS = ("tloss", "ts2vec")


def checkpoint_path(ckpt_dir: Path, method: str, dataset: str, seed: int) -> Path:
    """The file each pretraining script writes. T-Loss appends its architecture name."""
    if method == "tloss":
        return Path(f"{ckpt_dir / f'tloss_{dataset}_seed{seed}'}"
                    f"_{tloss.ARCHITECTURE}_encoder.pth")
    return ckpt_dir / f"ts2vec_{dataset}_seed{seed}.pth"


def load_pretrained(method: str, cfg: dict, ckpt_dir: Path, dataset: str, seed: int,
                    channels: int, cuda: bool = True):
    """Rebuild the architecture and restore trained weights. Returns None if absent.

    Shared by dev/5_ and dev/7_ so both load encoders identically - a divergence here would
    silently mean Part D evaluated a differently-constructed model than Part C.
    """
    if not checkpoint_path(ckpt_dir, method, dataset, seed).exists():
        return None
    if method == "tloss":
        model = tloss.build(cfg, in_channels=channels, cuda=cuda)
        return tloss.load_encoder(model, ckpt_dir / f"tloss_{dataset}_seed{seed}")
    model = ts2vec.build(cfg, input_dims=channels, device="cuda" if cuda else "cpu")
    return ts2vec.load_encoder(model, ckpt_dir / f"ts2vec_{dataset}_seed{seed}.pth")


def encode(method: str, model, X):
    """Frozen forward pass, shape (n_series, 320)."""
    return tloss.encode(model, X) if method == "tloss" else ts2vec.encode(model, X)
