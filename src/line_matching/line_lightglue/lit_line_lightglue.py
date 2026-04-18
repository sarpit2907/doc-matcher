import pytorch_lightning as pl
import torch
from torch import optim

from .line_lightglue_model import LineLightGlue


DEFAULT_MATCHER_CONF = {
    "name": "matchers.linelightglue",
    "filter_threshold": 0.1,
    "flash": False,
    "checkpointed": True,
    # Keep the released DocMatcher checkpoint baseline-correct by default.
    "use_graph_transformer": False,
    "graph_k_neighbors": 5,
    "graph_edge_dim": 4,
    "graph_sparse_attention": False,
}


def build_matcher_conf(conf=None):
    merged = dict(DEFAULT_MATCHER_CONF)
    if conf:
        merged.update({key: value for key, value in conf.items() if value is not None})
    return merged


class LitLineLightglue(pl.LightningModule):
    """
    PyTorch Lightning wrapper for the LineLightGlue matching model.

    The matcher config is explicit so notebook workflows can switch between the
    released baseline and a GNN fine-tuning run without editing source files.
    """
    def __init__(self, conf=None, learning_rate: float = 0.0001):
        super().__init__()

        self.model_conf = build_matcher_conf(conf)
        self.save_hyperparameters({"conf": self.model_conf, "learning_rate": learning_rate})

        self.model = LineLightGlue(self.model_conf)
        self.learning_rate = learning_rate

    def load_state_dict(self, state_dict, strict=True):
        """
        Override to use strict=False for Graph Transformer compatibility.

        The pretrained checkpoint contains weights for the original LineSelfBlock
        parameters (Wqkv, out_proj, ffn). The new GraphTransformerBlock has the
        SAME parameters (so they load correctly) PLUS additional graph-specific
        parameters (edge_encoder, graph_gate) that are missing from the checkpoint.

        Using strict=False allows:
        - Original parameters → loaded from checkpoint ✓
        - edge_encoder → default random initialization ✓
        - graph_gate → initialized to 0.0 ✓
        """
        return super().load_state_dict(state_dict, strict=False)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx, mode="train"):
        out = self.model(batch)

        losses, metrics = self.model.loss(out, batch)

        if torch.isnan(losses["total"]).any():
            return None  # NaN loss found. Skipping batch!

        batch_size = batch["descriptors0"].shape[0]
        for loss_name, loss in losses.items():
            self.log(
                f"{mode}/loss/{loss_name}", torch.mean(loss), batch_size=batch_size
            )

        for metric_name, metric in metrics.items():
            self.log(
                f"{mode}/metric/{metric_name}",
                torch.mean(metric),
                batch_size=batch_size,
            )

        return torch.mean(losses["total"])

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            self.training_step(batch, batch_idx, mode="val")

    def configure_optimizers(self):
        params = self.parameters()
        optimizer = optim.Adam(params=params, lr=self.learning_rate)

        scheduler = get_original_lr_scheduler(optimizer)

        return [optimizer], [scheduler]


def get_original_lr_scheduler(optimizer):
    exp_div_10 = 10
    start = 20

    # backward compatibility
    def lr_fn(it):  # noqa: E306
        gam = 10 ** (-1 / exp_div_10)
        return 1.0 if it < start else gam

    return torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_fn)
