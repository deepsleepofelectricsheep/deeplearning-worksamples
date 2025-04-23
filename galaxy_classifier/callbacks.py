import pytorch_lightning as pl

KEY = "optimizer"


class LearningRateMonitor(pl.callbacks.LearningRateMonitor):

    def _add_prefix(self, *args, **kwargs) -> str:
        return f"{KEY}/" + super()._add_prefix(*args, **kwargs)