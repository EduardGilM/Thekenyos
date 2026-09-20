"""Small, optional training metrics logger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


class TrainingLog:
    """Append metrics locally and optionally mirror them to Weights & Biases."""

    def __init__(
        self,
        output: str | Path,
        config: Mapping[str, Any] | None = None,
        *,
        wandb_mode: str = 'disabled',
        wandb_project: str = 'Thekenyos',
        wandb_entity: str | None = None,
        wandb_name: str | None = None,
        upload_checkpoints: bool = False,
        wandb_run_id: str | None = None,
        wandb_module: Any | None = None,
    ) -> None:
        if wandb_mode not in {'online', 'offline', 'disabled'}:
            raise ValueError("wandb_mode must be 'online', 'offline', or 'disabled'")
        output = Path(output)
        self.path = output if output.suffix == '.jsonl' else output / 'training.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.output = self.path.parent
        self.wandb_mode = wandb_mode
        self.upload_checkpoints = upload_checkpoints
        self._wandb = None
        self._run = None
        self._closed = False
        self.config = dict(config or {})
        for key in ('approximations', 'exptseed', 'worlds', 'rates'):
            self.config.setdefault(key, None)
        if wandb_mode != 'disabled':
            try:
                module = wandb_module if wandb_module is not None else __import__('wandb')
            except ImportError as exc:
                raise RuntimeError(
                    f"W&B mode '{wandb_mode}' requires the optional 'wandb' package; "
                    "install it or use --wandb-mode disabled"
                ) from exc
            self._wandb = module
            try:
                kwargs = dict(project=wandb_project, config=self.config, mode=wandb_mode, dir=str(self.output.resolve()))
                if wandb_entity is not None:
                    kwargs['entity'] = wandb_entity
                if wandb_name is not None:
                    kwargs['name'] = wandb_name
                settings = getattr(module, 'Settings', None)
                if settings is not None:
                    kwargs['settings'] = settings(disable_code=True, disable_git=True, console='off')
                if wandb_run_id is not None:
                    kwargs.update(id=wandb_run_id, resume='must')
                self._run = module.init(**kwargs)
            except Exception as exc:
                raise RuntimeError(
                    f"W&B {wandb_mode} run could not be started; check W&B installation/authentication"
                ) from exc

    @property
    def url(self) -> str | None:
        return getattr(self._run, "url", None)

    def log(self, metrics: Mapping[str, Any], *, step: int) -> None:
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError('step must be an integer')
        if 'step' in metrics:
            raise ValueError("metrics must not contain reserved key 'step'")
        row = {'step': step, **dict(metrics)}
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        if self._run is not None:
            self._run.log(dict(metrics), step=step)

    log_metrics = log

    def log_checkpoint(self, checkpoint: str | Path, *, step: int | None = None) -> None:
        """Upload a checkpoint only when explicitly enabled."""
        if not self.upload_checkpoints or self._run is None:
            return
        path = str(checkpoint)
        saver = getattr(self._run, 'save', None) or getattr(self._wandb, 'save', None)
        if saver is None:
            raise RuntimeError('W&B checkpoint upload was enabled, but the SDK has no save method')
        saver(path)

    def finish(self, *, success: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        if self._run is not None:
            self._run.finish(exit_code=0 if success else 1)

    close = finish


def add_training_log_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared, optional W&B arguments to a training CLI."""
    parser.add_argument('--wandb-mode', choices=('online', 'offline', 'disabled'), default='disabled')
    parser.add_argument('--wandb-project', default='Thekenyos')
    parser.add_argument('--wandb-entity')
    parser.add_argument('--wandb-name')
    parser.add_argument('--upload-checkpoints', action='store_true')


def training_log_from_args(output: str | Path, args: Any, config: Mapping[str, Any]) -> TrainingLog:
    return TrainingLog(output, config, wandb_mode=args.wandb_mode,
                       wandb_project=args.wandb_project, wandb_entity=args.wandb_entity,
                       wandb_name=args.wandb_name, upload_checkpoints=args.upload_checkpoints)
