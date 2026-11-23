from dataclasses import dataclass, fields
import json
from pathlib import Path

VARIANTS = ("full", "no_validity", "no_reciprocal_weighting", "no_gate", "constant_gate")


@dataclass
class Config:
    seed: int = 42
    image_height: int = 192
    image_width: int = 640
    token_height: int = 12
    token_width: int = 40
    channels: int = 128
    attention_heads: int = 4
    gamma: float = 1.
    range_m: float = 80.
    rotation_perturb_deg: float = 10.
    translation_perturb_m: float = .25
    rotation_twist_bound_deg: float = 15.
    translation_twist_bound_m: float = .35
    train_steps: int = 2
    eval_steps: int = 3
    batch_size: int = 4
    epochs: int = 50
    lr: float = 1e-4
    lr_step: int = 10
    lr_gamma: float = .5
    lambda_rotation: float = 1.
    lambda_translation: float = 2.
    lambda_projection: float = .005
    smooth_l1_beta_px: float = 3.
    invalid_penalty: float = 670.
    invalid_slope: float = 100.
    depth_epsilon_m: float = .001
    success_rotation_deg: float = .5
    success_translation_cm: float = 3.
    variant: str = "full"
    workers: int = 0
    rgb_mean: tuple = (.485, .456, .406)
    rgb_std: tuple = (.229, .224, .225)

    def validate(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"Unknown variant {self.variant}")
        if self.channels % self.attention_heads:
            raise ValueError("channels must be divisible by attention_heads")
        if min(self.image_height, self.image_width, self.token_height, self.token_width,
               self.train_steps, self.eval_steps, self.batch_size, self.epochs) <= 0:
            raise ValueError("Sizes and step counts must be positive")
        return self


def load_config(path=None, **overrides):
    values = json.loads(Path(path).read_text()) if path else {}
    values.update({k: v for k, v in overrides.items() if v is not None})
    unknown = values.keys() - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return Config(**values).validate()
