import math
import torch


class NoamScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        warmup_steps: int = 4000
    ):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def get_learning_rate(self) -> float:
        step = max(self.step_num, 1)

        scale = self.d_model ** (-0.5)

        warmup_factor = min(
            step ** (-0.5),
            step * (self.warmup_steps ** (-1.5))
        )

        return scale * warmup_factor

    def step(self):
        self.step_num += 1

        learning_rate = self.get_learning_rate()

        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = learning_rate

    def get_last_lr(self):
        return [
            parameter_group["lr"]
            for parameter_group in self.optimizer.param_groups
        ]

    def state_dict(self):
        return {
            "step_num": self.step_num
        }

    def load_state_dict(self, state_dict):
        self.step_num = state_dict["step_num"]

        learning_rate = self.get_learning_rate()

        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = learning_rate