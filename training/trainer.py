import torch
from torch.nn.utils import clip_grad_norm_

from model.masks import (
    create_padding_mask,
    create_decoder_mask
)

from training.batch_utils import (
    prepare_decoder_inputs
)


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        loss_fn,
        device,
        grad_clip: float = 1.0,
        gradient_accumulation_steps: int = 1,
        use_amp: bool = True
    ):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device

        self.grad_clip = grad_clip

        self.gradient_accumulation_steps = (
            gradient_accumulation_steps
        )

        self.use_amp = (
            use_amp
            and device.type == "cuda"
        )

        self.scaler = (
            torch.amp.GradScaler("cuda")
            if self.use_amp
            else None
        )

    # ========================================================
    # FORWARD + LOSS
    # ========================================================

    def _forward_and_loss(
        self,
        src,
        tgt
    ):

        decoder_input, expected_output = (
            prepare_decoder_inputs(tgt)
        )

        src_mask = create_padding_mask(
            src
        )

        tgt_mask = create_decoder_mask(
            decoder_input
        )

        with torch.amp.autocast(
            device_type=self.device.type,
            enabled=self.use_amp
        ):

            output = self.model(
                src_tokens=src,
                tgt_tokens=decoder_input,
                src_mask=src_mask,
                tgt_mask=tgt_mask
            )

            logits = output["logits"]

            loss = self.loss_fn(
                logits,
                expected_output
            )

        return loss

    # ========================================================
    # OPTIMIZER STEP
    # ========================================================

    def _optimizer_step(self):

        if self.scaler is not None:

            self.scaler.unscale_(
                self.optimizer
            )

            clip_grad_norm_(
                self.model.parameters(),
                self.grad_clip
            )

            self.scaler.step(
                self.optimizer
            )

            self.scaler.update()

        else:

            clip_grad_norm_(
                self.model.parameters(),
                self.grad_clip
            )

            self.optimizer.step()

        self.scheduler.step()

        self.optimizer.zero_grad(
            set_to_none=True
        )

    # ========================================================
    # TRAIN ONE EPOCH
    # ========================================================

    def train_epoch(
        self,
        dataloader,
        max_batches: int | None = None
    ):

        self.model.train()

        total_loss = 0.0
        total_batches = 0

        self.optimizer.zero_grad(
            set_to_none=True
        )

        for batch_index, batch in enumerate(
            dataloader
        ):

            # ------------------------------------------------
            # Stop early for smoke test
            # ------------------------------------------------

            if (
                max_batches is not None
                and batch_index >= max_batches
            ):
                break

            src = batch["src"].to(
                self.device,
                non_blocking=True
            )

            tgt = batch["tgt"].to(
                self.device,
                non_blocking=True
            )

            loss = self._forward_and_loss(
                src,
                tgt
            )

            # ------------------------------------------------
            # Gradient accumulation
            # ------------------------------------------------

            loss_for_backward = (
                loss
                / self.gradient_accumulation_steps
            )

            if self.scaler is not None:

                self.scaler.scale(
                    loss_for_backward
                ).backward()

            else:

                loss_for_backward.backward()

            processed_batches = (
                batch_index + 1
            )

            should_update = (
                processed_batches
                % self.gradient_accumulation_steps
                == 0
            )

            reached_limit = (
                max_batches is not None
                and processed_batches >= max_batches
            )

            reached_dataloader_end = (
                processed_batches == len(dataloader)
            )

            if (
                should_update
                or reached_limit
                or reached_dataloader_end
            ):

                self._optimizer_step()

            total_loss += loss.item()

            total_batches += 1

        if total_batches == 0:

            raise ValueError(
                "No batches were processed."
            )

        return (
            total_loss
            / total_batches
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    @torch.no_grad()
    def validate(
        self,
        dataloader,
        max_batches: int | None = None
    ):

        self.model.eval()

        total_loss = 0.0
        total_batches = 0

        for batch_index, batch in enumerate(
            dataloader
        ):

            # ------------------------------------------------
            # Stop early for smoke test
            # ------------------------------------------------

            if (
                max_batches is not None
                and batch_index >= max_batches
            ):
                break

            src = batch["src"].to(
                self.device,
                non_blocking=True
            )

            tgt = batch["tgt"].to(
                self.device,
                non_blocking=True
            )

            loss = self._forward_and_loss(
                src,
                tgt
            )

            total_loss += loss.item()

            total_batches += 1

        if total_batches == 0:

            raise ValueError(
                "No validation batches were processed."
            )

        return (
            total_loss
            / total_batches
        )