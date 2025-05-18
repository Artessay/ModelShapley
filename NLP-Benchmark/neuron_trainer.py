import time
import torch
import torch.nn as nn
import os

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

from typing import Optional

import neural_hook
from utils import setup_logger


class NeuronTrainer:
    def __init__(
            self, 
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            config: Optional[dict] = None
    ):
        # Basic settings
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Training configuration
        self.config = config or {}
        self.epochs = self.config.get("epochs", 1)
        self.lr = self.config.get("learning_rate", 3e-5)
        self.weight_decay = self.config.get("weight_decay", 0.0)
        self.max_grad_norm = self.config.get("max_grad_norm", 1.0)
        self.warmup_ratio = self.config.get("warmup_ratio", 1e-3)
        self.gradient_accumulation_steps = self.config.get("gradient_accumulation_steps", 4)
        self.logging_steps = self.config.get("logging_steps", 1)
        self.eval_steps = self.config.get("eval_steps", 0)
        self.save_steps = self.config.get("save_steps", 200)
        self.output_dir = self.config.get("output_dir", "./sft_model")
        self.neuron_mask_dict = self.config.get("neuron_mask_dict", {})
        log_path = self.config.get("log_path", "./logs/demo.log")

        # Optimizer and learning rate scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Training state
        self.global_step = 0
        self.best_eval_loss = float('inf')

        self.logger = setup_logger(log_path)

    # Mask only important neurons
    def _set_param_freeze_hook(self):
        neural_hook.set_neuron_freeze_hook(self.model, self.neuron_mask_dict)

    def _clear_optimizer_buffers(self):
        neural_hook.clear_optimizer_buffers(self.model, self.optimizer, self.neuron_mask_dict)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(optimizer_grouped_parameters, lr=self.lr)
    
    def _create_scheduler(self) -> LambdaLR:
        """Create learning rate scheduler"""
        num_training_steps = len(self.train_loader) * self.epochs // self.gradient_accumulation_steps
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        return get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
    

    def train(self):
        """Execute model training"""
        self.logger.info("***** Start Training *****")
        self.logger.info(f"  Num examples = {len(self.train_loader.dataset)}")
        self.logger.info(f"  Num epochs = {self.epochs}")
        self.logger.info(f"  Batch size = {self.train_loader.batch_size}")
        self.logger.info(f"  Gradient accumulation steps = {self.gradient_accumulation_steps}")
        self.logger.info(f"  Total training steps = {len(self.train_loader) * self.epochs // self.gradient_accumulation_steps}")
        
        # Training start time
        start_time = time.time()
        self._set_param_freeze_hook()
        
        # Training loop
        for epoch in range(self.epochs):
            self.logger.info(f"***** Epoch {epoch+1}/{self.epochs} *****")
            
            # Set to training mode
            self.model.train()
            total_loss = 0.0
            self.optimizer.zero_grad()
            
            # Training progress bar
            train_iter = self.train_loader # tqdm(self.train_loader, ncols=80, desc=f"Epoch {epoch+1}")
            
            for step, model_inputs in enumerate(train_iter):
                # Forward pass
                loss = self._compute_loss_and_backward(model_inputs, do_backward=True)
                
                # Gradient accumulation
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps
                
                # Gradient clipping
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    if self.max_grad_norm > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    # Optimizer update
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    self._clear_optimizer_buffers()
                    
                    # Logging
                    total_loss += loss.item()
                    if self.global_step % self.logging_steps == 0:
                        # Calculate average loss
                        avg_loss = total_loss / self.logging_steps
                        total_loss = 0.0
                        
                        # Get current learning rate
                        current_lr = self.scheduler.get_last_lr()[0]
                        
                        # Calculate training progress
                        progress = self.global_step / (len(self.train_loader) * self.epochs // self.gradient_accumulation_steps)
                        eta = (time.time() - start_time) * (1 / progress - 1)
                        
                        # Log information
                        self.logger.info(f"Step {self.global_step} - lr: {current_lr:.2e} - loss: {avg_loss:.4f} - ETA: {self._format_time(eta)}")
                    
                    # Evaluate model
                    if self.val_loader is not None and self.eval_steps > 0 and self.global_step % self.eval_steps == 0:
                        eval_loss = self.evaluate()
                        self.logger.info(f"Step {self.global_step} - eval_loss: {eval_loss:.4f}")
                        
                        # Save best model
                        if eval_loss < self.best_eval_loss:
                            self.best_eval_loss = eval_loss
                            self.save_model(f"{self.output_dir}/best")
                    
                    # Save model
                    if self.save_steps > 0 and self.global_step % self.save_steps == 0:
                        self.save_model(f"{self.output_dir}/checkpoint-{self.global_step}")
        
        # Training completed
        # Save final model
        self.save_model(self.output_dir)
        
        # Calculate total training time
        total_time = time.time() - start_time
        self.logger.info(f"Training completed, total time: {self._format_time(total_time)}")
    
    def evaluate(self) -> float:
        """Evaluate model performance"""
        self.logger.info("***** Start Evaluation *****")
        
        # Set to evaluation mode
        self.model.eval()
        total_loss = 0.0
        eval_steps = 0
        
        with torch.no_grad():
            val_iter = tqdm(self.val_loader, ncols=80, desc="Evaluation")
            for model_inputs in val_iter:
                # Forward pass
                loss = self._compute_loss_and_backward(model_inputs, do_backward=False)
                
                # Accumulate loss
                total_loss += loss.item()
                eval_steps += 1
        
        # Calculate average loss
        avg_loss = total_loss / eval_steps
        
        # Resume training mode
        self.model.train()
        
        return avg_loss
    
    def save_model(self, output_dir: str):
        """Save model and tokenizer"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training configuration
        torch.save(self.config, os.path.join(output_dir, "training_config.bin"))
        self.logger.info(f"Model saved to {output_dir}")
    
    def _compute_loss_and_backward(self, model_inputs: dict, do_backward: bool = True):
        """
        Compute the loss and perform backward pass for the model.
        Args:
            model (PreTrainedModel): The Vision Transformer model.
            model_inputs (dict): The model inputs.
        Returns:
            None: The function modifies the requires_grad attribute of model parameters in-place.
        """
        model = self.model
        input_ids: torch.Tensor = model_inputs["input_ids"].cuda()
        loss_mask: torch.Tensor = model_inputs.pop("loss_mask")[:, :-1].reshape(-1).cuda()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        # Forward pass
        outputs = model(**model_inputs)
        logits: torch.Tensor = outputs.logits
        labels = input_ids[:, 1:].contiguous()

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels.contiguous()
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        loss = loss * loss_mask.to(loss.device)

        valid_token_num = torch.sum(loss_mask)
        loss = torch.sum(loss) / (valid_token_num + 1e-8)

        if do_backward:
            loss.backward()
        return loss

    def _format_time(self, seconds: float) -> str:
        """Convert seconds to human-readable time format"""
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            return f"{seconds/60:.2f}m"
        else:
            return f"{seconds/3600:.2f}h"