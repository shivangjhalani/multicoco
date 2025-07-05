import logging

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import LogitsProcessorList

from .generation import SingleDigitLogitsProcessor


logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        config,
        model,
        tokenizer,
        train_dataloader=None,
        eval_dataloader=None,
        optimizer=None,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.device = next(model.parameters()).device if hasattr(model, 'parameters') and next(model.parameters(), None) is not None else torch.device('cpu')


    def train(self):
        if self.train_dataloader is None or self.optimizer is None:
            logger.error("Trainer is not configured for training. Missing train_dataloader or optimizer.")
            return
            
        self.model.train()
        for epoch in range(self.config['epochs']):
            if dist.is_initialized() and hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(epoch)

            for batch in tqdm(self.train_dataloader, desc=f"Training Epoch {epoch}"):
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
                
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            logger.info(f"Epoch {epoch} loss: {loss.item()}")

            if self.eval_dataloader:
                self.evaluate(epoch)

    def evaluate(self, epoch=0):
        if self.eval_dataloader is None:
            logger.warning("No evaluation dataloader provided, skipping evaluation.")
            return

        self.model.eval()
        all_results = []
        
        # On main process, clear log file and write header
        if not dist.is_initialized() or dist.get_rank() == 0:
            with open("evaluation.log", "w") as f:
                f.write("========================================\n")
                f.write("        Evaluation Log\n")
                f.write("========================================\n\n")

        logits_processor = LogitsProcessorList([SingleDigitLogitsProcessor(self.tokenizer)])

        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device)
            
            # Separate model inputs from evaluation metadata
            model_inputs = {
                "input_ids": batch.get("input_ids"),
                "attention_mask": batch.get("attention_mask"),
                "pixel_values": batch.get("pixel_values"),
            }
            # The patched model expects image_flags, the vanilla one does not.
            if 'image_flags' in batch:
                if self.config.get('cot') or self.config.get('coconut'):
                    model_inputs['image_flags'] = batch['image_flags']

            ground_truth_answers = batch.get("direct_answers", [])
            original_questions = batch.get("original_questions", [])

            with torch.no_grad():
                model_to_generate = self.model.module if hasattr(self.model, 'module') else self.model

                outputs = model_to_generate.generate(
                    **model_inputs,
                    max_new_tokens=2,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    logits_processor=logits_processor,
                )

            # Decode generated tokens
            input_ids_len = batch["input_ids"].shape[1]
            generated_texts = self.tokenizer.batch_decode(outputs[:, input_ids_len:], skip_special_tokens=True)
            
            for j, (gt_ans, gen_text) in enumerate(zip(ground_truth_answers, generated_texts)):
                question = original_questions[j] if j < len(original_questions) else "Question not found"
                is_correct = gen_text.strip() in gt_ans
                all_results.append(is_correct)
                
                # Only log from the main process
                if not dist.is_initialized() or dist.get_rank() == 0:
                    log_entry = (
                        f"---------- Sample {j+1} ----------\n"
                        f"Question: {question}\n"
                        f"Generated Answer: {gen_text.strip()}\n"
                        f"Ground Truth: {gt_ans}\n"
                        f"Correct: {'Yes' if is_correct else 'No'}\n"
                        f"---------------------------------\n\n"
                    )
                    with open("evaluation.log", "a") as f:
                        f.write(log_entry)

        # Aggregate results across all processes
        if dist.is_initialized():
            world_size = dist.get_world_size()
            gathered_results = [None] * world_size
            dist.all_gather_object(gathered_results, all_results)
            all_results = [item for sublist in gathered_results for item in sublist]

        if not all_results:
            logger.warning("Evaluation produced no results.")
            if not dist.is_initialized() or dist.get_rank() == 0:
                return 0.0
            return

        accuracy = sum(all_results) / len(all_results)
        
        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info(f"Epoch {epoch} accuracy: {accuracy}")
            
        return accuracy
