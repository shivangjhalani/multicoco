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
        train_dataloader,
        eval_dataloader,
        tokenizer,
    ):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.tokenizer = tokenizer

    def train(self):
        for epoch in range(self.config.epochs):
            self.model.train()
            for batch in self.train_dataloader:
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.config.device)
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                # optimizer.step()
                # optimizer.zero_grad()
            logger.info(f"Epoch {epoch} loss: {loss.item()}")

            if self.eval_dataloader:
                self.evaluate(epoch)

    def evaluate(self, epoch=0):
        if self.eval_dataloader:
            self.model.eval()
            all_results = []
            
            logits_processor = LogitsProcessorList([SingleDigitLogitsProcessor(self.tokenizer)])

            for i, batch in enumerate(tqdm(self.eval_dataloader, desc="Evaluating")):
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.config.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **batch,
                        max_new_tokens=2,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                        logits_processor=logits_processor,
                    )

                # Decode generated tokens
                generated_texts = self.tokenizer.batch_decode(outputs[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
                
                ground_truth_answers = batch.pop("direct_answers")

                for j, (gt_ans, gen_text) in enumerate(zip(ground_truth_answers, generated_texts)):
                    question = self.tokenizer.decode(batch['input_ids'][j], skip_special_tokens=True)
                    # The ground truth is a list of possible correct answers.
                    # We consider the prediction correct if it's in the list.
                    is_correct = gen_text.strip() in gt_ans
                    all_results.append(is_correct)
                    
                    log_entry = f"Question: {question}\nGenerated: {gen_text.strip()}\nGround Truth: {gt_ans}\nCorrect: {is_correct}\n---\n"
                    with open("evaluation.log", "a") as f:
                        f.write(log_entry)


            # Aggregate results across all processes
            if dist.is_initialized():
                # Gather results from all processes
                world_size = dist.get_world_size()
                gathered_results = [None] * world_size
                dist.all_gather_object(gathered_results, all_results)
                
                # Flatten the list of lists
                all_results = [item for sublist in gathered_results for item in sublist]


            if not all_results:
                logger.warning("Evaluation produced no results.")
                return

            accuracy = sum(all_results) / len(all_results)
            logger.info(f"Epoch {epoch} accuracy: {accuracy}")
