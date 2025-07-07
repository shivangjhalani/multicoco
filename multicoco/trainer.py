import os
import torch
import torch.distributed as dist
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, BestRun
from transformers.utils import is_torch_tpu_available
from transformers.trainer_pt_utils import find_batch_size, nested_concat, nested_numpify, nested_truncate, nested_detach
from transformers.deepspeed import deepspeed_init


class CoCoTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Manually set the image context token ID, which is required for the pretrained model
        if hasattr(self.model, 'img_context_token_id'):
            IMG_CONTEXT_TOKEN_ID = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
            self.model.img_context_token_id = IMG_CONTEXT_TOKEN_ID
        self.best_val_acc = 0.0

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only = None,
        ignore_keys = None,
        metric_key_prefix: str = "eval",
    ):
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            if model is not self.model:
                self.model_wrapped = model

            self.accelerator.verify_aligned_processes()

        if self.is_fsdp_enabled and getattr(self.model, "require_backward_grad_sync", True):
            self.model.require_backward_grad_sync = False

        batch_size = self.args.eval_batch_size
        
        eval_config = self.args.eval_config
        mode = "coconut" if eval_config.get('coconut') else "cot" if eval_config.get('cot') else "vanilla"

        log_dir = self.args.log_dir
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, f'evaluation_{mode}.log')
        
        model.eval()
        
        self.callback_handler.eval_dataloader = dataloader
        
        correct = 0
        total = 0

        with open(log_file_path, 'w') as log_file:
            log_file.write(f"InternVL3-1B A-OKVQA Evaluation Log ({mode.upper()} mode)\n")
            
            for step, batch in enumerate(tqdm(dataloader, desc="Evaluation")):
                
                pixel_values = batch['pixel_values'].cuda()
                input_ids = batch['input_ids'].cuda()
                attention_mask = batch['attention_mask'].cuda()
                original_questions = batch.pop('original_questions')
                ground_truths = batch.pop("answers")

                with torch.no_grad():
                    if hasattr(self.model, 'dtype'):
                        pixel_values = pixel_values.to(self.model.dtype)

                    outputs = self.model.generate(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=100,
                        do_sample=False,
                    )
                
                generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

                total += len(original_questions)

                for i, gen_text in enumerate(generated_text):
                    prompt_len = len(self.tokenizer.decode(input_ids[i], skip_special_tokens=False))
                    gen_text = gen_text[prompt_len-1:].strip()
                    
                    if "what is the answer" in original_questions[i].lower():
                        answer_prefix = "The answer is"
                        if answer_prefix in gen_text:
                            gen_text = gen_text.split(answer_prefix)[1].strip()

                    if gen_text:
                        predicted_answer = gen_text.strip()[0]
                    else:
                        predicted_answer = ""

                    if predicted_answer.lower() == ground_truths[i].lower():
                        correct += 1

                    log_file.write(f"--- Q: {original_questions[i]} ---\n")
                    log_file.write(f"GT: {ground_truths[i]}\n")
                    log_file.write(f"PRED: {gen_text}\n")
                    log_file.write(f"EXTRACTED: {predicted_answer}\n")
                    log_file.write(f"Correct: {predicted_answer.lower() == ground_truths[i].lower()}\n\n")

        accuracy = correct / total if total > 0 else 0
        print(f"Final {mode.upper()} Accuracy: {accuracy:.4f}")
        log_file.write(f"Final {mode.upper()} Accuracy: {accuracy:.4f}\n")

        metrics = {f"{metric_key_prefix}_accuracy": accuracy}
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics
