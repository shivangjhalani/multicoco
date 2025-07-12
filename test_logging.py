import json
import logging
import os
import tempfile
from pathlib import Path

def test_evaluation_logging():
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f'Testing logging in: {temp_dir}')
        eval_logger = logging.getLogger('test_evaluation_details')
        eval_logger.setLevel(logging.INFO)
        eval_logger.propagate = False
        if eval_logger.hasHandlers():
            eval_logger.handlers.clear()
        json_formatter = logging.Formatter('%(message)s')
        eval_log_path = os.path.join(temp_dir, 'evaluation.log')
        eval_handler = logging.FileHandler(eval_log_path)
        eval_handler.setFormatter(json_formatter)
        eval_logger.addHandler(eval_handler)
        test_sample = {'question': 'What does the yellow sign advise you to watch for? The choices are 0 : pedestrians, 1 : speedbumps, 2 : dogs, 3 : deer', 'ground_truth': '0', 'generated_answer': '0 : pedestrians', 'extracted_answer': '0', 'generated_tokens': 3, 'correct': True}
        eval_logger.info(json.dumps(test_sample))
        eval_logger.handlers.clear()
        epoch = 0
        epoch_eval_log_path = os.path.join(temp_dir, f'evaluation_epoch_{epoch + 1}.log')
        epoch_eval_handler = logging.FileHandler(epoch_eval_log_path)
        epoch_eval_handler.setFormatter(json_formatter)
        eval_logger.addHandler(epoch_eval_handler)
        eval_logger.info(json.dumps(test_sample))
        print('✓ Testing evaluation.log')
        assert os.path.exists(eval_log_path), 'evaluation.log was not created'
        with open(eval_log_path, 'r') as f:
            content = f.read().strip()
            assert content.startswith('{"question"'), f'Expected JSON content, got: {content[:50]}'
            parsed = json.loads(content)
            assert parsed['correct'] == True, 'JSON parsing failed'
        print('✓ Testing evaluation_epoch_1.log')
        assert os.path.exists(epoch_eval_log_path), 'evaluation_epoch_1.log was not created'
        with open(epoch_eval_log_path, 'r') as f:
            content = f.read().strip()
            assert content.startswith('{"question"'), f'Expected JSON content, got: {content[:50]}'
            parsed = json.loads(content)
            assert parsed['correct'] == True, 'JSON parsing failed'
        print('✓ All tests passed!')
        print(f'evaluation.log contains: {Path(eval_log_path).read_text()[:100]}...')
        print(f'evaluation_epoch_1.log contains: {Path(epoch_eval_log_path).read_text()[:100]}...')
if __name__ == '__main__':
    test_evaluation_logging()