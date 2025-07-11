#!/usr/bin/env python3
"""
Test script to verify distributed evaluation is working correctly.
This script helps debug the multi-GPU evaluation setup.
"""

import os
import sys
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler


class SimpleTestDataset(Dataset):
    """Simple test dataset to verify distributed sampling."""
    
    def __init__(self, size=20):
        self.size = size
        self.data = [f"sample_{i}" for i in range(size)]
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            'id': idx,
            'data': self.data[idx],
            'questions': [f"question_{idx}"],
            'answers': [f"answer_{idx}"],
            'pixel_values': torch.randn(3, 224, 224)  # Dummy image data
        }


def test_distributed_sampling():
    """Test that DistributedSampler properly splits data across processes."""
    
    # Initialize distributed if not already done
    if not dist.is_initialized():
        print("Distributed training not initialized. This test requires torchrun.")
        return
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(f"Process {rank}/{world_size}: Starting distributed sampling test")
    
    # Create test dataset
    dataset = SimpleTestDataset(size=20)
    
    # Create distributed sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=2,
        collate_fn=lambda batch: {
            'questions': [item['questions'][0] for item in batch],
            'answers': [item['answers'][0] for item in batch],
            'pixel_values': torch.stack([item['pixel_values'] for item in batch]),
            'ids': [item['id'] for item in batch]
        }
    )
    
    # Process batches and collect sample IDs
    processed_ids = []
    processed_samples = []
    
    for batch_idx, batch in enumerate(dataloader):
        batch_ids = batch['ids']
        batch_questions = batch['questions']
        
        processed_ids.extend(batch_ids)
        processed_samples.extend(batch_questions)
        
        print(f"Process {rank}: Batch {batch_idx}, IDs: {batch_ids}, Questions: {batch_questions}")
    
    print(f"Process {rank}: Processed {len(processed_ids)} samples with IDs: {processed_ids}")
    
    # Gather results from all processes
    gathered_ids = [None for _ in range(world_size)]
    gathered_samples = [None for _ in range(world_size)]
    
    dist.all_gather_object(gathered_ids, processed_ids)
    dist.all_gather_object(gathered_samples, processed_samples)
    
    if rank == 0:
        print("\n" + "="*50)
        print("DISTRIBUTED SAMPLING TEST RESULTS")
        print("="*50)
        
        all_ids = []
        all_samples = []
        
        for r in range(world_size):
            print(f"Process {r} processed IDs: {gathered_ids[r]}")
            print(f"Process {r} processed samples: {gathered_samples[r]}")
            
            if gathered_ids[r] is not None:
                all_ids.extend(gathered_ids[r])
            if gathered_samples[r] is not None:
                all_samples.extend(gathered_samples[r])
        
        print(f"\nCombined results:")
        print(f"Total samples processed: {len(all_samples)}")
        print(f"Total unique IDs: {len(set(all_ids))}")
        print(f"All processed IDs: {sorted(all_ids)}")
        
        # Verify correctness
        expected_ids = list(range(20))
        if sorted(all_ids) == expected_ids:
            print("✅ SUCCESS: All samples processed exactly once!")
        else:
            print("❌ FAILURE: Sample duplication or missing samples detected!")
            print(f"Expected: {expected_ids}")
            print(f"Got: {sorted(all_ids)}")
        
        print("="*50)


if __name__ == "__main__":
    test_distributed_sampling() 