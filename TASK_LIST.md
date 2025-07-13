# MultiCoCo Fix Task List

## Critical Issues (Must Fix for Latent Reasoning to Work)

### Phase 1: Data Pipeline Fixes
- [x] **Task 1.1**: Fix progressive curriculum to preserve image fields
- [x] **Task 1.2**: Ensure reasoning field is preserved in __getitem__
- [x] **Task 1.3**: Update collate_fn to handle reasoning with latent tokens
- [x] **Task 1.4**: Fix image token consistency (use <img> everywhere)

### Phase 2: Token and Tokenizer Fixes  
- [x] **Task 2.1**: Add missing special tokens (chat markers, image tokens)
- [x] **Task 2.2**: Update constants to use consistent image token
- [x] **Task 2.3**: Verify token ID consistency across codebase

### Phase 3: Latent Injection Mechanism Redesign
- [x] **Task 3.1**: Simplify LatentWrapper to separate multimodal prep from latent injection
- [x] **Task 3.2**: Implement clean Coconut-style multi-pass forward
- [x] **Task 3.3**: Remove complex KV caching that breaks the latent injection logic
- [x] **Task 3.4**: Ensure vision embeddings are computed once and reused

### Phase 4: Integration and Testing
- [x] **Task 4.1**: Add generation config support in evaluation
- [ ] **Task 4.2**: Test data pipeline end-to-end
- [ ] **Task 4.3**: Verify latent spans are detected and processed
- [ ] **Task 4.4**: Test multimodal + latent reasoning integration

## Implementation Order (One Task at a Time)

1. Start with data pipeline (Phase 1) - foundation must be solid
2. Fix tokenization issues (Phase 2) - ensure clean token handling  
3. Redesign latent injection (Phase 3) - core functionality
4. Integration testing (Phase 4) - verify everything works together

## Success Criteria

- [x] Progressive curriculum preserves all data fields including images
- [x] Latent tokens are properly tokenized and detected
- [x] LatentWrapper successfully injects hidden states into latent positions
- [ ] Multimodal reasoning works with latent tokens
- [x] Generation respects config parameters
- [x] No KeyError or data loss during training/evaluation

## Risk Mitigation

- Make incremental changes with testing at each step
- Keep backup versions of working components
- Add logging to verify data flow at each stage
- Test with small dataset first before full training
