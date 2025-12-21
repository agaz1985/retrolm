## Misc.
1. Retrain model with GPU

## Performance Optimizations

### Critical (High Impact)
1. **KV caching**: Recomputes entire attention every token → O(n²) latency growth
   - Context reduced to 16 chars as workaround
   - Proper implementation would enable 64-256 token context without slowdown
   
2. **Memory pooling in generation loop** (`chat.c:53-91`)
   - Allocates/frees input matrix + logits matrix every token (200+ mallocs per response)
   - Pre-allocate buffers and reuse → ~30-50% speedup

### Medium Impact
4. **MMX vectorization** (Pentium II has MMX)
   - Vectorize element-wise ops (`_matadd`, `_matsub`, softmax)
   - 2x speedup for add/sub operations
   - Example: Process 2 floats at once with MMX intrinsics
   
5. **Static allocation for fixed-size buffers**
   - `all_tokens`, `response` buffers in `generate_interactive()`
   - Stack allocation avoids heap fragmentation on retro systems
   
6. **Batch matrix cleanup**
   - Free multiple matrices at end of loop instead of incrementally
   - Reduces allocator overhead

### Low Impact (Nice to Have)
7. **Remove fflush(stdout)** in token loop (`chat.c:107`)
   - Flushes every char → heavy I/O overhead
   - Buffer until newline or every N tokens
   
8. **Quantization** (int8/int16)
   - 2-4x memory reduction
   - Faster matrix ops on retro CPUs (integer ALU)
   - Requires retraining or post-training quantization
   
9. **Better tokenization** (BPE vs byte-level)
   - Byte-level = 1 char per token (inefficient)
   - BPE 512-1024 vocab = 3-4x compression → shorter sequences
   - Requires retraining tokenizer and model