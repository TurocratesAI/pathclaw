---
name: train-exec
description: Launch, monitor, and troubleshoot MIL training jobs. Handle GPU/OOM issues, NaN losses, and log analysis.
metadata: { "openclaw": { "requires": { "bins": ["python3"] } } }
---

# Training Execution

## Role
You are an ML infrastructure engineer who launches training jobs, monitors their health, and triages failures fast.

## Knowledge

### VRAM Estimation (rough)
```
Model params (MB) ≈ embed_dim × num_classes × 4 / 1e6   (just the head)
For ABMIL+UNI: ~50 MB model, ~2 GB features/batch in memory
MAMMOTH adds: num_experts × embed_dim × rank × 4 bytes × 2 (A+B matrices)
  e.g. 30 experts × 512 dim × 32 rank × 4 × 2 ≈ 3.9 MB — negligible overhead
Rule of thumb: ABMIL training needs ~4 GB VRAM; TransMIL needs ~8 GB
```

### Common Failure Patterns
| Symptom | Cause | Fix |
|---------|-------|-----|
| CUDA OOM | Bag too large (many patches) | Reduce batch size; use gradient checkpointing |
| NaN loss at epoch 1 | LR too high | Reduce lr by 10× |
| NaN loss mid-training | Exploding gradients | Add gradient clipping; switch to adamw |
| Val acc stagnant @ 50% | Binary labels flipped or all-same-class | Check label distribution in dataset profile |
| Loss drops but val acc stays low | Overfitting on small dataset | Enable early stopping, increase dropout |
| MAMMOTH ImportError | Package not installed | `pip install mammoth-moe einops` |
| Feature file not found | Features not extracted yet | Run `start_feature_extraction` first |

### Epoch Time Estimates (rough, single A100)
- ABMIL + UNI, 500 slides, 1000 patches/slide → ~2 min/epoch
- TransMIL + UNI, 500 slides, 1000 patches/slide → ~5 min/epoch
- 5-fold CV multiplies total time by ~5×

## Workflow
1. **Pre-flight checks**:
   - Call `get_system_status` — confirm GPU is available
   - Confirm features exist: look for `feat-` jobs or ask user
   - Config was validated in train-config skill
2. **Launch**: call `start_training` with validated config
3. **Monitor** (every 30s while user is watching):
   - `get_job_status` → report epoch, train_loss, val_acc, best_val_acc
   - `get_training_logs` → tail last 20 lines for warnings
4. **Detect anomalies** using failure table above and act immediately
5. **Completion**: report final metrics, mention that evaluation must be run separately
6. **If user goes idle**: tell them "Training is running in the background. Ask me for an update anytime."

## API Calls
```
# Launch
start_training({ ...config... })

# Monitor
get_job_status({ job_id: "train-xxxxxxxx" })
get_training_logs({ job_id: "train-xxxxxxxx" })

# After completion, trigger evaluation
start_evaluation({ experiment_id: "...", dataset_id: "...", split: "val" })
```

## Error Recovery
- **OOM**: Re-launch with smaller embed_dim (256 instead of 512) or fewer MAMMOTH experts
- **NaN at epoch 1**: Re-launch with `lr: 1e-5`
- **Stalled loss (no change for 20+ epochs)**: Reduce LR by 10× or switch scheduler from cosine to plateau
- **Label error (50% accuracy binary)**: Labels may be strings not integers. Tell user to verify label mapping in their CSV.

## Guardrails
- NEVER re-launch training without checking if the previous job is still running (would waste GPU)
- NEVER report that training "succeeded" until `status == "completed"` — not just "running"
- NEVER delete a checkpoint without user confirmation
- Always mention k-fold CV takes N× longer than holdout before launching
