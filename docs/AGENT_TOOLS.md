# Agent Tools

PathClaw exposes 66 tools to the in-app LLM agent. Reference list grouped by
category. Full JSON schemas live in
`backend/pathclaw/api/routes/chat.py` (search for `TOOLS =`).

To regenerate this file from source:

```bash
python scripts/dump_tools.py > docs/AGENT_TOOLS.md
```

---

## Memory & manuscript (7)

| Tool | Purpose |
|------|---------|
| `remember_fact` | Persist a fact to global memory across sessions |
| `recall_facts` | Retrieve stored facts by query |
| `write_note` | Append to the current session's notes.md |
| `read_notes` | Read the session notebook |
| `write_manuscript` | Edit `main.tex` / `refs.bib` for the session manuscript |
| `read_manuscript` | Read manuscript files |
| `compile_manuscript` | Run tectonic/pdflatex to produce a PDF |

## Datasets (4)

| Tool | Purpose |
|------|---------|
| `list_datasets` | List registered datasets for the current session |
| `register_dataset` | Register a folder of slides + labels CSV |
| `list_dataset_slides` | Enumerate slides in a dataset |
| `get_dataset_profile` | Class balance, label column diagnostics, slide-row match rate |

## GDC / TCGA (3)

| Tool | Purpose |
|------|---------|
| `search_gdc` | Query the GDC API with filters (project, data type, diagnosis) |
| `download_gdc` | Fetch GDC files with resumable checkpoints |
| `gdc_job_status` | Poll a running download job |

## Preprocessing & training (4)

| Tool | Purpose |
|------|---------|
| `start_preprocessing` | Tissue mask + patching at a given magnification |
| `start_training` | Kick off a MIL training run |
| `get_training_logs` | Tail the training log |
| `start_feature_extraction` | Run a backbone over patches, cache .pt tensors |

## Queue & jobs (4)

| Tool | Purpose |
|------|---------|
| `queue_task` | Submit a task to the serialized FIFO |
| `list_queue` | View queued / running / recent tasks |
| `get_job_status` | Poll any job by id |
| `wait_for_job` | Block until a job completes (bounded) |

## Evaluation (4)

| Tool | Purpose |
|------|---------|
| `start_evaluation` | Run inference + metrics on a holdout split |
| `get_eval_metrics` | Fetch AUROC / F1 / confusion matrix for a training run |
| `get_eval_plots` | Pull ROC / PR / calibration plots |
| `compare_experiments` | Side-by-side metric comparison across runs |

## Artifacts & inference (3)

| Tool | Purpose |
|------|---------|
| `list_artifacts` | Enumerate plots, heatmaps, checkpoints for a run |
| `generate_heatmap` | Attention overlay for a single slide |
| `start_lora_finetuning` | Adapter-based fine-tuning of a backbone |

## System (2)

| Tool | Purpose |
|------|---------|
| `system_status` | Backend + GPU + LLM health snapshot |
| `get_config` | Read user config (tokens redacted) |

## Code execution (1)

| Tool | Purpose |
|------|---------|
| `run_python` | Execute Python in the session workspace (sandboxed cwd) |

## Literature & papers (6)

| Tool | Purpose |
|------|---------|
| `search_literature` | OpenAlex / Crossref search |
| `deep_literature_review` | Multi-step review with citation walking |
| `get_paper_citations` | Fetch a paper's incoming / outgoing citations |
| `pubmed_search` | PubMed E-utilities search |
| `fetch_url` | Retrieve + clean a web page |
| `download_paper_pdf` | Save a PDF to the session folder |

## Folders & PDFs (2)

| Tool | Purpose |
|------|---------|
| `list_folders` | Attached PDF collections for the session |
| `read_pdf` | Extract text from a PDF in uploads/ or folders/ |

## Genomics (10)

| Tool | Purpose |
|------|---------|
| `parse_genomic_file` | MAF / VCF / MAF-lite parser |
| `query_mutations` | Per-gene / per-sample mutation filters |
| `compute_tmb` | Tumor mutational burden (SNV + indel count per Mb) |
| `extract_labels_from_genomic` | Build a labels CSV from MAF features |
| `query_cbioportal` | Pull clinical / genomic data from cBioPortal |
| `run_survival_analysis` | Kaplan-Meier with strata + log-rank |
| `build_multi_omic_labels` | Combine WSI predictions + genomic strata |
| `generate_oncoplot` | Top-N mutated genes heatmap |
| `parse_gene_expression` | RNA-seq matrix loader |
| `biomarker_discovery` | Differentially mutated / expressed features vs a label |

## Workspace & plugins (15)

| Tool | Purpose |
|------|---------|
| `list_workspace_files` | Tree listing of the session workspace |
| `read_workspace_file` | Read a file from workspace/ |
| `write_workspace_file` | Write or overwrite a workspace file |
| `delete_workspace_file` | Delete a workspace file |
| `clone_repo` | Clone from github/gitlab/huggingface/codeberg/bitbucket |
| `analyze_repo` | Summarize a cloned repo's structure and key modules |
| `list_plugins` | Registered plugins (built-in + user) |
| `register_plugin` | Add a user plugin entry |
| `update_plugin_config` | Override default_config for a plugin |
| `smoke_test_plugin` | Instantiate + forward-pass a plugin with synthetic tensors |
| `register_hf_backbone` | Register a custom HuggingFace model as a feature backbone |
| `list_backbones` | Available feature extraction backbones |
| `run_cellpose_segmentation` | Nuclear segmentation preprocessing pass |
| `implement_from_paper` | Workflow prompt for translating a method paper into a plugin |
| `make_plot` | Matplotlib figure from tabular data in the session |

## Interaction (1)

| Tool | Purpose |
|------|---------|
| `ask_user` | Request clarification mid-turn; pauses the agent for user input |

---

Total: **66 tools**. Invocation loop caps at 6 tool rounds per user turn.
