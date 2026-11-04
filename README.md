# Med-SA score calibration study

The present work compares the native quality scores produced by the original SAM model and by a released Medical SAM Adapter (Med-SA) checkpoint to the actual IoU of the optic-disc masks generated on REFUGE Test-400.

Native-score reliability has been studied so far. No calibration method has yet been fitted or evaluated.

## Research question

Does a score correction fitted at one click count (one, two, or three clicks) also work at the other click counts? Does this differ between SAM and Med-SA?

We expect a correction to work best at the click count it was fitted on, and to work worse the further the target click count is from it. For example, a one-click correction should transfer better to two clicks than to three. We keep the segmentation checkpoints fixed. We do not train a segmentation model or reproduce Med-SA's training or full evaluation.

## Project status

| Part | Status |
| --- | --- |
| SAM and Med-SA predictions on REFUGE Test-400 | Complete |
| Native-score reliability analysis | Complete |
| Calibration-fit, validation, and evaluation roles | Implemented |
| Calibration fitting and transfer analysis | Not implemented |
| Drishti-GS1 inference | Implemented, not run (data must be added by hand) |
| Independent final evaluation | Not available |

The REFUGE results remain exploratory. Med-SA reports REFUGE2 and the upstream REFUGE code can use Test-400 to select a checkpoint. No case-level record of the data used to train or select the released adapter is available so exposure to these cases cannot be confirmed. Evaluation cases are kept separate from calibration fitting and validation yet independence from checkpoint development cannot be verified. They are labelled `internal_evaluation_overlap_risk`.

## Experiment setup

Two models are compared:

- `base_sam`: the official SAM ViT-B checkpoint.
- `med_sa_adapter`: the same base checkpoint together with the released Med-SA optic-disc adapter.

Every REFUGE case supplies an image plus seven optic-disc rater masks. A pixel is treated as foreground in the consensus mask when at least four raters mark it as foreground. Each image is resized so that its longer side equals 1024 pixels. Mask token 0 is used together with its matching predicted-IoU score.

Prompts contain one, two, or three positive clicks. Each larger prompt includes the smaller ones so the clicks are added sequentially. The clicks are simulated from the consensus mask. The first click lands on the foreground pixel farthest from the boundary. Each subsequent click lands on a remaining foreground pixel farthest from the earlier clicks. These are simulated prompts rather than clicks recorded from clinicians.

Cases are assigned to roles by case ID with fixed random seeds:

| Role | Cases | Purpose |
| --- | ---: | --- |
| `calibration_fit` | 200 | Fit score corrections |
| `validation` | 100 | Choose methods or settings |
| `internal_evaluation_overlap_risk` | 100 | Current REFUGE evaluation |

The native-score analysis uses only the evaluation cases. It neither fits nor selects a method.

Drishti-GS1 was also considered as a possible external evaluation set. Its 50-case training pool could supply 30 cases for calibration fitting and 20 for validation. Its separate 51-case test pool could serve for final evaluation yet obtaining the labels requires registration. The data are absent from this repository and cannot be downloaded automatically. The runner reads Drishti from `data/drishti_gs1/` together with a `case_manifest.csv` (columns `case_id`, `source_pool`, `image_path`, `optic_disc_mask`) but the run has not been performed yet. Exposure of the released adapter to these cases is likewise unknown so they cannot be treated as an unseen dataset.

Shared settings live in [`experiment_protocol.py`](experiment_protocol.py). Saved predictions are reused for a case only when the settings, inputs, checkpoints, and code all match so a change of setting forces inference to be rerun for the affected cases.

| Setting | Value |
| --- | --- |
| Sampling, split, model, and bootstrap seeds | `1` |
| Bootstrap replicates | `2000` |
| Image size | 1024 pixels on the longer side |
| Click counts | 1, 2, and 3 positive clicks |
| Ground truth | At least 4 of 7 raters |
| Mask and score | Token 0 and its matching predicted-IoU score |

## Current native-score results

For every case the clipped model score \(q_i\) is compared against the observed mask IoU \(y_i\):

- **MAE** equals the average absolute difference between the score and the observed IoU.
- **RMSE** equals the square root of the average squared difference.
- **Signed bias** equals the average of \(q_i-y_i\). A positive value means the score exceeds the observed IoU.
- **Coarse ECE** equals the weighted average absolute gap between the score and observed IoU means across ten equal-width bins between 0 and 1.

MAE and signed bias are accompanied by percentile 95% confidence intervals obtained from 2,000 case-bootstrap samples. Confidence intervals for paired differences between the models are not reported. The table below lists native-score results for the 100 evaluation cases stored in [`runs/refuge2/native_summary.json`](runs/refuge2/native_summary.json). These scores have not been calibrated.

| Model | Clicks | MAE (95% CI) | Signed bias (95% CI) | Coarse ECE |
| --- | ---: | ---: | ---: | ---: |
| Original SAM | 1 | 0.379 (0.335–0.423) | +0.353 (+0.300–+0.404) | 0.357 |
| Original SAM | 2 | 0.345 (0.299–0.395) | +0.298 (+0.237–+0.359) | 0.300 |
| Original SAM | 3 | 0.313 (0.255–0.377) | +0.252 (+0.183–+0.325) | 0.256 |
| Med-SA adapter | 1 | 0.246 (0.205–0.291) | +0.200 (+0.149–+0.253) | 0.208 |
| Med-SA adapter | 2 | 0.142 (0.112–0.175) | +0.105 (+0.070–+0.143) | 0.113 |
| Med-SA adapter | 3 | 0.089 (0.068–0.114) | +0.028 (−0.001–+0.059) | 0.056 |

In the present run both models showed declines in MAE, bias, and ECE as more clicks were added and the decline was larger for Med-SA. These figures do not reveal whether calibration transfers across click settings or how the models would perform on independent data.

Inference was executed on an RTX 5080 GPU. The saved summary originates from a later rerun that reused all 400 cached predictions and therefore captures neither GPU use nor inference time. The inference metadata stores the checkpoint hashes, the dataset archive hash, and the inference-script hash.

## Running the experiment

Inference requires CUDA. The setup that was tested used Python 3.12 together with PyTorch 2.11.0 and CUDA 12.8 wheels. [`requirements.txt`](requirements.txt) does not pin every dependency. Pillow and certain transitive dependencies remain unpinned so the installed package versions should be recorded when the experiment is run. The NVIDIA driver must support the installed PyTorch CUDA runtime.

Create the environment and verify that PyTorch can access the GPU:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

Clone the repository with the upstream Medical-SAM-Adapter submodule or initialize the submodule in an existing clone:

```bash
git submodule update --init --recursive
```

We used submodule revision `5888e722876511f177e8e762498a7986753b7f7d`.

The dataset and the checkpoints are not included in this repository. On startup the runner downloads any missing public files into `data/` which Git ignores:

| File | Saved to | Source | SHA-256 |
| --- | --- | --- | --- |
| `sam_vit_b_01ec64.pth` | `data/checkpoints/` | [SAM ViT-B checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) | `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912` |
| `OpticDisc_Fundus_SAM_1024.pth` | `data/checkpoints/` | [Med-SA adapter](https://huggingface.co/KidsWithTokens/Medical-Adapter-Zoo/resolve/main/OpticDisc_Fundus_SAM_1024.pth) | `a1e39b25aa17e71fe18a6776bd771fc862b7d28c7562e574646473effdff1af1` |
| `REFUGE-Multirater.zip` | `data/downloads/`, unpacked to `data/refuge2/` | [REFUGE data](https://huggingface.co/datasets/realslimman/REFUGE-MultiRater/resolve/main/REFUGE-Multirater.zip) | Recorded in the run metadata |

Drishti-GS1 requires a registered download so its files and `case_manifest.csv` must be placed by hand in `data/drishti_gs1/`.

The REFUGE data should contain one `Test-400` folder with 400 cases. Each case needs a `.jpg` image and seven masks named `<case_id>_seg_disc_<rater>.png`. The runner checks the number of cases and the checkpoint hashes against the protocol.

Run one dataset at a time. Each dataset writes to `runs/<dataset_id>/`:

```bash
.venv/bin/python run_experiment.py --dataset-id refuge2
.venv/bin/python run_experiment.py --dataset-id drishti_gs1
```

Running the same command again reuses every saved prediction that still matches and runs inference only for missing or changed cases. It always rebuilds the analysis files from the saved predictions. It recalculates observed IoU from the saved masks and consensus masks and does not fit or apply calibration.

## Code and output files

| File | Saved to | Source | SHA-256 |
| --- | --- | --- | --- |
| `sam_vit_b_01ec64.pth` | `data/checkpoints/` | [SAM ViT-B checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) | `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912` |
| `OpticDisc_Fundus_SAM_1024.pth` | `data/checkpoints/` | [Med-SA adapter](https://huggingface.co/KidsWithTokens/Medical-Adapter-Zoo/resolve/main/OpticDisc_Fundus_SAM_1024.pth) | `a1e39b25aa17e71fe18a6776bd771fc862b7d28c7562e574646473effdff1af1` |
| `REFUGE-Multirater.zip` | `data/downloads/`, unpacked to `data/refuge2/` | [REFUGE data](https://huggingface.co/datasets/realslimman/REFUGE-MultiRater/resolve/main/REFUGE-Multirater.zip) | Recorded in the run metadata |

Each inference run contains:

- `summary.json` with the cases, roles, seeds, GPU, and run times.
- `metadata.json` with the protocol, checkpoint hashes, and exposure note.
- `predictions/<case_id>.npz` with the consensus mask, clicks, model masks and scores, and case details.

The analysis produces `shared_case_results.csv`, `native_case_results.csv`, `native_metrics.csv`, `native_reliability.png`, and `native_summary.json`. Keep the analysis and inference outputs together. The results shown in the table above come from the saved `native_summary.json`.

The metadata records the inference-script hash yet does not record the full Git working tree or a complete package lockfile. When results are shared the Git commit, any local changes, and the installed package versions should also be recorded. This project does not train a model so no training time needs to be reported.

## Planned calibration analysis

We have not implemented calibration yet. The planned method fits a straight-line correction and clips the result to the range from 0 to 1:

\[
\hat{y} = \operatorname{clip}(a q + b, 0, 1),
\]

where \(q\) is the model's predicted-IoU score and \(y\) is the observed mask IoU.

For each model we will fit one correction on the `calibration_fit` cases at each click count (one, two, and three clicks). Then we will apply each correction to all three click counts which gives a 3 × 3 table of fit and target click counts. We will compare the results with the uncorrected scores and with a baseline that always predicts the calibration-set mean.

We will use the `validation` cases only to choose methods or settings. We will not use evaluation cases to fit or choose a method. Before drawing conclusions about the research question we still need to implement the calibration analysis and compare paired results. We will report uncertainty for those comparisons when it helps answer the question.

## Sources

- Junde Wu et al., “Medical SAM adapter: Adapting segment anything model for medical image segmentation.” *Medical Image Analysis*, 102, 103547 (2025). [Publisher record](https://doi.org/10.1016/j.media.2025.103547).
- Kirillov et al., “Segment Anything.” [arXiv:2304.02643](https://arxiv.org/abs/2304.02643).
- [Project GitHub repository](https://github.com/Rumle/atdl-a3-medical-sam-calibration).
