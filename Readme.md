# Diagnosis-Aware Multimodal Chest X-ray Report Generation: Effect of Filter Granularity on Clinical Factuality and Cross-Country Evaluation

Code for a Final Year Design Project (Department of CSE, Daffodil International University).

The project measures how the **granularity of temporal-reference filtering** in training reports
(removing whole sentences vs. only the time-related phrase) changes the clinical correctness of a
diagnosis-aware chest X-ray report generator, whether BLEU can detect that change, and where the
pipeline loses clinical information on X-rays from another country.

**Main result (1,227 MIMIC-CXR test images):** phrase-level filtering raises clinical efficacy (CE)
micro F1 from 0.1585 to 0.3967 (+0.2385, 95% bootstrap CI [+0.2164, +0.2616]); repeated with a
second seed and confirmed with CheXbert (0.289 → 0.455). BLEU-4 changes by only +0.0019.

---

## ⚠️ Data is NOT included

This repository contains **code only**. No image, report, label file, prediction file or checkpoint
from MIMIC-CXR is included, as required by the
[PhysioNet Credentialed Health Data Use Agreement](https://physionet.org/content/mimic-cxr-jpg/2.1.0/).
To run the code you need your own credentialed access to:

- [MIMIC-CXR-JPG v2.1.0](https://physionet.org/content/mimic-cxr-jpg/2.1.0/) (images + labels)
- [MIMIC-CXR v2.1.0](https://physionet.org/content/mimic-cxr/2.1.0/) (`mimic-cxr-reports.zip`)
- [IN-CXR](https://www.nirt.res.in/html/xray.html) (ICMR-NIRT, registration required) for the cross-country test

---

## System

```
X-ray ──► DenseNet121 classifier (13 findings) ──► thresholded findings ──► text prompt ─┐
   │                                                                                     ▼
   └────► DenseNet121 encoder ──► 49 visual tokens ─────────────────────► 3-layer Transformer decoder ──► report
```

- **Classifier:** DenseNet121 (ImageNet init), 320 px, trained once, mean test AUC 0.780.
- **Decoder:** 3-layer Transformer, 8 heads, 224 px input (7×7 = 49 visual tokens), prompt masked
  out of the loss, label smoothing 0.10, ~25.6 M parameters.
- **Hardware:** one NVIDIA RTX 3050 (8 GB), Windows, Python 3.11, PyTorch 2.5.1 + CUDA 12.1.

---

## Environment

```bash
conda create -n cxr python=3.11
conda activate cxr
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pandas numpy scikit-learn pillow tqdm nltk rouge-score pydicom matplotlib
```

All scripts are run from the data folder (e.g. `E:\mimic_data`). Paths at the top of each script
may need changing for your machine.

---

## Pipeline: scripts in run order

### 1. Data selection and download
| Script | What it does |
| --- | --- |
| `select_subset_final_version.py` | Chooses the label-balanced frontal subset (common findings capped at 10,000, normal at 15,000) and writes the download list |
| `check_distribution.py`, `check_pos_neg.py`, `check_full_availability.py`, `check_final_counts.py` | Count positives/negatives per finding in the full set and in the chosen subset |
| `make_missing.py` | Lists images not yet downloaded, for resuming `wget` |
| `build_available_subset.py` | Keeps only images that are fully on disk (checks the JPEG end marker) |
| `prepare_data.ipynb` | The same data steps in one notebook |

### 2. Report preparation and filtering
| Script | What it does |
| --- | --- |
| `prepare_reports.py` | Extracts FINDINGS and IMPRESSION from the report archive and pairs them with images (`image_report_pairs.csv`) |
| `clean_reports.py` | **Sentence-level filter:** deletes every sentence containing a temporal reference |
| `clean_reports_phrase.py` | **Phrase-level filter:** deletes only the temporal words or clause, keeps the finding |
| `check_temporal.py`, `check_length.py` | Count temporal references and report lengths before and after filtering |

### 3. Classifier and prompts
| Script | What it does |
| --- | --- |
| `cxr_data.py`, `cxr_data_224.py` | Dataset loaders for the classifier (320 px / 224 px) |
| `train_classifier.py` | Trains the DenseNet121 multi-label classifier (resumable) |
| `chest_xray_classifier.ipynb` | Notebook version with EDA and result plots |
| `calibrate.py` | Calibrates classifier probabilities (validation split only) |
| `predict_findings.py` | Writes per-image probabilities (`prompt_probs.csv`) |
| `tune_thresholds.py` | Chooses per-finding thresholds on validation for ≥50% precision (`prompt_thresholds.json`) |
| `check_prompts.py`, `compute_accuracy.py` | Prompt statistics and classifier precision/recall |

### 4. Report generator
| Script | What it does |
| --- | --- |
| `rg_data.py`, `rgp_data.py`, `gen_data.py` | Tokenisation and datasets for the decoder (without / with prompt) |
| `train_generator.py` | Trains one decoder run, selected by flags (see below) |
| `tune_decoding.py` | Decoding parameter sweep on validation (gain < 0.01; not adopted) |
| `gpt2_data.py`, `train_gpt2.py` | Experimental GPT-2 decoder (not in the main results) |
| `report_generation.ipynb`, `report_generation_prompted.ipynb` | Earlier notebook versions of the generator |

Example (seed replication of the sentence-level, predicted-prompt run):
```bash
python train_generator.py --prompts uncalibrated --reports filtered --img_size 224 --tag uncal224_s2 --seed 2
```
Run `python train_generator.py --help` for the exact flag values of the other runs.

| Run tag | Training text | Prompt |
| --- | --- | --- |
| `none224` | sentence-level | none |
| `uncal224` | sentence-level | predicted |
| `main224` | sentence-level | calibrated |
| `oracle224` | sentence-level | oracle (true labels) |
| `phrase224` ⭐ | phrase-level | predicted |
| `none_phrase` | phrase-level | none |
| `oracle_phrase` | phrase-level | oracle |
| `unfilt224` | unfiltered | predicted |
| `uncal224_s2`, `phrase224_s2` | as above | predicted, seed 2 |

### 5. Evaluation
| Script | What it does |
| --- | --- |
| `rg_metrics.py` | BLEU-1–4, ROUGE-L, METEOR |
| `clinical_eval.py` | Rule-based clinical efficacy (CE) over 11 findings |
| `compare_runs.py`, `extra_pairs.py`, `check_unfilt.py` | Paired bootstrap CIs between runs on the same test images |
| `chexbert_eval.py`, `show_chexbert.py` | Second labeller: CheXbert micro F1 over 14 observations |
| `bleu_parts.py`, `check_ref.py` | Why BLEU stays low (n-gram precision, brevity, word overlap) |
| `count_per_100.py`, `report_diversity.py`, `check_bias.py` | Findings per report, output diversity, length-controlled checks |
| `mimic_flag_rate.py` | Share of abnormal MIMIC test images the classifier flags (in-domain partner of the IN-CXR figure) |

### 6. Cross-country test (IN-CXR, India)
| Script | What it does |
| --- | --- |
| `inspect_dicom.py`, `convert_dicom.py` | Reads IN-CXR DICOM files and converts them the same way MIMIC-CXR-JPG was made |
| `make_external_manifest.py` | Builds the image list for an external dataset |
| `external_eval.py`, `incxr_eval.py` | Runs the trained system without fine-tuning and computes per-report measures |

### 7. Demo
| Script | What it does |
| --- | --- |
| `predict_one.py` | Prints predicted findings and a generated report for one image path |

---

## Not part of the final pipeline
Kept for the record, not needed to reproduce the results: `select_subset.py`, `select_subset_v2.py`,
`select_subset_v3.py`, `select_subset_final.py` (earlier versions), `gen_data_backup.py`,
`count_kermany.py`, `check_docx_images.py`, and the withdrawn human-evaluation scripts
`make_radiologist_package.py`, `rating_check.py`, `analyse_ratings.py`, `fix_key.py`.

---

## Limitations
Research prototype only, **not for clinical use** (report recall of findings 0.334). One architecture
and one training dataset. See the thesis for the full list.

## Citation
[Thesis / paper citation to be added at publication.]