# Data schema

The competition data are **not** distributed with this repository (see the README,
section *Dataset*). This file documents the layout the code expects so that the
pipeline can be run on the original files or on data with the same structure.

## Files

One CSV per panel, placed in a single directory (default `data/`). The mapping
panel → file name is set in `configs/*.yaml` (`data.panel_files`):

| Panel key | File name                  | Rows | Label = 1 | Label = 0 |
|-----------|----------------------------|-----:|----------:|----------:|
| `master`  | `YARISMA_TRAIN_MASTER.csv` | 2931 | 2149      | 782       |
| `kanser`  | `YARISMA_TRAIN_KANSER.csv` |  388 |  268      | 120       |
| `pah`     | `YARISMA_TRAIN_PAH.csv`    |  372 |  310      |  62       |
| `cftr`    | `YARISMA_TRAIN_CFTR.csv`   |  111 |   90      |  21       |

Counts refer to the competition release used during development (3802 rows,
3224 distinct `Variant_ID`s).

## Columns

| Column(s)            | Type        | Role |
|----------------------|-------------|------|
| `Variant_ID`         | string      | Variant identifier; used as the grouping key |
| `AL_1` … `AL_334`    | float       | Anonymised numeric annotations (≈55 % missing overall; 57 columns are constant) |
| `EK_1` … `EK_9`      | float       | Additional anonymised numeric annotations |
| `CAT_1` … `CAT_6`    | string      | Categorical annotations (population labels, genotypes). `CAT_6` is absent from the PAH file and is treated as missing |
| `AA_1`, `AA_2`       | string      | Reference / alternative amino acid (one-letter code). Missing or non-standard codes (e.g. `*`, multi-letter codes) map to zero descriptors |
| `Label`              | 0/1         | 1 = pathogenic, 0 = benign; rows without a label are dropped |

The loader adds a `panel` column from the file name. Every column that is not
reserved (`Variant_ID`, `Label`, `AA_1`, `AA_2`, `panel`) and does not start with
`CAT_` is treated as a numeric candidate; the numeric/categorical decision is made
inside each training fold.

## Properties relevant to evaluation

* **No gene, protein or transcript identifier** is present (names and values were
  scanned for gene symbols, HGNC, UniProt, Ensembl and RefSeq patterns).
* **Cross-panel duplicates.** 578 rows of the KANSER (246), PAH (255) and CFTR (77)
  files have a `Variant_ID` that also appears in MASTER. Labels and residue codes
  agree; the numeric annotations are slightly perturbed, so the duplicates are not
  exact copies. Grouping on `Variant_ID` keeps each pair on one side of a split.
* **Rows without numeric information.** 137 rows have every numeric annotation
  missing and conflicting labels among identical profiles. They are removed from
  *training* partitions only (detected within each partition), never from
  evaluation folds.

## Inference input

`scripts/predict.py` and the demo require every numeric column seen during
training plus `AA_1` and `AA_2`. Categorical columns may be absent (treated as
missing). Missing numeric columns are rejected rather than filled.

## Synthetic example

`synthetic_sample.csv` has the full column layout (334 `AL_*`, 9 `EK_*`,
6 `CAT_*`) with 12 artificial rows produced by `varpath.synthetic`. The values are
random draws with a planted signal; they are not derived from the competition data
or from any patient record. Larger synthetic panel files can be written with

```bash
python scripts/make_synthetic_data.py --out-dir data/synthetic --n-al 334 --n-ek 9
```
