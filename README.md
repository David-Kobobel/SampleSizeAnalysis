# SampleSizeAnalysis

Code and data to reproduce the results in Kobobel, Degregori et al., *Beyond the headcount: sparse reporting of sequencing and quality-control methods in cancer microbiome studies* (Access Microbiology).

## Setup

Python 3.12.

```bash
pip install -r requirements.txt
```

## Reproduce the results

Run from the repository root. Outputs are written to `results/`.

| Command | Reproduces |
|---|---|
| `python scripts/storms_results.py` | Table 1 and the statistics reported in the Results |
| `python scripts/figure1.py` | Figure 1 |
| `python scripts/figure_s3.py` | Figure S3 and its tests |
| `python scripts/figure_s2.py path/to/Data_S1.xlsx` | Figure S2 and its tests |

Figure S2 uses the 2,305-study dataset, which is published with the article as Data S1.

## Data

- `data/storms_items.csv` lists the 67 audited studies with the ten STORMS items (1 = reported, 0 = not reported), publication year, journal, 2024 journal impact factor, sequencing type and specimen type (`low_biomass`: 1 = tissue, fluid or swab, 0 = includes stool). Item definitions are given in the Methods.
- `data/prisma_corpus.csv` lists the 1,938 records of the PRISMA-guided query used for comparison. `audited = 1` marks the 43 audited studies that appear in it.
