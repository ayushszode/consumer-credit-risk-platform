# Data

No loan records are stored in this repository. To reproduce the results you need to download the
data yourself.

## Source

- **Dataset:** "All Lending Club loan data" on Kaggle (uploaded by Nathan George).
- **File used:** `accepted_2007_to_2018Q4.csv.gz` (accepted loans, 2007 to end of 2018).
- **What it is:** a third-party copy of loan data that LendingClub used to publish. It is not an
  official LendingClub upload, and LendingClub has not reviewed or endorsed this project.
- **Licence / terms:** check the licence shown on the Kaggle page before using or sharing anything
  derived from the data. Record what you found, and the download date, below.

| Item | Value |
| --- | --- |
| Downloaded on | _fill in_ |
| Kaggle dataset version | _fill in_ |
| Licence shown on Kaggle | _fill in_ |
| File size (compressed) | _fill in_ |

## Where to put it

```
data/raw/accepted_2007_to_2018Q4.csv.gz
```

Leave it compressed if you can; the pipeline reads the `.gz` file directly. If your download came
unzipped (`accepted_2007_to_2018Q4.csv`), or inside a folder with that name, that works too: the
pipeline looks for any file starting with `accepted` in `data/raw/` and one folder below it.

## What the pipeline keeps

36-month loans issued January 2012 to December 2015 that ended as Fully Paid, Charged Off or Default.
Every rule and how many loans it removes is written to `reports/results/population_audit.csv`.

## No download? Use synthetic data

```
python -m credit_risk.pipeline --synthetic
```

This generates fake loans in the same format (see `src/credit_risk/synthetic.py`). It is what the
tests and the GitHub Actions checks use.
