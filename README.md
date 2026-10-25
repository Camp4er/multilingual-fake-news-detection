# Multilingual Fake News Detection using Transformers (English + Hindi)

Run everything on your own laptop (Windows PowerShell commands shown; on Mac/Linux use
`source venv/bin/activate` and `/` in paths).

## 0. Setup
```
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
nvidia-smi          # shows an NVIDIA GPU if you have one; an error means CPU-only
```

## 1. Download data
English (ISOT), run in Python once and note the printed path:
```
import kagglehub
print(kagglehub.dataset_download("rahulogoel/isot-fake-news-dataset"))
```
Hindi (CONSTRAINT-2021), PowerShell:
```
mkdir data\hindi_raw
$u="https://raw.githubusercontent.com/mohit19014/Hindi-Hostility-Detection-CONSTRAINT-2021/main/Dataset"
foreach($f in "train","valid","test"){ Invoke-WebRequest "$u/$f.csv" -OutFile "data\hindi_raw\$f.csv" }
```

## 2. Build and clean the dataset
```
python build_dataset.py --isot_dir "<path printed above>" --hindi_dir data\hindi_raw --out data
python data_prep.py --inputs data\en.csv data\hi.csv --out data\clean
```

## 3. Classical ML baselines
```
python baseline_ml.py --data data\clean --out results
```

## 4. Transformer (dry run first, then real run)
```
python train_transformer.py --data data\clean --out results\dry --epochs 1 --max_len 128 --batch 8 --max_train_samples 200 --max_eval_samples 100
python train_transformer.py --data data\clean --out results\xlmr --epochs 3
```
CPU-only laptop: add `--max_len 128 --batch 8 --max_train_samples 4000 --epochs 2`.

## 5. Run the app locally
```
$env:MODEL_ID="results\xlmr\final"
python app.py
```
Open http://127.0.0.1:7860

## 6. Deploy
```
hf auth login
hf upload YOUR-USERNAME/fakenews-xlmr results\xlmr\final .
```
Create a Gradio Space on huggingface.co, upload `app.py` and `space/requirements.txt`
(as `requirements.txt`), and set the Space variable `MODEL_ID` = `YOUR-USERNAME/fakenews-xlmr`.

Data licence: the CONSTRAINT data is for non-commercial academic research and must not be
redistributed, so do not upload the CSVs to a public repo or Space.
