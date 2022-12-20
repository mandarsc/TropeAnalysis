# TropeAnalysis

## Prerequisites
1. Make sure you have `python3` installed on your machine before running the experiments.
2. Update the `DATA_DIR`, `HOME_DIR` `OUT_DIR` variables in `utils/utils.py` with the local path of your machine.
3. Install python packages listed in the `requirements.txt` file using the following command,
    * ```pip install -r requirements.txt```

## Dataset
Please contact the authors at manchaudhary@ebay.com to get access to the datasets.

## Trope Prediction
In this work, we perform trope prediction using two feature extraction methods. Both the approaches are implemented in `trope_prediction.py`. The first approach extracts features from movie scripts using Bag-of-Words (BoW) model implemented with the term-frequency inverse-document frequency method, and the second approach uses doc2vec embedding algorithm for representing movie scripts as embedding vectors.

Run trope prediction pipeline for top-5 tropes frequently occuring tropes namely, `ShoutOut`, `OhCrap`, `ChekhovsGun`, `Foreshadowing`, `BittersweetEnding` using the following python script.
   * ```python3 trope_prediction.py```

The output from this script are two comma-separated (csv) files which contain several metrics measured using 2-fold stratified cross-validation. The results from TF-IDF feature extraction method are stored in `Tfidf_Results_{trope}.csv` where `trope` is one of the five tropes listed above. Similarly, results from doc2vec method are stored in `Doc2Vec_Results_{trope}.csv`.
