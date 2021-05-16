from collections import Counter, defaultdict
from datetime import datetime
import logging
import re
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from data_prep.load_movie_script_data import get_clean_timos_tropes, preprocess_movie_script_data
from models.train_classifier import train_eval_classifiers
from utils.utils import DATA_DIR, configure_logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def trope_detection_tfidf(movie_script_dialog: Dict[str, str], movie_tropes_dict: Dict[str, List[str]], tfidf_vocab_size: int, d2v_size: int, trope_of_interest: List[str], train_tfidf: bool = True, train_d2v: bool = False):

	tfidf_rf_metrics = defaultdict(str)
	tfidf_xgb_metrics = defaultdict(str)

	d2v_rf_metrics = defaultdict(str)
	d2v_xgb_metrics = defaultdict(str)

	d2v_rf_metrics['AP'] = []
	d2v_xgb_metrics['AUC'] = []

	d2v_rf_metrics['AP'] = []
	d2v_xgb_metrics['AUC'] = []

	tfidf_xgb_metrics['AP'] = []
	tfidf_xgb_metrics['AUC'] = []

	tfidf_rf_metrics['AP'] = []
	tfidf_rf_metrics['AUC'] = []

	rf_estimators = 200
	xgb_estimators = 200

	for trope in trope_of_interest:
		multi_label = MultiLabelBinarizer()
		y = multi_label.fit_transform([set(movie_tropes) for movie_tropes in movie_tropes_dict.values()])
		trope_idx = trope_idx = list(multi_label.classes_).index(trope)

		if y[:, trope_idx].sum() < 2:
			logger.info(f"Skipping trope {trope} with less than 2 instances")
			continue

		if train_tfidf:
			X_train_docs, X_test_docs, y_train, y_test = train_test_split(list(movie_script_dialog.values()), y[:, trope_idx], train_size=0.8, stratify=y[:, trope_idx])

			logger.info(f"{trope} distribution: train ({np.bincount(y_train)}), test ({np.bincount(y_test)})")

			X_train_tfidf_docs = [' '.join(list(x)) for x in X_train_docs]
			X_test_tfidf_docs = [' '.join(list(x)) for x in X_test_docs]

			tfidf_fit = TfidfVectorizer(max_features=tfidf_vocab_size, ngram_range=(1, 2), stop_words='english').fit(raw_documents=X_train_tfidf_docs)
			X_tfidf_train_vec = tfidf_fit.transform(X_train_tfidf_docs)
			X_tfidf_test_vec = tfidf_fit.transform(X_test_tfidf_docs)

			rf_auc, rf_ap = train_eval_classifiers(X_tfidf_train_vec, X_tfidf_test_vec, y_train, y_test, classifier='rf', n_estimators = rf_estimators)
			xgb_auc, xgb_ap = train_eval_classifiers(X_tfidf_train_vec, X_tfidf_test_vec, y_train, y_test, n_estimators = xgb_estimators)

			tfidf_rf_metrics['AUC'].append(rf_auc)
			tfidf_rf_metrics['AP'].append(rf_ap)
			tfidf_xgb_metrics['AUC'].append(xgb_auc)
			tfidf_xgb_metrics['AP'].append(xgb_ap)

		if train_d2v:
			documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(movie_script_dialog.values())]
			X_train_docs, X_test_docs, y_train, y_test = train_test_split(documents, y[:, trope_idx], train_size=0.8, stratify=y[:, trope_idx])

			dm_model = Doc2Vec(X_train_docs, vector_size=d2v_size, min_count=5, dm=1)
			dbow_model = Doc2Vec(X_train_docs, vector_size=d2v_size, min_count=5, dm=0)

			X_train_dm_dv = []
			X_train_dbow_dv = []

			X_test_dm_dv = []
			X_test_dbow_dv = []

			for i in range(len(X_train_docs)):
				X_train_dm_dv.append(dm_model.docvecs[i])
				X_train_dbow_dv.append(dbow_model.docvecs[i])

			for i in range(len(X_test_docs)):
				X_test_dm_dv.append(dm_model.infer_vector(X_test_docs[i][0]))
				X_test_dbow_dv.append(dbow_model.infer_vector(X_test_docs[i][0]))

			X_train_dv = pd.concat([pd.DataFrame(X_train_dm_dv), pd.DataFrame(X_train_dbow_dv)], axis=1)
			X_test_dv = pd.concat([pd.DataFrame(X_test_dm_dv), pd.DataFrame(X_test_dbow_dv)], axis=1)

			X_train_dv.columns = np.arange(X_train_dv.shape[1])
			X_test_dv.columns = np.arange(X_test_dv.shape[1])

			rf_auc, rf_ap = train_eval_classifiers(X_train_dv, X_test_dv, y_train, y_test, classifier='rf')
			xgb_auc, xgb_ap = train_eval_classifiers(X_train_dv, X_test_dv, y_train, y_test)

			d2v_rf_metrics['AUC'].append(rf_auc)
			d2v_rf_metrics['AP'].append(rf_ap)
			d2v_xgb_metrics['AUC'].append(xgb_auc)
			d2v_xgb_metrics['AP'].append(xgb_ap)

	return tfidf_rf_metrics, tfidf_xgb_metrics, d2v_rf_metrics, d2v_xgb_metrics

if __name__ == "__main__":
	configure_logging(logger)
	start_time = datetime.now()

	train_tfidf = True
	train_d2v = False

	logger.info("Preprocessing movie scripts data")
	movie_script_dialog, movie_tropes_dict = preprocess_movie_script_data()

	unique_tropes_set = list()
	for tropes in movie_tropes_dict.values():
	    unique_tropes_set += list(set(tropes))
	    
	# Get movie count per trope
	tropes_count_dict = Counter(unique_tropes_set)

	tfidf_vocab_size = 50
	d2v_dim = 50

	logger.info("Get all TiMoS tropes")
	tropes_of_interest = get_clean_timos_tropes(tropes_count_dict)

	logger.info(f"Running trope detection...")
	tfidf_rf_metrics, tfidf_xgb_metrics, d2v_rf_metrics, d2v_xgb_metrics = trope_detection_tfidf(movie_script_dialog, movie_tropes_dict, tfidf_vocab_size, 
		d2v_dim, tropes_of_interest, train_tfidf, train_d2v)
	logger.info(f"Completed training and evaluating models. Time taken: {(datetime.now() - start_time).seconds}")

	if train_tfidf:
		logger.info("Results from Tf-Idf features")
		logger.info("Random Forest")
		logger.info(f"Mean AUC: {round(np.mean(tfidf_rf_metrics['AUC']), 2)} ({round(np.std(tfidf_rf_metrics['AUC']), 2)})")
		logger.info(f"Mean AP: {round(np.mean(tfidf_rf_metrics['AP']), 2)} ({round(np.std(tfidf_rf_metrics['AP']), 2)})")

		logger.info("Xgboost")
		logger.info(f"Mean AUC: {round(np.mean(tfidf_rf_metrics['AUC']), 2)} ({round(np.std(tfidf_rf_metrics['AUC']), 2)})")
		logger.info(f"Mean AP: {round(np.mean(tfidf_rf_metrics['AP']), 2)} ({round(np.std(tfidf_rf_metrics['AP']), 2)})")

	if train_d2v:
		logger.info("Results from Doc2Vec features")
		logger.info("Random Forest")
		logger.info(f"Mean AUC: {round(np.mean(d2v_rf_metrics['AUC']), 2)} ({round(np.std(d2v_rf_metrics['AUC']), 2)})")
		logger.info(f"Mean AP: {round(np.mean(d2v_rf_metrics['AP']), 2)} ({round(np.std(d2v_rf_metrics['AP']), 2)})")

		logger.info("Xgboost")
		logger.info(f"Mean AUC: {round(np.mean(d2v_xgb_metrics['AUC']), 2)} ({round(np.std(d2v_xgb_metrics['AUC']), 2)})")
		logger.info(f"Mean AP: {round(np.mean(d2v_xgb_metrics['AP']), 2)} ({round(np.std(d2v_xgb_metrics['AP']), 2)})")
