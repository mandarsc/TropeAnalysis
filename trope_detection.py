from collections import Counter, defaultdict
from datetime import datetime
import logging
from os.path import join
import re
from typing import DataFrame, Dict, List, Tuple

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import preprocess_string, remove_stopwords, stem_text, strip_punctuation, strip_short
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer

from data_prep.load_movie_script_data import get_clean_timos_tropes, preprocess_movie_script_data
from models.train_classifier import train_eval_classifiers
from utils.utils import DATA_DIR, OUT_DIR, configure_logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def extract_tfidf_features(X_train_docs: List[List[str]], X_test_docs: List[List[str]], tfidf_vocab_size: int, ngram_range: int) -> Tuple[np.array, np.array]:
	"""
	"""

	X_train_tfidf_docs = [' '.join(list(x)) for x in X_train_docs]
	X_test_tfidf_docs = [' '.join(list(x)) for x in X_test_docs]

	tfidf_fit = TfidfVectorizer(max_features=tfidf_vocab_size, ngram_range=(1, ngram_range), stop_words='english').fit(raw_documents=X_train_tfidf_docs)
	X_train_tfidf = tfidf_fit.transform(X_train_tfidf_docs)
	X_test_tfidf = tfidf_fit.transform(X_test_tfidf_docs)

	return X_train_tfidf, X_test_tfidf


def extract_doc2vec_features(X_train_docs: List[TaggedDocument], X_test_docs: List[TaggedDocument], d2v_size: int) -> Dict[str, pd.DataFrame]:
	"""
	"""
	dm_model = Doc2Vec(X_train_docs, vector_size=d2v_size, min_count=5, dm=1, dm_mean=1, dm_concat=0)
	dm_concat_model = Doc2Vec(X_train_docs, vector_size=d2v_size, min_count=5, dm=1, dm_mean=1, dm_concat=1)
	dbow_model = Doc2Vec(X_train_docs, vector_size=d2v_size, min_count=5, dm=0, dbow_words=0)

	X_train_dm = []
	X_train_dm_concat = []
	X_train_dbow = []

	X_test_dm = []
	X_test_dm_concat = []
	X_test_dbow = []

	for i in range(len(X_train_docs)):
		X_train_dm.append(dm_model.docvecs[i])
		X_train_dm_concat.append(dm_concat_model[i])
		X_train_dbow.append(dbow_model.docvecs[i])

	for i in range(len(X_test_docs)):
		X_test_dm.append(dm_model.infer_vector(X_test_docs[i][0]))
		X_test_dm_concat.append(dm_concat_model.infer_vector(X_test_docs[i][0]))
		X_test_dbow.append(dbow_model.infer_vector(X_test_docs[i][0]))

	X_train_dm_df = pd.DataFrame(X_train_dm, columns=np.arange(d2v_size))
	X_train_dm_concat_df = pd.DataFrame(X_train_dm_concat, columns=np.arange(d2v_size))
	X_train_dbow_df = pd.DataFrame(X_train_dbow, columns=np.arange(d2v_size))

	X_test_dm_df = pd.DataFrame(X_test_dm, columns=np.arange(d2v_size))
	X_test_dm_concat_df = pd.DataFrame(X_test_dm_concat, columns=np.arange(d2v_size))
	X_test_dbow_df = pd.DataFrame(X_test_dbow, columns=np.arange(d2v_size))

	return {
	"d2v_dm_train": X_train_dm_df, "d2v_dm_concat_train": X_train_dm_concat_df, "d2v_dbow_train": X_train_dbow_df,
	"d2v_dm_test": X_test_dm_df, "d2v_dm_concat_test": X_test_dm_concat_df, "d2v_dbow_test": X_test_dbow_df
	}


def trope_detection(movie_script_dialog: Dict[str, str], movie_tropes_dict: Dict[str, List[str]], tfidf_vocab_size: int, d2v_size: int, trope: str, train_tfidf: bool = True, train_d2v: bool = False):

	tfidf_rf_metrics = defaultdict(int)
	tfidf_xgb_metrics = defaultdict(int)

	d2v_dm_rf_metrics = defaultdict(int)
	d2v_dm_concat_rf_metrics = defaultdict(int)
	d2v_dbow_rf_metrics = defaultdict(int)

	d2v_dm_xgb_metrics = defaultdict(int)
	d2v_dm_concat_xgb_metrics = defaultdict(int)
	d2v_dbow_xgb_metrics = defaultdict(int)

	rf_estimators = 200
	xgb_estimators = 200

	multi_label = MultiLabelBinarizer()
	y = multi_label.fit_transform([set(movie_tropes) for movie_tropes in movie_tropes_dict.values()])
	trope_idx = list(multi_label.classes_).index(trope)

	num_cv_folds = 2

	if y[:, trope_idx].sum() < 2:
		logger.info(f"Skipping trope {trope} with less than 2 instances")
	else:
		random_state = np.random.randint(10000, size=num_cv_folds)

		for i in range(num_cv_folds):
			if train_tfidf:
				X_train_docs, X_test_docs, y_train, y_test = train_test_split(list(movie_script_dialog.values()), y[:, trope_idx], shuffle=True, train_size=0.8, stratify=y[:, trope_idx], random_state=random_state[i])

				logger.info(f"Iteration: {i+1}) {trope} distribution: train ({np.bincount(y_train)}), test ({np.bincount(y_test)})")

				X_train_tfidf, X_test_tfidf = extract_tfidf_features(X_train_docs, X_test_docs, tfidf_vocab_size, ngram_range=2)

				tfidf_rf_metrics_dict = train_eval_classifiers(X_train_tfidf, X_test_tfidf, y_train, y_test, classifier='rf', n_estimators = rf_estimators)
				tfidf_xgb_metrics_dict = train_eval_classifiers(X_train_tfidf, X_test_tfidf, y_train, y_test, n_estimators = xgb_estimators)
	
				tfidf_rf_metrics[i] = tfidf_rf_metrics_dict
				tfidf_xgb_metrics[i] = tfidf_xgb_metrics_dict

			if train_d2v:
				documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(movie_script_dialog.values())]
				X_train_docs, X_test_docs, y_train, y_test = train_test_split(documents, y[:, trope_idx], shuffle=True, train_size=0.8, stratify=y[:, trope_idx], random_state=random_state[i])

				X_d2v_docs = extract_doc2vec_features(X_train_docs, X_test_docs, d2v_size)

				d2v_dm_rf_metrics_dict = train_eval_classifiers(X_d2v_docs["d2v_dm_train"], X_d2v_docs["d2v_dm_test"], y_train, y_test, classifier='rf')
				d2v_dm_concat_rf_metrics_dict = train_eval_classifiers(X_d2v_docs["d2v_dm_concat_train"], X_d2v_docs["d2v_dm_concat_test"], y_train, y_test, classifier='rf')
				d2v_dbow_rf_metrics_dict = train_eval_classifiers(X_d2v_docs["d2v_dbow_train"], X_d2v_docs["d2v_dbow_test"], y_train, y_test, classifier='rf')
				
				d2v_dm_xgb_metrics_dict = train_eval_classifiers(X_d2v_docs["d2v_dm_train"], X_d2v_docs["d2v_dm_test"], y_train, y_test)
				d2v_dm_concat_xgb_metrics_dict = train_eval_classifiers(X_d2v_docs["d2v_dm_concat_train"], X_d2v_docs["d2v_dm_concat_test"], y_train, y_test)
				d2v_dbow_xgb_metrics_dict = train_eval_classifiers(X_d2v_docs["d2v_dbow_train"], X_d2v_docs["d2v_dbow_test"], y_train, y_test)

				d2v_dm_rf_metrics[i] = d2v_dm_rf_metrics_dict
				d2v_dm_concat_rf_metrics[i] = d2v_dm_rf_metrics_dict
				d2v_dbow_rf_metrics[i] = d2v_dbow_rf_metrics_dict

				d2v_dm_xgb_metrics[i] = d2v_dm_xgb_metrics_dict
				d2v_dm_concat_xgb_metrics[i] = d2v_dm_xgb_metrics_dict
				d2v_dbow_xgb_metrics[i] = d2v_dbow_xgb_metrics_dict

	return {
			"tfidf-rf": tfidf_rf_metrics, "tfidf-xgb": tfidf_xgb_metrics, 
			"d2v-dm-rf": d2v_dm_rf_metrics, "d2v-dm-concat-rf": d2v_dm_concat_rf_metrics, "d2v-dbow-rf": d2v_dbow_rf_metrics, 
			"d2v-dm-xgb": d2v_dm_xgb_metrics, "d2v-dm-concat-xgb": d2v_dm_concat_xgb_metrics, "d2v-dbow-xgb": d2v_dbow_xgb_metrics
			}


if __name__ == "__main__":
	configure_logging(logger)
	start_time = datetime.now()

	train_tfidf = True
	train_d2v = True

	logger.info("Preprocessing movie scripts data")
	movie_script_dialog, movie_tropes_dict = preprocess_movie_script_data()

	unique_tropes_set = list()
	for tropes in movie_tropes_dict.values():
	    unique_tropes_set += list(set(tropes))
	    
	# Get movie count per trope
	tropes_count_dict = Counter(unique_tropes_set)

	tfidf_vocab_size = 200
	d2v_dim = 100

	tropes_of_interest = ['ShoutOut', 'OhCrap', 'ChekhovsGun', 'Foreshadowing', 'BittersweetEnding']
	logger.info(f"Using top-{len(tropes_of_interest)} frequently occuring tropes")

	logger.info(f"Running trope detection...")
	for trope in tropes_of_interest:
		metrics_dict = trope_detection(movie_script_dialog, movie_tropes_dict, tfidf_vocab_size, d2v_dim, trope, train_tfidf, train_d2v)

		if train_tfidf:
			tfidf_rf_metrics = pd.DataFrame(metrics_dict['tfidf-rf']).T
			tfidf_xgb_metrics = pd.DataFrame(metrics_dict['tfidf-xgb']).T

			logger.info("Results from Tf-Idf features")
			logger.info("Random Forest")
			logger.info(f"Mean AUC: {round(tfidf_rf_metrics.AUC.mean(), 2)} ({round(tfidf_rf_metrics.AUC.std(), 2)})")
			logger.info(f"Mean AP: {round(tfidf_rf_metrics.AP.mean(), 2)} ({round(tfidf_rf_metrics.AP.std(), 2)})")
			logger.info(f"Mean F1: {round(tfidf_rf_metrics.F1.mean(), 2)} ({round(tfidf_rf_metrics.F1.std(), 2)})")
			logger.info(f"Null Model F1: {round(tfidf_rf_metrics['Null model'].mean(), 2)} ({round(tfidf_rf_metrics['Null model'].std(), 2)})")
			logger.info(f"Mean Balanced Accuracy: {round(tfidf_rf_metrics['Balanced Accuracy'].mean(), 2)} ({round(tfidf_rf_metrics['Balanced Accuracy'].mean(), 2)})")
			logger.info(f"Null Model Balanced Accuracy: {round(tfidf_rf_metrics['Balanced Accuracy Null'].mean(), 2)} ({round(tfidf_rf_metrics['Balanced Accuracy Null'].std(), 2)})")

			logger.info("Xgboost")
			logger.info(f"Mean AUC: {round(tfidf_xgb_metrics.AUC.mean(), 2)} ({round(tfidf_xgb_metrics.AUC.std(), 2)})")
			logger.info(f"Mean AP: {round(tfidf_xgb_metrics.AP.mean(), 2)} ({round(tfidf_xgb_metrics.AP.std(), 2)})")
			logger.info(f"Mean F1: {round(tfidf_xgb_metrics.F1.mean(), 2)} ({round(tfidf_xgb_metrics.F1.std(), 2)})")
			logger.info(f"Null Model F1: {round(tfidf_xgb_metrics['Null model'].mean(), 2)} ({round(tfidf_xgb_metrics['Null model'].std(), 2)})")
			logger.info(f"Mean Balanced Accuracy: {round(tfidf_xgb_metrics['Balanced Accuracy'].mean(), 2)} ({round(tfidf_xgb_metrics['Balanced Accuracy'].std(), 2)})")
			logger.info(f"Null Model Balanced Accuracy: {round(tfidf_xgb_metrics['Balanced Accuracy Null'].mean(), 2)} ({round(tfidf_xgb_metrics['Balanced Accuracy Null'].std(), 2)})")

			pd.DataFrame(zip(tfidf_rf_metrics['AUC'], tfidf_xgb_metrics['AUC'], tfidf_rf_metrics['AP'], tfidf_xgb_metrics['AP'], tfidf_rf_metrics['Accuracy'], tfidf_xgb_metrics['Accuracy']), 
			 columns=["RandomForest_AUC", "Xgboost_AUC", "RandomForest_AP", "Xgboost_AP", "RandomForest_Accuracy", "Xgboost_Accuracy"]).to_csv(join(OUT_DIR, f'TfIdf_Results_{trope}.csv'), index=False)

		if train_d2v:
			d2v_dm_rf_metrics = pd.DataFrame(metrics_dict['d2v-dm-rf']).T
			d2v_dm_xgb_metrics = pd.DataFrame(metrics_dict['d2v-dm-xgb']).T

			d2v_dm_concat_rf_metrics = pd.DataFrame(metrics_dict['d2v-dm-concat-rf']).T
			d2v_dm_concat_xgb_metrics = pd.DataFrame(metrics_dict['d2v-dm-concat-xgb']).T

			d2v_dbow_rf_metrics = pd.DataFrame(metrics_dict['d2v-dbow-rf']).T
			d2v_dbow_xgb_metrics = pd.DataFrame(metrics_dict['d2v-dbow-xgb']).T

			logger.info("Results from Doc2Vec features")
			logger.info("Random Forest")
			logger.info(f"Mean AUC: {round(np.mean(d2v_rf_metrics['AUC']), 2)} ({round(np.std(d2v_rf_metrics['AUC']), 2)})")
			logger.info(f"Mean AP: {round(np.mean(d2v_rf_metrics['AP']), 2)} ({round(np.std(d2v_rf_metrics['AP']), 2)})")
			logger.info(f"Mean F1: {round(np.mean(d2v_rf_metrics['F1']), 2)} ({round(np.std(d2v_rf_metrics['F1']), 2)})")
			logger.info(f"Null Model F1: {round(np.mean(d2v_rf_metrics['Null model']), 2)} ({round(np.std(d2v_rf_metrics['Null model']), 2)})")
			logger.info(f"Mean Balanced Accuracy: {round(np.mean(d2v_rf_metrics['Balanced Accuracy']), 2)} ({round(np.std(d2v_rf_metrics['Balanced Accuracy']), 2)})")
			logger.info(f"Null Model Balanced Accuracy: {round(np.mean(d2v_rf_metrics['Balanced Accuracy Null']), 2)} ({round(np.std(d2v_rf_metrics['Balanced Accuracy Null']), 2)})")

			logger.info("Xgboost")
			logger.info(f"Mean AUC: {round(np.mean(d2v_xgb_metrics['AUC']), 2)} ({round(np.std(d2v_xgb_metrics['AUC']), 2)})")
			logger.info(f"Mean AP: {round(np.mean(d2v_xgb_metrics['AP']), 2)} ({round(np.std(d2v_xgb_metrics['AP']), 2)})")
			logger.info(f"Mean F1: {round(np.mean(d2v_xgb_metrics['F1']), 2)} ({round(np.std(d2v_xgb_metrics['F1']), 2)})")
			logger.info(f"Null Model F1: {round(np.mean(d2v_xgb_metrics['Null model']), 2)} ({round(np.std(d2v_xgb_metrics['Null model']), 2)})")
			logger.info(f"Mean Balanced Accuracy: {round(np.mean(d2v_xgb_metrics['Balanced Accuracy']), 2)} ({round(np.std(d2v_xgb_metrics['Balanced Accuracy']), 2)})")
			logger.info(f"Null Model Balanced Accuracy: {round(np.mean(d2v_xgb_metrics['Balanced Accuracy Null']), 2)} ({round(np.std(d2v_xgb_metrics['Balanced Accuracy Null']), 2)})")

			pd.DataFrame(zip(d2v_xgb_metrics['AUC'], d2v_xgb_metrics['AUC'], d2v_xgb_metrics['AP'], d2v_xgb_metrics['AP'], d2v_xgb_metrics['Accuracy'], d2v_xgb_metrics['Accuracy']), 
			 columns=["RandomForest_AUC", "Xgboost_AUC", "RandomForest_AP", "Xgboost_AP", "RandomForest_Accuracy", "Xgboost_Accuracy"]).to_csv(join(OUT_DIR, f'Doc2Vec_Results_{trope}.csv'), index=False)

	logger.info(f"Completed training and evaluating models. Time taken: {(datetime.now() - start_time).seconds} seconds")
