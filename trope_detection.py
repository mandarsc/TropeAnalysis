from collections import Counter, defaultdict
from datetime import datetime
import logging
from os.path import join
import re
from typing import Dict, List

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import preprocess_string, remove_stopwords, stem_text, strip_punctuation, strip_short
import matplotlib.pyplot as plt
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



def trope_detection_tfidf(movie_script_dialog: Dict[str, str], movie_tropes_dict: Dict[str, List[str]], tfidf_vocab_size: int, d2v_size: int, trope: str, train_tfidf: bool = True, train_d2v: bool = False):

	tfidf_rf_metrics = defaultdict(str)
	tfidf_xgb_metrics = defaultdict(str)

	d2v_rf_metrics = defaultdict(str)
	d2v_xgb_metrics = defaultdict(str)

	d2v_rf_metrics['AP'] = []
	d2v_rf_metrics['AUC'] = []

	d2v_xgb_metrics['AP'] = []
	d2v_xgb_metrics['AUC'] = []

	tfidf_rf_metrics['AP'] = []
	tfidf_rf_metrics['AUC'] = []
	tfidf_rf_metrics['F1'] = []
	tfidf_rf_metrics['Null model'] = []
	tfidf_rf_metrics['Balanced Accuracy'] = []
	tfidf_rf_metrics['Balanced Accuracy Null'] = []

	tfidf_xgb_metrics['AP'] = []
	tfidf_xgb_metrics['AUC'] = []
	tfidf_xgb_metrics['F1'] = []
	tfidf_xgb_metrics['Null model'] = []
	tfidf_xgb_metrics['Balanced Accuracy'] = []
	tfidf_xgb_metrics['Balanced Accuracy Null'] = []

	rf_estimators = 200
	xgb_estimators = 200

	multi_label = MultiLabelBinarizer()
	y = multi_label.fit_transform([set(movie_tropes) for movie_tropes in movie_tropes_dict.values()])
	trope_idx = list(multi_label.classes_).index(trope)

	tfidf_rf_auc = 0
	tfidf_rf_ap = 0
	tfidf_xgb_auc = 0
	tfidf_xgb_ap = 0

	d2v_rf_auc = 0
	d2v_rf_ap = 0
	d2v_xgb_auc = 0
	d2v_xgb_ap = 0

	num_cv_folds = 10
	rf_tfidf_dict = defaultdict(int)
	xgb_tfidf_dict = defaultdict(int)

	if y[:, trope_idx].sum() < 2:
		logger.info(f"Skipping trope {trope} with less than 2 instances")
	else:
		random_state = np.random.randint(10000, size=num_cv_folds)
		documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(movie_script_dialog.values())]

		for i in range(num_cv_folds):

			if train_tfidf:
				X_train_docs, X_test_docs, y_train, y_test = train_test_split(list(movie_script_dialog.values()), y[:, trope_idx], shuffle=True, train_size=0.8, stratify=y[:, trope_idx], random_state=random_state[i])

				logger.info(f"Iteration: {i+1}) {trope} distribution: train ({np.bincount(y_train)}), test ({np.bincount(y_test)})")

				X_train_tfidf_docs = [' '.join(list(x)) for x in X_train_docs]
				X_test_tfidf_docs = [' '.join(list(x)) for x in X_test_docs]

				tfidf_fit = TfidfVectorizer(max_features=tfidf_vocab_size, ngram_range=(1, 1), stop_words='english').fit(raw_documents=X_train_tfidf_docs)
				X_tfidf_train_vec = tfidf_fit.transform(X_train_tfidf_docs)
				X_tfidf_test_vec = tfidf_fit.transform(X_test_tfidf_docs)

				tfidf_rf_metrics_dict = train_eval_classifiers(X_tfidf_train_vec, X_tfidf_test_vec, y_train, y_test, classifier='rf', n_estimators = rf_estimators)
				tfidf_xgb_metrics_dict = train_eval_classifiers(X_tfidf_train_vec, X_tfidf_test_vec, y_train, y_test, n_estimators = xgb_estimators)
	
				tfidf_rf_metrics['AUC'].append(tfidf_rf_metrics_dict['AUC'])
				tfidf_rf_metrics['AP'].append(tfidf_rf_metrics_dict['AP'])
				tfidf_rf_metrics['F1'].append(tfidf_rf_metrics_dict['F1'])
				tfidf_rf_metrics['Null model'].append(tfidf_rf_metrics_dict['Null model'])
				tfidf_rf_metrics['Balanced Accuracy'].append(tfidf_rf_metrics_dict['Balanced Accuracy'])
				tfidf_rf_metrics['Balanced Accuracy Null'].append(tfidf_rf_metrics_dict['Balanced Accuracy Null'])

				tfidf_xgb_metrics['AUC'].append(tfidf_xgb_metrics_dict['AUC'])
				tfidf_xgb_metrics['AP'].append(tfidf_xgb_metrics_dict['AP'])
				tfidf_xgb_metrics['F1'].append(tfidf_xgb_metrics_dict['F1'])
				tfidf_xgb_metrics['Null model'].append(tfidf_xgb_metrics_dict['Null model'])
				tfidf_xgb_metrics['Balanced Accuracy'].append(tfidf_xgb_metrics_dict['Balanced Accuracy'])
				tfidf_xgb_metrics['Balanced Accuracy Null'].append(tfidf_xgb_metrics_dict['Balanced Accuracy Null'])

			if train_d2v:
				X_train_docs, X_test_docs, y_train, y_test = train_test_split(documents, y[:, trope_idx], shuffle=True, train_size=0.8, stratify=y[:, trope_idx], random_state=random_state[i])

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

				d2v_rf_metrics_dict = train_eval_classifiers(X_train_dv, X_test_dv, y_train, y_test, classifier='rf')
				d2v_xgb_metrics_dict = train_eval_classifiers(X_train_dv, X_test_dv, y_train, y_test)

				d2v_rf_metrics['AUC'].append(d2v_rf_metrics_dict['AUC'])
				d2v_rf_metrics['AP'].append(d2v_rf_metrics_dict['AP'])
				d2v_xgb_metrics['AUC'].append(d2v_xgb_metrics_dict['AUC'])
				d2v_xgb_metrics['AP'].append(d2v_rf_metrics_dict['AP'])

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

	tfidf_vocab_size = 100
	d2v_dim = 200

	# logger.info("Get all TiMoS tropes")
	# tropes_of_interest = get_clean_timos_tropes(tropes_count_dict)
	tropes_of_interest = ['ShoutOut', 'OhCrap', 'ChekhovsGun', 'Foreshadowing', 'BittersweetEnding']
	# tropes_of_interest = ['ShoutOut']
	logger.info(f"Using top-{len(tropes_of_interest)} frequently occuring tropes")

	logger.info(f"Running trope detection...")
	for trope in tropes_of_interest:
		tfidf_rf_metrics, tfidf_xgb_metrics, d2v_rf_metrics, d2v_xgb_metrics = trope_detection_tfidf(movie_script_dialog, movie_tropes_dict, tfidf_vocab_size, 
			d2v_dim, trope, train_tfidf, train_d2v)

		if train_tfidf:
			logger.info("Results from Tf-Idf features")
			logger.info("Random Forest")
			logger.info(f"Mean AUC: {round(np.mean(tfidf_rf_metrics['AUC']), 2)} ({round(np.std(tfidf_rf_metrics['AUC']), 2)})")
			logger.info(f"Mean AP: {round(np.mean(tfidf_rf_metrics['AP']), 2)} ({round(np.std(tfidf_rf_metrics['AP']), 2)})")
			logger.info(f"Mean F1: {round(np.mean(tfidf_rf_metrics['F1']), 2)} ({round(np.std(tfidf_rf_metrics['F1']), 2)})")
			logger.info(f"Null Model F1: {round(np.mean(tfidf_rf_metrics['Null model']), 2)} ({round(np.std(tfidf_rf_metrics['Null model']), 2)})")
			logger.info(f"Mean Balanced Accuracy: {round(np.mean(tfidf_rf_metrics['Balanced Accuracy']), 2)} ({round(np.std(tfidf_rf_metrics['Balanced Accuracy']), 2)})")
			logger.info(f"Null Model Balanced Accuracy: {round(np.mean(tfidf_rf_metrics['Balanced Accuracy Null']), 2)} ({round(np.std(tfidf_rf_metrics['Balanced Accuracy Null']), 2)})")
			# logger.info(f"Mean F1-score: {round(np.mean(tfidf_rf_metrics['F1']), 2)} ({round(np.std(tfidf_rf_metrics['F1']), 2)})")

			logger.info("Xgboost")
			logger.info(f"Mean AUC: {round(np.mean(tfidf_xgb_metrics['AUC']), 2)} ({round(np.std(tfidf_xgb_metrics['AUC']), 2)})")
			logger.info(f"Mean AP: {round(np.mean(tfidf_xgb_metrics['AP']), 2)} ({round(np.std(tfidf_xgb_metrics['AP']), 2)})")
			logger.info(f"Mean F1: {round(np.mean(tfidf_xgb_metrics['F1']), 2)} ({round(np.std(tfidf_xgb_metrics['F1']), 2)})")
			logger.info(f"Null Model F1: {round(np.mean(tfidf_xgb_metrics['Null model']), 2)} ({round(np.std(tfidf_xgb_metrics['Null model']), 2)})")
			logger.info(f"Mean Balanced Accuracy: {round(np.mean(tfidf_xgb_metrics['Balanced Accuracy']), 2)} ({round(np.std(tfidf_xgb_metrics['Balanced Accuracy']), 2)})")
			logger.info(f"Null Model Balanced Accuracy: {round(np.mean(tfidf_xgb_metrics['Balanced Accuracy Null']), 2)} ({round(np.std(tfidf_xgb_metrics['Balanced Accuracy Null']), 2)})")
			# logger.info(f"Mean F1-score: {round(np.mean(tfidf_xgb_metrics['F1']), 2)} ({round(np.std(tfidf_xgb_metrics['F1']), 2)})")

			pd.DataFrame(zip(tfidf_rf_metrics['AUC'], tfidf_xgb_metrics['AUC'], tfidf_rf_metrics['AP'], tfidf_xgb_metrics['AP'], tfidf_rf_metrics['Accuracy'], tfidf_xgb_metrics['Accuracy']), 
			 columns=["RandomForest_AUC", "Xgboost_AUC", "RandomForest_AP", "Xgboost_AP", "RandomForest_Accuracy", "Xgboost_Accuracy"]).to_csv(join(OUT_DIR, f'TfIdf_Results_{trope}.csv'), index=False)

		if train_d2v:
			logger.info("Results from Doc2Vec features")
			logger.info("Random Forest")
			logger.info(f"Mean AUC: {round(np.mean(d2v_rf_metrics['AUC']), 2)} ({round(np.std(d2v_rf_metrics['AUC']), 2)})")
			logger.info(f"Mean AP: {round(np.mean(d2v_rf_metrics['AP']), 2)} ({round(np.std(d2v_rf_metrics['AP']), 2)})")
			# logger.info(f"Mean F1-score: {round(np.mean(d2v_rf_metrics['F1']), 2)} ({round(np.std(d2v_rf_metrics['F1']), 2)})")

			logger.info("Xgboost")
			logger.info(f"Mean AUC: {round(np.mean(d2v_xgb_metrics['AUC']), 2)} ({round(np.std(d2v_xgb_metrics['AUC']), 2)})")
			logger.info(f"Mean AP: {round(np.mean(d2v_xgb_metrics['AP']), 2)} ({round(np.std(d2v_xgb_metrics['AP']), 2)})")
			# logger.info(f"Mean F1-score: {round(np.mean(d2v_xgb_metrics['F1']), 2)} ({round(np.std(d2v_xgb_metrics['F1']), 2)})")

			pd.DataFrame(zip(d2v_rf_metrics['AUC'], d2v_xgb_metrics['AUC'], d2v_rf_metrics['AP'], d2v_xgb_metrics['AP']),
			 columns=["RandomForest_AUC", "Xgboost_AUC", "RandomForest_AP", "Xgboost_AP"]).to_csv(join(OUT_DIR, f'D2v_Results_{trope}.csv'), index=False)

	logger.info(f"Completed training and evaluating models. Time taken: {(datetime.now() - start_time).seconds} seconds")
