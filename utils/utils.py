import logging
from os.path import join

HOME_DIR = '/home/mandar/Mandar/Tropes/'
DATA_DIR = join(HOME_DIR, 'TropesDataset')
OUT_DIR = join(HOME_DIR, 'TropeAnalysis', 'output')


def configure_logging(logger):
    """
    This function intializes a logger to record logging statements.
    """
    # create console handler and set level to info
    logger_handler = logging.StreamHandler() 
    logger_handler.setLevel(logging.INFO)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    logger_handler.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(logger_handler)


def plot_auc_scores(rf_tfidf, xgb_tfidf, rf_d2v, xgb_d2v, tfidf_vocab, d2v_vocab, trope):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 5))
    
    ax1.plot(np.arange(len(rf_tfidf)), rf_tfidf, color='red', marker='+', label='Random Forest')
    ax1.plot(np.arange(len(xgb_tfidf)), xgb_tfidf, color='black', marker='o', label='XGBoost')
    ax1.set_xlabel('Tf-Idf Vocabulary Size')
    ax1.set_ylabel('AUC Score')
    ax1.set_xticks(np.arange(len(tfidf_vocab)))
    ax1.set_xticklabels(tfidf_vocab, rotation=45)
    ax1.set_title(f"Tf-Idf Trope {trope}")
    ax1.legend(loc='upper right')
    ax1.set_ylim([0, 1])
    
    ax2.plot(np.arange(len(rf_d2v)), rf_d2v, color='red', marker='+', label='Random Forest')
    ax2.plot(np.arange(len(xgb_d2v)), xgb_d2v, color='black', marker='o', label='XGBoost')
    ax2.set_xlabel('Doc2Vec Vocabulary Size')
    ax2.set_ylabel('AUC Score')
    ax2.set_xticks(np.arange(len(d2v_vocab)))
    ax2.set_xticklabels(d2v_vocab, rotation=45)
    ax2.set_title(f"Doc2Vec Trope {trope}")
    ax2.legend(loc='upper right')
    ax2.set_ylim([0, 1])    
    
    plt.savefig(f"Trope{trope}Result.jpg", dpi=300)
    plt.show()
