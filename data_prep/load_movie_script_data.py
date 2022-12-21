from collections import defaultdict
import json
import logging
import os
from os.path import join
import re
from typing import Dict, List

from gensim.parsing.preprocessing import preprocess_string, remove_stopwords, stem_text, strip_punctuation, strip_short
import numpy as np
import pandas as pd

from utils.utils import DATA_DIR, configure_logging, MOVIE_SCRIPT_DATA_DIR, MOVIE_SCRIPT_TROPE_DATA_DIR, MOVIE_TV_TROPES_DATA_DIR


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_clean_timos_tropes(tropes_count_dict) -> List[str]:
    """
    This function gets list of 95 tropes from TiMoS dataset.
    Args:
        tropes_count_dict (Dict[str, int]): Dictionary containing trope name as key and its frequency count as value
    Returns:
        List[str]: List of pre-processed trope names
    """
    tropes_df = pd.read_csv(join(DATA_DIR, 'TiMoS', 'tropes.csv'))
    trope_match = []
    trope_no_match = []
    for trope in tropes_df.trope.tolist():
        trope_capitalize = ''.join([x.capitalize() for x in trope.split()])
        trope_no_special_chars = re.sub(r"[^a-zA-Z]", "", trope_capitalize)
        if trope_no_special_chars in tropes_count_dict.keys():
            trope_match.append(trope_no_special_chars)
        else:
            trope_no_match.append(trope_no_special_chars)
    trope_no_match[0] = 'ButtMonkey'
    trope_no_match[1] = 'AxCrazy'
    trope_no_match[2] = 'HeroicBSOD'
    trope_no_match[3] = 'BigNo'
    trope_no_match[4] = 'TheReasonYouSuckSpeech'
    trope_no_match[5] = 'PrecisionFStrike'
    trope_no_match[6] = 'ClusterFBomb'
    trope_no_match[7] = 'LaserGuidedKarma'

    return trope_match + trope_no_match


def get_genre_movie_list(genre: str) -> List[str]:
    """
    This function returns all the json filenames containing movie dialogs
    Args:
        genre (str): String containing genre name.
        
    Returns:
        List[str]: List of strings containing json filenames.
    """
    movie_genre_json_list = []

    if os.path.exists(join(MOVIE_SCRIPT_DATA_DIR, genre)):
        movies_per_genre = os.listdir(join(DATA_DIR, 'MovieScripts', genre))
        for movie in movies_per_genre:
            if movie.endswith('.json'):
                movie_genre_json_list.append(movie)
    else:
        logger.info("Genre path does not exist!")

    return movie_genre_json_list


def get_movies_with_tropes() -> pd.DataFrame:
    """
    This function gets all movies across different genres and find movies with scripts and tropes.
    Args:
    Returns:
        pd.DataFrame: Pandas dataframe containing three columns namely movie names, tropes and genre
    """

    # Read all movie script json files for Action genre
    genres = ['Action', 'Drama', 'Thriller', 'Comedy', 'Crime', 'Romance', 'Adventure', 'Sci-Fi', 'Horror', 
              'Animation', 'War', 'Family', 'Musical', 'Mystery']
    genre_movie_json_list = []
    movie_names = []
    tropes = []
    genre_list = []

    for genre in genres:
        genre_movie_json_list = get_genre_movie_list(genre)
        
        # Remove .json file extension from movie filenames
        movie_list = [movie.split('.json')[0] for movie in genre_movie_json_list]
        
        # Find movies that match with TvTropes
        # Read csv file mapping movie names containing script data with their tropes
        genre_movie_script_trope_df = pd.read_csv(join(MOVIE_SCRIPT_TROPE_DATA_DIR, f'{genre.lower()}_movie_script_trope_match.csv'))
        movie_match_df = genre_movie_script_trope_df.loc[genre_movie_script_trope_df.Movie_Script.isin(movie_list)].copy()
        
        logger.info(f"Genre: {genre}, total movies: {len(movie_list)}, movies matched: {len(movie_match_df)}")
        
        movie_names += movie_match_df.Movie_Script.tolist()
        tropes += movie_match_df.Movie_Trope.tolist()    
        genre_list += [genre] * len(movie_match_df)

    movie_tropes_df = pd.DataFrame(list(zip(movie_names, tropes, genre_list)), columns=['Movies', 'Tropes', 'Genre'])

    return movie_tropes_df


def load_json_movie_dialog_file(genre: str, movie_filename: str) -> List[List[Dict[str, str]]]:
    """
    Loads the json data contained in movie file:
    Args:
        genre (str): String containing genre name.
        movie_filename (str): String containing movie filename.
    
    Returns:
        List[List[Dict[str]]]: List of lists with each nested list containing a dictionary.
    """

    if os.path.exists(join(MOVIE_SCRIPT_DATA_DIR, genre, movie_filename)):
        with open(join(MOVIE_SCRIPT_DATA_DIR, genre, movie_filename), 'r') as f:
            movie_dialog_json = json.loads(f.read())
    else:
        logger.info(f"Path to ScreenPy parser output with {genre}/{movie_filename} does not exist")
    return movie_dialog_json


def parse_movie_dialog_data(movie_json_data: List[List[Dict[str, str]]], 
                            verbose: bool = False):
    """
    This function parses the movie json data, and collects the following information,
        1. Unique characters with dialogs
        2. Number of dialogs per character
        3. Dialogs of all characters concatenated into a string
    Args:
        movie_json_data (List[List[Dict[str, str]]]): Json data containing movie character names and dialogs.
        verbose (bool): Boolean indicating whether raw dialogs should be printed.
        
    Returns:
        Dict[str, Any]: Dictionary with movie name as key and various nested dictionaries 
        containing data mentioned in function description.
    """
    movie_characters = set()
    movie_dialogs = list()
    dialogs_per_character = defaultdict(int)
    movie_info_dict = defaultdict()
    for scene_dialogs in movie_json_data:
        for dialog_info in scene_dialogs:
            if 'speaker/title' in dialog_info['head_text']:
                dialog_speaker = dialog_info['head_text']['speaker/title']
                if verbose:
                    logger.info(f"Speaker: {dialog_speaker}")
                    logger.info(dialog_info['text'])
                character = dialog_speaker.split('(')[0].strip()
                movie_characters = movie_characters.union([character])
                dialogs_per_character[character] += 1
            movie_dialogs.append(dialog_info['text'])

    movie_info_dict['characters'] = movie_characters
    movie_info_dict['actor_dialog_count'] = dialogs_per_character
    movie_info_dict['dialogs'] = ' '.join(movie_dialogs)

    return movie_info_dict


def preprocess_movie_script_data():
    """
    Preprocess movie scripts by applying few text processing tasks such as lowercase, remove stopwords,
    remove punctuation and strip words shorter than 3 letters.
    Args:
    Returns:
        Tuple[Dict, Dict]: A tuple of two dictionaries. The first dictionary contains movie as key and movie dialogs as
        a single string as value. The second dictionary also contains movie as key and list of tropes as value.
    """
    custom_filters = [lambda x: x.lower(), remove_stopwords, strip_punctuation, strip_short]

    all_raw_movie_dialogs = defaultdict()
    all_preprocess_movie_dialogs = defaultdict()
    movie_trope_dict = defaultdict()

    logger.info("Get dataframe of all movies that match with movie list in TvTropes")
    movie_tropes_df = get_movies_with_tropes()
    movie_tropes_df = movie_tropes_df.drop_duplicates(subset=['Movies', 'Tropes'])
    logger.info(f"Movies with tropes: {len(movie_tropes_df)}")

    logger.info("Read TVTropes Json file")
    tvtropes_json_dict = read_tvtropes_json_file()

    for movie_row in movie_tropes_df.iterrows():
        movie = movie_row[1].Movies
        genre = movie_row[1].Genre
        movie_filename = movie + '.json'
        movie_json_data = load_json_movie_dialog_file(genre, movie_filename)
        
        # Parse movie dialogs and preprocess text
        all_raw_movie_dialogs[movie] = parse_movie_dialog_data(movie_json_data)
        preprocess_movie_dialog = preprocess_string(all_raw_movie_dialogs[movie]['dialogs'], custom_filters)
        if len(preprocess_movie_dialog) > 0:
            all_preprocess_movie_dialogs[movie] = preprocess_movie_dialog
            # Collect list of tropes for the movie
            movie_trope_dict[movie] = tvtropes_json_dict[movie_row[1].Tropes]

    # For each movie filter out tropes which appear in less than min_trope_count movies
    movie_tropes_subset_dict = defaultdict()
    all_movie_dialogs_subset = defaultdict()
    for movie, trope in movie_trope_dict.items():
        if len(trope) == 0:
            logger.info(f'{movie} has no tropes in json file.')
        elif len(all_preprocess_movie_dialogs[movie])==0:
            logger.info(f"{movie} has no movie script data.")
        else:
            movie_tropes_subset_dict[movie] = list(trope)
            all_movie_dialogs_subset[movie] = all_preprocess_movie_dialogs[movie]

    return all_movie_dialogs_subset, movie_tropes_subset_dict


def read_tvtropes_json_file():
    """
    Read json file contianing movie tropes
    """
    with open(join(MOVIE_TV_TROPES_DATA_DIR, 'films_tropes_20190501.json'), 'rb') as file:
        tvtropes_json_dict = json.load(file)

    return tvtropes_json_dict
