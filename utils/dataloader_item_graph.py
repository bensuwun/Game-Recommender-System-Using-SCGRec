import sys
import torch
import logging
import utils.logs.log_config as log_config
logger = logging.getLogger(__name__)
log_config.SetDefaultConfig(logger)

import dgl
import pandas as pd
import numpy as np
from tqdm import tqdm
from dgl.data import DGLDataset

class Dataloader_item_graph(DGLDataset):
    """
        Initializes the item graph or the game context graph (5.1 in Liangwei et al's paper.)
    """
    def __init__(self, graph, app_id_path, publisher_path, developer_path, genre_path, tags_path, cos_similarity_path, categorical_review_score_path, app_sentiments_path):
        self.app_id_path = app_id_path
        self.publisher_path = publisher_path
        self.developer_path = developer_path
        self.genre_path = genre_path
        self.tags_path = tags_path
        self.cos_similarity_path = cos_similarity_path
        self.categorical_review_score_path = categorical_review_score_path
        self.app_sentiments_path = app_sentiments_path

        # Retrieve co-features of games
        logging.info("reading item graph")
        self.app_id_mapping = self.read_id_mapping(self.app_id_path)
        self.publisher = self.read_mapping(self.publisher_path)
        self.developer = self.read_mapping(self.developer_path)
        self.genre = self.read_mapping(self.genre_path)
        self.similarity_score_nodes, self.similarity_scores = self.read_cos_similarity(self.cos_similarity_path)
        self.tag = self.read_mapping(self.tags_path)
        self.categorical_review_scores = self.read_categorical_review_scores(self.categorical_review_score_path)
        self.sentiment_scores = self.read_sentiment_scores(self.app_sentiments_path)

        # Initialize game context graph from co-features
        graph_data = {
            ('game', 'co_publisher', 'game'): self.publisher,
            ('game', 'co_developer', 'game'): self.developer,
            ('game', 'co_genre', 'game'): self.genre,
            #* added tags
            ('game', 'co_tag', 'game'): self.tag,
            #* added categorical review scores
            ('game', 'co_categorical_reviews', 'game'): self.categorical_review_scores,
            #* added sentiment scores
            ('game', 'co_sentiment_reviews', 'game'): self.sentiment_scores,
            #* added similarity score
            ('game', 'desc_similarity', 'game'): self.similarity_score_nodes,
        }
        self.graph = dgl.heterograph(graph_data)

        #* Add actual scores to cosine similarity edges
        self.graph.edges['desc_similarity'].data['score'] = self.similarity_scores

        # Add app info graph dataloader_steam.graph into game nodes
        self.graph.nodes['game'].data['h'] = graph.ndata['h']['game'].float()

    def read_id_mapping(self, path):
        """
            Reads app_id.txt, maps app ids to new ids.
            Sample return value: {'273813': 0, '2312515': 1}

            :return: Dictionary, key = ID, value = mapped ID
        """
        mapping = {}
        count = 0
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line not in mapping:
                    mapping[line] = count
                    count += 1
        return mapping
    
    def read_mapping(self, path):
        """
            Used to read the Developers, Genres, and Publishers txt files.
            It then retrieves games with co-developer, co-genre, or co-publisher through a tuple (Tensor src, Tensor destination).
            Sample return value (numbers = mapped game IDs): 
                (tensor([2, 3, 4, 5, 4, 6, 5, 6]), tensor([3, 2, 5, 4, 6, 4, 6, 5]))

            :return: Tuple, 2 values (Tensor src, Tensor destination)
        """
        mapping = {}
        with open(path, 'r') as f:
            for line in f:
                line = line.strip().split(',')
                if line[1] != '':
                    # Create new list if app id not in list
                    # NOTE: Only obtains last value due to weird if condition
                    if line[0] not in mapping:
                        mapping[self.app_id_mapping[line[0]]] = [line[1]]
                    else:
                        mapping[self.app_id_mapping[line[0]]].append(line[1])

        for key in mapping:
            mapping[key] = set(mapping[key])

        # Retrieve games with co-feature
        src = []
        dst = []
        keys = list(mapping.keys())
        for i in range(len(keys) - 1):
            for j in range(i +1, len(keys)):
                game1 = keys[i]
                game2 = keys[j]
                # Binary operator AND, not Logical, check if equal values
                # NOTE: Values must ALL be the same, no problem if list has single value like current implementation though.
                if len(mapping[game1] & mapping[game2]) > 0:
                    src.extend([game1, game2])
                    dst.extend([game2, game1])
        return (torch.tensor(src), torch.tensor(dst))

    def read_categorical_review_scores(self, path):
        df = pd.read_csv(path)

        # Filter dataframe to only obtain necessary columns
        df = df[["appids", "overall_review_score"]]

        # Map app ids, key = mapped app id | value = categorical review score
        mapping = {}
        for i in range(len(df)):
            mapped_appid = self.app_id_mapping[str(df.iloc[i, 0])]        
            mapping[mapped_appid] = df.iloc[i, 1]

        # Map values (e.g. Very Positive = 0, Positive = 1, nan = 2)
        mapping_value2id = {}
        count = 0
        for value in mapping.values():
            if value not in mapping_value2id:
                # Extra check for nan
                mapping_value2id[value] = count
                count += 1

        for key in mapping:
            mapping[key] = mapping_value2id[mapping[key]]

        # Retrieve games with same categorical review scores
        src = []
        dst = []
        keys = list(mapping.keys())
        for i in range(len(keys) - 1):
            for j in range(i + 1, len(keys)):
                game1 = keys[i]
                game2 = keys[j]
                # only establish connections for games WITH THE SAME categorical review scores, excluding nan values
                if (mapping[game1] == mapping[game2]):
                    src.extend([game1, game2])
                    dst.extend([game2, game1])

        return (torch.tensor(src), torch.tensor(dst))

    def read_sentiment_scores(self, path):
        senti_scores = pd.read_pickle(path)
        senti_scores.drop(columns=["index"], inplace=True)
        senti_scores = senti_scores.set_index('appid')

        # Get the mean/median/mode of each app's sentiment scores
        generalized_senti_scores = senti_scores.mean(axis = 1)
        generalized_senti_scores = self.convert_senti_scores(generalized_senti_scores)

        # Map app ids, key = mapped app id | value = categorical sentiment score
        mapping = {}
        for appid, score in generalized_senti_scores.items():
            mapping[self.app_id_mapping[str(appid)]] = score

        # Map values (e.g. Very Negative = 0, Negative = 1)
        mapping_value2id = {}
        count = 0
        for value in mapping.values():
            if value not in mapping_value2id:
                mapping_value2id[value] = count
                count += 1
        for key in mapping:
            mapping[key] = mapping_value2id[mapping[key]]

        # Retrieve games with same sentiment score bin
        src = []
        dst = []
        keys = list(mapping.keys())
        for i in range(len(keys) - 1):
            for j in range(i + 1, len(keys)):
                game1 = keys[i]
                game2 = keys[j]
                # only establish connections for games with categorical review scores
                if (mapping[game1] == mapping[game2]):
                    src.extend([game1, game2])
                    dst.extend([game2, game1])
        return (torch.tensor(src), torch.tensor(dst))

    def convert_senti_scores(self, senti_scores):
        # -1.0 to -0.5 = Very Negative ,-0.499 to 0.01 = Negative ,0 = Neutral, 0.01 - 0.499 = Positive, 0.5 - 1.0 = Very Positive
        mapped_scores = {}
        for appid, score in senti_scores.items():
            if score == 0:
                label = 'Neutral'         
            elif score >= -1 and score <= -0.5:
                label = 'Very Negative'
            elif score > -0.5 and score < 0:
                label = 'Negative'
            elif score > 0 and score < 0.5:
                label = 'Positive'
            elif score >= 0.5 and score <= 1:
                label = 'Very Positive'
            else:
                label = np.nan
            # print("App: {} | Score: {} | Label: {}".format(appid, score, label))
            mapped_scores[appid] = label
        return mapped_scores

    def read_cos_similarity(self, path):
        df = pd.read_pickle(path)

        # Iterate through dataframe, do not include (1) same appids
        src = []
        dst = []
        similarity_scores = []

        ctr = 0
        for row_idx in tqdm(range(df.shape[0])):
            ctr += 1
            game1 = df.index[row_idx]
            for col_idx in range(ctr, df.shape[1]):
                game2 = df.columns[col_idx]
                score = df[game1][game2]
                mapped_game1 = self.app_id_mapping[str(game1)]
                mapped_game2 = self.app_id_mapping[str(game2)]
                src.extend([mapped_game1, mapped_game2])
                dst.extend([mapped_game2, mapped_game1])
                similarity_scores.extend([round(score,2), round(score,2)])
        
        return (torch.tensor(src), torch.tensor(dst)), torch.tensor(similarity_scores)
