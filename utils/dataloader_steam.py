import os
import sys
from dgl.data.utils import save_graphs
from tqdm import tqdm   # used to show progress bar when iterating
from .NegativeSampler import NegativeSampler
import torch
import utils.logs.log_config as log_config
import logging
logger = logging.getLogger(__name__)
log_config.SetDefaultConfig(logger)

import numpy as np
import dgl
from dgl.data import DGLDataset
import pandas as pd

class Dataloader_steam(DGLDataset):
    def __init__(self, args, root_path, user_id_path, app_id_path, app_info_path, friends_path, developer_path, publisher_path, genres_path, tags_path, device = 'cpu', name = 'steam'):
        logger.info("steam dataloader init")

        self.args = args
        self.root_path = root_path
        self.user_id_path = user_id_path
        self.app_id_path = app_id_path
        self.app_info_path = app_info_path
        self.friends_path = friends_path
        self.developer_path = developer_path
        self.publisher_path = publisher_path
        self.genres_path = genres_path
        self.tags_path = tags_path
        self.device = device
        self.graph_path = self.root_path + '/graph.bin'     # graph.bin derived from dgl.save_graphs(...)
        self.game_path = self.root_path + '/train_game.txt'
        self.time_path = self.root_path + '/train_time.txt'
        self.valid_path = self.root_path + '/valid_data/valid_game.txt'
        self.test_path = self.root_path + '/test_data/test_game.txt'

        logger.info("reading user id mapping from {}".format(self.user_id_path))
        self.user_id_mapping = self.read_id_mapping(self.user_id_path)
        logger.info("reading app id mapping from {}".format(self.app_id_path))
        self.app_id_mapping = self.read_id_mapping(self.app_id_path)

        logger.info("build valid data")
        self.valid_data = self.build_valid_data(self.valid_path)

        logger.info("build test data")
        self.test_data = self.build_valid_data(self.test_path)

        # If preprocessed graphs exist, load those (currently none)
        if os.path.exists(self.graph_path):
            logger.info("loading preprocessed data")
            self.graph = dgl.load_graphs(self.graph_path)
            self.graph = self.graph[0][0]
            logger.info("reading user game information")
            self.dic_user_game = self.read_dic_user_game(self.game_path)

        else:
            self.process()
            # TODO: Uncomment when ready to save
            dgl.save_graphs(self.graph_path, self.graph)

        self.dataloader = self.build_dataloader(self.args, self.graph)

    def __getitem__(self, i):
        pass

    def __len__(self):
        pass

    def read_id_mapping(self, path):
        """
            Used for reading the app_id.txt and users.txt, both of which contain a list of IDs in single-data format.
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

    def build_valid_data(self, path):
        """
            Used for reading and building the validation and test sets.
            Example return value: {'1234': [23, 42, 52]} 

            :return: Dictionary where keys = mapped user IDs, values = list of mapped game IDs owned by user.
        """
        users = {}
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(',')
                user = self.user_id_mapping[line[0]]
                games = [self.app_id_mapping[game] for game in line[1:]]
                users[user] = games
        return users

    def process(self):
        """
            When the graph (graph.bin) has not been generated, this method is used to set the following: 
            (1) app_info, (2) publisher, (3) developer, (4) genre, (5) user_game, (6) dic_user_game, (7) friends, (8) graph
        """
        logger.info("reading app info from {}".format(self.app_info_path))
        self.app_info = self.read_app_info(self.app_info_path)

        logger.info("reading publisher from {}".format(self.publisher_path))
        self.publisher = self.read_mapping(self.publisher_path)

        logger.info("reading developer from {}".format(self.developer_path))
        self.developer = self.read_mapping(self.developer_path)

        logger.info("reading genre from {}".format(self.genres_path))
        self.genre = self.read_mapping(self.genres_path)

        logger.info("reading tag from {}".format(self.tags_path))
        self.tag = self.read_mapping(self.tags_path)

        logger.info("reading user item play time from {}".format(self.game_path))
        self.user_game, self.dic_user_game = self.read_play_time_rank(self.game_path, self.time_path)

        logger.info("reading friend list from {}".format(self.friends_path))
        self.friends = self.read_friends(self.friends_path)

        graph_data = {
            #   (Source Node Type | Edge Type | Destination Node Type)
            ('user', 'friend of', 'user'): (self.friends[:, 0], self.friends[:, 1]),

            ('game', 'developed by', 'developer'): (torch.tensor(list(self.developer.keys())), torch.tensor(list(self.developer.values()))),

            ('developer', 'develop', 'game'): (torch.tensor(list(self.developer.values())), torch.tensor(list(self.developer.keys()))),

            ('game', 'published by', 'publisher'): (torch.tensor(list(self.publisher.keys())), torch.tensor(list(self.publisher.values()))),

            ('publisher', 'publish', 'game'): (torch.tensor(list(self.publisher.values())), torch.tensor(list(self.publisher.keys()))),

            ('game', 'genre', 'type'): (torch.tensor(list(self.genre.keys())), torch.tensor(list(self.genre.values()))),

            ('type', 'genred', 'game'): (torch.tensor(list(self.genre.values())), torch.tensor(list(self.genre.keys()))),

            #* added tags to graph
            ('game', 'tag', 'tag_type'): (torch.tensor(list(self.tag.keys())), torch.tensor(list(self.tag.values()))),

            ('tag_type', 'tagged', 'game'): (torch.tensor(list(self.tag.values())), torch.tensor(list(self.tag.keys()))),

            ('user', 'play', 'game'): (self.user_game[:, 0].long(), self.user_game[:, 1].long()),

            ('game', 'played by', 'user'): (self.user_game[:, 1].long(), self.user_game[:, 0].long())
        }
        # Create heterogenous graph
        graph = dgl.heterograph(graph_data)

        ls_feature = []

        # Get app info of game nodes and store in ls_feature
        for node in graph.nodes('game'):
            node = int(node)
            if node in self.app_info:
                ls_feature.append(self.app_info[node])

        ls_feature = np.vstack(ls_feature)
        feature_mean = ls_feature.mean(0) # Compute mean vertically (e.g. compute the mean of each feature)
        
        ls_feature = []

        # Store game node's app info in ls_feature, if game ID is not in app info, use computed mean instead
        count_total = 0
        count_without_feature = 0
        for node in graph.nodes('game'):
            count_total += 1
            node = int(node)
            if node in self.app_info:
                ls_feature.append(self.app_info[node])
            else:
                count_without_feature += 1
                ls_feature.append(feature_mean)
        logger.info("total game number is {}, games without features number is {}".format(count_total,count_without_feature ))

        # Add app info to game nodes, which have been stored in ls_feature
        graph.nodes['game'].data['h'] = torch.tensor(np.vstack(ls_feature))

        # Add dwelling time to edges with type "play" and "played by" (1D Tensor Array consisting of dwelling time)
        graph.edges['play'].data['time'] = self.user_game[:, 2]
        graph.edges['played by'].data['time'] = self.user_game[:, 2]

        # Add percentile to edges
        graph.edges['play'].data['percentile'] = self.user_game[:, 3]
        graph.edges['played by'].data['percentile'] = self.user_game[:, 3]
        self.graph = graph
    
    def read_app_info(self, path):
        """
            Reads the appInfo txt file. Performs one hot encoding on the Type column, 
            replaces unrated metacritic scores with the global mean. Normalizes certain columns. Also normalized numerical values 
            and changes required_age to NaN if required age is 0.

            :return: Dictionary, keys = mapped App ID, 
                                 values = numpy array containing normalized price, metacritic score, required age, isMulti, OHE, and days elapsed.
        """
        dic = {}
        df = pd.read_csv(path, header = None)

        # Do OneHotEncoding on Type column (e.g. game, dlc, mod)
        df = pd.get_dummies(df, columns = [2])

        df_time = pd.to_datetime(df.iloc[:, 3])
        date_end = pd.to_datetime('2013-06-25')
        time_sub = date_end - df_time
        time_sub = time_sub.dt.days
        df = pd.concat([df, time_sub], axis = 1)

        # Get column indexes beyond released date (ALL numericals e.g. price, metacritic score, required age, isMulti, OHE, days elapsed)
        #       column index = [2,4,5,6,7,8,9,10,11]
        column_num = len(df.columns)
        column_index = [2]
        column_index.extend([i for i in range(4, column_num)])

        logger.info("begin feature engineering")
        # Replace no metacritic scores to NaN, then replace NaN to mean of all metacritic scores
        df.iloc[:, 4].replace(to_replace = -1, value = np.nan, inplace = True)
        mean = df.iloc[:, 4].mean()
        df.iloc[:, 4].replace(to_replace = np.nan, value = mean, inplace = True)

        # Normalize 2-price, 4-metacritic scores, 5-required age, and 11-days elapsed
        columns_norm = [2, 4, 5, 11]
        mean = df.iloc[:, columns_norm].mean()
        std = df.iloc[:, columns_norm].std()
        df.iloc[:, columns_norm] = (df.iloc[:, columns_norm] - mean) / std

        for i in range(len(df)):
            app_id = self.app_id_mapping[str(df.iloc[i, 0])]
            feature = df.iloc[i, column_index].to_numpy()
            feature = feature.astype(np.float64)
            dic[app_id] = feature
        dic['feature_num'] = len(feature)
        return dic
    
    def read_mapping(self, path):
        """
            Used to read the Developers, Genres, and Publishers txt files. Only reads the first developer/genre/publisher found.
            Sample return value: {0: 0, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 3}

            :return: Dictionary, where keys = mapped AppIDs, values = mapped Developer/Genre/Publisher ID (first record found)
        """
        mapping = {}
        with open(path, 'r') as f:
            for line in f:
                line = line.strip().split(',')
                # If app ID not yet in mapping
                if line[0] not in mapping:
                    if line[1] != '':
                        mapping[self.app_id_mapping[line[0]]] = line[1]
        mapping_value2id = {}
        count = 0
        # Map values too (e.g. Valve = 0, SEGA = 1)
        for value in mapping.values():
            if value not in mapping_value2id:
                mapping_value2id[value] = count
                count += 1
        for key in mapping:
            mapping[key] = mapping_value2id[mapping[key]]
        # print(mapping)
        return mapping
    
    def read_play_time_rank(self, game_path, time_path):
        """
            Reads train_game and train_time text files. Uses the generate_percentile to generate rankings for dwelling time.
        """
        ls = [] # list that will store [user, game, dwelling time, ranking] values
        dic_game = {}
        dic_time = {}
        with open(game_path, 'r') as f_game:
            with open(time_path, 'r') as f_time:
                lines_game = f_game.readlines()
                lines_time = f_time.readlines()
                # For each user
                for i in tqdm(range(len(lines_game))):
                    line_game = lines_game[i].strip().split(',')
                    line_time = lines_time[i].strip().split(',')
                    user = self.user_id_mapping[line_game[0]]
                    dic_game[user] = []

                    # For each game owned by user
                    for j in range(1, len(line_game)):
                        game = self.app_id_mapping[line_game[j]]
                        dic_game[user].append(game)
                        time = line_time[j]
                        # Replace \N dwelling time values with 0
                        if time == r'\N':
                            ls.append([user, game, 0])
                        else:
                            ls.append([user, game, float(time)])
        logger.info('generate percentiles')
        # Append new percentile column to each [user, game, dwelling time] list 
        ls = self.generate_percentile(ls)

        # print(torch.tensor(ls))
        # print(torch.tensor(ls).shape)
        return torch.tensor(ls), dic_game

    def generate_percentile(self, ls):
        """
            Generates a percentile score for a given [user, game, dwelling time] list.
            @param ls (2D List) - rows = [mapped user ID, mapped game ID, dwelling time].
            
            :return: 2D List, appended percentage score for each row
        """
        dic = {}   
        # Create dictionary where keys = mapped game IDs, values = list of dwelling times
        for ls_i in ls:
            if ls_i[1] in dic:
                dic[ls_i[1]].append(ls_i[2])
            else:
                dic[ls_i[1]] = [ls_i[2]]
        # Sort dwelling times for each game, remove repeating values
        for key in tqdm(dic):
            dic[key] = sorted(list(set(dic[key])))

        dic_percentile = {}
        for key in tqdm(dic):
            dic_percentile[key] = {}
            length = len(dic[key])
            # Create percentile for each time of a given game
            # Formula: i / number of dwelling times, where i = position of element in list
            for i in range(len(dic[key])):
                time = dic[key][i]
                dic_percentile[key][time] = (i + 1) / length

        # Append new percentiles of game's dwelling time to ls dictionary as a new column
        for i in tqdm(range(len(ls))):
            ls[i].append(dic_percentile[ls[i][1]][ls[i][2]])
        return ls

    def read_friends(self, path):
        """
            Reads the friends.txt file. Creates a 2D Tensor with size (n, 2), where n is the number of user-user friends.
            Each row represents a friend relationship using mapped user IDs 
                Ex: [1, 2], user 1 is friends with user 2
            
            :return: Tensor, size (n, 2) representing list of friends.
        """
        ls = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip().split(',')
                ls.append([self.user_id_mapping[line[0]], self.user_id_mapping[line[1]]])
        return torch.tensor(ls)

    def read_dic_user_game(self, game_path):
        """
            Reads the train_game.txt file. Used when the graph.bin file has already been generated and 
            the program does not need to go through the entire process() method.
            
            :return: Dictionary, keys = mapped user IDs, values = list of mapped game IDs owner by user
        """
        dic_game = {}
        with open(game_path, 'r') as f_game:
            lines_game = f_game.readlines()
            for i in tqdm(range(len(lines_game))):
                line_game = lines_game[i].strip().split(',')
                user = self.user_id_mapping[line_game[0]]

                dic_game[user] = []
                for j in range(1, len(line_game)):
                    game = self.app_id_mapping[line_game[j]]
                    dic_game[user].append(game)
        return dic_game


    def read_play_time(self, path):
        """
            NOTE: Not used in this project, but was originally intended for user_game.txt - a variation of train_game.txt and train_time.txt.
            Reads the user_game.txt file, returns a Tensor with size (n, 4).

            :return: Tensor, size (n,4) where each row contains [mapped user ID, mapped game ID, dwelling time, percentile]
        """
        ls = []
        with open(path, 'r', encoding = 'utf8') as f:
            for line in f:
                line = line.strip().split(',')
                if line[-1] == r'\N':
                    ls.append([self.user_id_mapping[line[0]], self.app_id_mapping[line[1]], 0])
                else:
                    ls.append([self.user_id_mapping[line[0]], self.app_id_mapping[line[1]], int(line[2])])
        logger.info('generate percentiles')
        ls = self.generate_percentile(ls)
        return torch.tensor(ls)

    def build_dataloader(self, args, graph):
        """
            Build EdgeDataLoader from graph using MultiLayerFullNeighbor sample and custom NegativeSampler.

            :return: EdgeDataLoader object containing {inputNodes, sub_graph, neg_sub_graph, blocks}
        """
        # Create sampler that takes messages from all neighbors
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.layers)
        
        # Generate unique ids for each edge with type 'play'
        train_id = torch.tensor([i for i in range(graph.edges(etype = 'play')[0].shape[0])], dtype = torch.long)

        dataloader = dgl.dataloading.EdgeDataLoader(
            graph, {('user', 'play', 'game'): train_id},
            sampler, negative_sampler = NegativeSampler(self.dic_user_game), batch_size = args.batch_size, shuffle = True, num_workers = 2
        )

        return dataloader
