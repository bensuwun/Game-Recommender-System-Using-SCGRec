import sys
import torch
import logging
logging.basicConfig(stream = sys.stdout, level = logging.INFO)
import dgl
from dgl.data import DGLDataset

class Dataloader_item_graph(DGLDataset):
    """
        Initializes the item graph or the game context graph (5.1 in Liangwei et al's paper.)
    """
    def __init__(self, graph, app_id_path, publisher_path, developer_path, genre_path):
        self.app_id_path = app_id_path
        self.publisher_path = publisher_path
        self.developer_path = developer_path
        self.genre_path = genre_path

        # Retrieve co-features of games
        logging.info("reading item graph")
        self.app_id_mapping = self.read_id_mapping(self.app_id_path)
        self.publisher = self.read_mapping(self.publisher_path)
        self.developer = self.read_mapping(self.developer_path)
        self.genre = self.read_mapping(self.genre_path)

        # Initialize game context graph from co-features
        graph_data = {
            ('game', 'co_publisher', 'game'): self.publisher,
            ('game', 'co_developer', 'game'): self.developer,
            ('game', 'co_genre', 'game'): self.genre
        }
        self.graph = dgl.heterograph(graph_data)

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

    
