import sys
import dgl
import dgl.function as fn
sys.path.append('../')
import os
import multiprocessing as mp
# mp.set_start_method('spawn')
from tqdm import tqdm
import pdb
import random
import numpy as np
import torch
import torch.nn as nn
import utils.logs.log_config as log_config
import logging
logger = logging.getLogger(__name__)
log_config.SetDefaultConfig(logger)

# Imports from utils directory
from utils.parser import parse_args
from utils.dataloader_steam import Dataloader_steam
from utils.dataloader_item_graph import Dataloader_item_graph

from datetime import datetime
import matplotlib.pyplot as plt

# Not found in repository, originally used for performance comparison 
# from models.RGCNModel_steam_rank import RGCNModel_steam_rank

from models.Predictor import HeteroDotProductPredictor
from models.model import Proposed_model

from utils.metrics import MAE, RMSE, ndcg_at_k, recall_at_k, hit_at_k, precision_at_k

def validate(train_mask, dic, h, ls_k):
    """
        :param: train_mask () - 
        :param: dic (Dictionary) - the validation data {userId: [gameIds owned]}
        :param: h () - model/graph
        :param: ls_k (List) - list of possible folds for k-fold validation
    """
    users = torch.tensor(list(dic.keys())).long()
    user_embedding = h['user'][users]
    game_embedding = h['game']
    rating = torch.mm(user_embedding, game_embedding.t())
    rating[train_mask] = -float('inf')

    valid_mask = torch.zeros_like(train_mask)
    for i in range(users.shape[0]):
        user = int(users[i])
        items = torch.tensor(dic[user])
        valid_mask[i, items] = 1

    _, indices = torch.sort(rating, descending = True)
    ls = [valid_mask[i,:][indices[i, :]] for i in range(valid_mask.shape[0])]
    result = torch.stack(ls).float()

    res = []
    for k in ls_k:
        discount = (torch.tensor([i for i in range(k)]) + 2).log2()
        ideal, _ = result.sort(descending = True)
        # TODO: Bugfix tensor size of ideal (7) and discount (10)
        idcg = (ideal[:, :k] / discount).sum(dim = 1) 
        dcg = (result[:, :k] / discount).sum(dim = 1)
        ndcg = torch.mean(dcg / idcg)

        recall = torch.mean(result[:, :k].sum(1) / result.sum(1))
        hit = torch.mean((result[:, :k].sum(1) > 0).float())
        precision = torch.mean(result[:, :k].mean(1))

        logger_result = "For k = {}, ndcg = {}, recall = {}, hit = {}, precision = {}".format(k, ndcg, recall, hit, precision)
        logger.info(logger_result)
        res.append(logger_result)
    return ndcg, str(res)


def construct_negative_graph(graph, etype):
    utype, _ , vtype = etype
    src, _ = graph.edges(etype = etype)
    dst = torch.randint(graph.num_nodes(vtype), size = src.shape)
    return dgl.heterograph({etype: (src, dst)}, num_nodes_dict = {ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes})

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    args = parse_args()
    setup_seed(2020)

    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'

    # Path: current working directory
    path = 'steam_data'

    user_id_path = path + '/users.txt'
    app_id_path = path + '/app_id.txt'
    app_info_path = path + '/App_ID_Info.txt'
    friends_path = path + '/friends.txt'
    developer_path = path + '/Games_Developers.txt'
    publisher_path = path + '/Games_Publishers.txt'
    genres_path = path + '/Games_Genres.txt'
    country_path = path + '/user_country.txt'

    # Build user-item and user-user heterogeneous
    DataLoader = Dataloader_steam(args, path, user_id_path, app_id_path, app_info_path, friends_path, developer_path, publisher_path, genres_path, country_path)

    graph = DataLoader.graph
    # Build item-item heterogeneous graph
    DataLoader_item = Dataloader_item_graph(graph, app_id_path, publisher_path, developer_path, genres_path)

    graph_item = DataLoader_item.graph

    graph_social = dgl.edge_type_subgraph(graph, [('user', 'friend of', 'user')])

    graph = dgl.edge_type_subgraph(graph, [('user', 'play', 'game'), ('game', 'played by', 'user')])
    graph.update_all(fn.copy_edge('percentile', 'm'), fn.sum('m', 'total'), etype = 'played by')
    graph.apply_edges(func = fn.e_div_v('percentile', 'total', 'weight'), etype = 'played by')

    # Load validation set
    valid_user = list(DataLoader.valid_data.keys())
    train_mask = torch.zeros(len(valid_user), graph.num_nodes('game'))
    for i in range(len(valid_user)):
        user = valid_user[i]
        item_train = torch.tensor(DataLoader.dic_user_game[user])
        train_mask[i, :][item_train] = 1
    train_mask = train_mask.bool()

    model = Proposed_model(args, graph, graph_item)

    predictor = HeteroDotProductPredictor()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr = args.lr)

    stop_count = 0
    ndcg_val_best = 0
    ls_k = args.k

    total_epoch = 0
    loss_values = []
    for epoch in range(args.epoch):
        model.train()
        graph_neg = construct_negative_graph(graph, ('user', 'play', 'game'))
        h = model.forward(graph, graph_item, graph_social)

        score = predictor(graph, h, ('user', 'play', 'game'))
        score_neg = predictor(graph_neg, h, ('user', 'play', 'game'))
        # loss = tensor(loss, requires_grad=True)
        loss = -(score - score_neg).sigmoid().log().sum()
        loss_values.append(loss.item())
        logger.info("loss = {}".format(loss))
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_epoch += 1

        # score, h = model.forward_all(graph, 'play')
        logger.info('Epoch {}'.format(epoch))
        if total_epoch > 1:
            model.eval()
            logger.info("begin validation")

            ndcg, _ = validate(train_mask, DataLoader.valid_data, h, ls_k)

            if ndcg > ndcg_val_best:
                ndcg_val_best = ndcg
                stop_count = 0
                logger.info("begin test")
                ndcg_test, test_result = validate(train_mask, DataLoader.test_data, h, ls_k)
            else:
                stop_count += 1
                if stop_count > args.early_stop:
                    logger.info('early stop')
                    break

    logger.info('Final ndcg {}'.format(ndcg_test))
    logger.info(test_result)

    # Files Prefix:  Month-Day-Year Hour.Min AM/PM
    prefix = datetime.today().strftime('%m-%d-%Y %H.%M %p')

    # Save Model (Note: Create folder named "saved" inside models dir)
    if args.save_model == True:
        torch.save(model.state_dict(), f'models/saved/{prefix}_model.pt')

    # Plot Loss Values
    plt.plot(loss_values)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(f'models/saved/{prefix}_loss_graph.png')
    plt.show()