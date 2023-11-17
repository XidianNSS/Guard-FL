
import logging
import os
import torch.nn as nn
import torch.nn.init as init
import torchvision
from torch.utils.data import Dataset, TensorDataset, ConcatDataset
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN

# Data manipulation
import pandas as pd  # for data manipulation
import numpy as np  # for data manipulation
import torch
# Visualization
import plotly.express as px  # for data visualization
import matplotlib.pyplot as plt  # for showing handwritten digits

# UMAP dimensionality reduction
from umap import UMAP

logger = logging.getLogger(__name__)


#######################
# TensorBaord setting #
#######################
def launch_tensor_board(log_path, port, host):
    """Function for initiating TensorBoard.

    Args:
        log_path: Path where the log is stored.
        port: Port number used for launching TensorBoard.
        host: Address used for launching TensorBoard.
    """
    os.system(f"tensorboard --logdir={log_path} --port={port} --host={host}")
    return True


#########################
# Weight initialization #
#########################
def init_weights(model, init_type, init_gain):
    """Function for initializing network weights.

    Args:
        model: A torch.nn instance to be initialized.
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal).
        init_gain: Scaling factor for (normal | xavier | orthogonal).
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError(f'[ERROR] ...initialization method [{init_type}] is not implemented!')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    model.apply(init_func)


def init_net(model, init_type, init_gain, gpu_ids):
    """Function for initializing network weights.

    Args:
        model: A torch.nn.Module to be initialized
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal)l
        init_gain: Scaling factor for (normal | xavier | orthogonal).
        gpu_ids: List or int indicating which GPU(s) the network runs on. (e.g., [0, 1, 2], 0)

    Returns:
        An initialized torch.nn.Module instance.
    """
    # gpu_ids=[0]
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        model.to(gpu_ids[0])
        model = nn.DataParallel(model, gpu_ids)
    init_weights(model, init_type, init_gain)
    return model


#################
# Dataset split #
#################
class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms."""

    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def posion(self):
        pass

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        if self.transform:
            x = self.transform(x.numpy().astype(np.uint8))
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

np.random.seed(42)
def dirichlet_split_noniid(train_labels, alpha, n_clients):
    '''
    参数为 alpha 的 Dirichlet 分布将数据索引划分为 n_clients 个子集
    '''

    n_classes = train_labels.max() + 1  # 总类别数目


    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)

    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]


    client_idcs = [[] for _ in range(n_clients)]


    for c, fracs in zip(class_idcs, label_distribution):

        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs


def create_datasets(data_path, dataset_name, num_clients, num_shards, iid):
    """Split the whole dataset in IID or non-IID manner for distributing to clients."""
    # dataset_name = dataset_name.upper()
    # get dataset from torchvision.datasets if exists
    if hasattr(torchvision.datasets, dataset_name):
        # set transformation differently per dataset
        if dataset_name in ["CIFAR10"]:
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),  # 转换为tensor
                    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 归一化
                ]
            )

        elif dataset_name in ["MNIST","FashionMNIST"]:
            transform = torchvision.transforms.ToTensor()

        # prepare raw training & test datasets
        training_dataset = torchvision.datasets.__dict__[dataset_name](
            root=data_path,
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = torchvision.datasets.__dict__[dataset_name](
            root=data_path,
            train=False,
            download=True,
            transform=transform
        )
    else:
        # dataset not found exception
        error_message = f"...dataset \"{dataset_name}\" is not supported or cannot be found in TorchVision Datasets!"
        raise AttributeError(error_message)

    # unsqueeze channel dimension for grayscale image datasets
    if training_dataset.data.ndim == 3:  # convert to NxHxW -> NxHxWx1
        training_dataset.data.unsqueeze_(3)
    num_categories = np.unique(training_dataset.targets).shape[0]

    if "ndarray" not in str(type(training_dataset.data)):
        training_dataset.data = np.asarray(training_dataset.data)
    if "list" not in str(type(training_dataset.targets)):
        training_dataset.targets = training_dataset.targets.tolist()

    # split dataset according to iid flag
    if iid:
        # shuffle data
        shuffled_indices = torch.randperm(len(training_dataset))
        training_inputs = training_dataset.data[shuffled_indices]
        training_labels = torch.Tensor(training_dataset.targets)[shuffled_indices]

        # partition data into num_clients
        split_size = len(training_dataset) // num_clients
        split_datasets = list(
            zip(
                torch.split(torch.Tensor(training_inputs), split_size),
                torch.split(torch.Tensor(training_labels), split_size)
            )
        )


        # finalize bunches of local_noniid_2nn_mnist_100_10_3p-iid-3-client datasets
        local_datasets = [
            CustomTensorDataset(local_dataset, transform=transform)
            for local_dataset in split_datasets
        ]
    else:
        N_CLIENTS = num_clients
        DIRICHLET_ALPHA = 0.5
        training_inputs = torch.Tensor(training_dataset.data)
        train_labels = np.array(training_dataset.targets)
        num_cls = len(training_dataset.classes)
        client_idcs = dirichlet_split_noniid(train_labels, alpha=DIRICHLET_ALPHA, n_clients=N_CLIENTS)

        local_datasets = []
        for client_index in client_idcs:
            # client_labels_tensor=torch.Tensor(client_labels)
            # print(client_labels_tensor)
            client_index_list = client_index.tolist()
            temp_data = []
            temp_labels = []
            for i in client_index_list:
                temp_data.append(training_inputs[i].tolist())
                temp_labels.append(train_labels[i].tolist())
            temp_data = torch.tensor(temp_data)
            temp_labels = torch.Tensor(temp_labels).long()
            temp = CustomTensorDataset(
                (
                    temp_data,
                    temp_labels
                ),
                transform=transform
            )
            local_datasets.append(temp)

    return local_datasets, test_dataset



reducer = UMAP(n_neighbors=5,
               # default 15, The size of local_noniid_2nn_mnist_100_10_3p-iid-3-client neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
               n_components=3,  # default 2, The dimension of the space to embed into.
               metric='cosine',
               # default 'euclidean correlation chebyshev', The metric to use to compute distances in high dimensional space.
               n_epochs=150000,
               # default None, The number of training epochs to be used in optimizing the low dimensional embedding. Larger values result in more accurate embeddings.
               learning_rate=1.0,  # default 1.0, The initial learning rate for the embedding optimization.
               init='spectral',
               # default 'spectral（谱聚类）', How to initialize the low dimensional embedding. Options are: {'spectral', 'random', A numpy array of initial embedding positions}.
               min_dist=0.05,  # default 0.1, The effective minimum distance between embedded points.（算a,b用的）
               spread=1.0,
               # default 1.0, The effective scale of embedded points. In combination with ``min_dist`` this determines how clustered/clumped the embedded points are.
               low_memory=False,
               # default False, For some datasets the nearest neighbor computation can consume a lot of memory. If you find that UMAP is failing due to memory constraints consider setting this option to True.
               set_op_mix_ratio=1.0,
               # default 1.0, The value of this parameter should be between 0.0 and 1.0; a value of 1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy intersection.
               local_connectivity=1,
               # default 1, The local_noniid_2nn_mnist_100_10_3p-iid-3-client connectivity required -- i.e. the number of nearest neighbors that should be assumed to be connected at a local_noniid_2nn_mnist_100_10_3p-iid-3-client level.
               repulsion_strength=1.0,
               # default 1.0, Weighting applied to negative samples in low dimensional embedding optimization.
               negative_sample_rate=6,
               # default 5, Increasing this value will result in greater repulsive force being applied, greater optimization cost, but slightly more accuracy.
               transform_queue_size=8.0,
               # default 4.0, Larger values will result in slower performance but more accurate nearest neighbor evaluation.
               a=None,
               # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
               b=None,
               # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
               random_state=42,
               # default: None, If int, random_state is the seed used by the random number generator;
               metric_kwds=None,
               # default None) Arguments to pass on to the metric, such as the ``p`` value for Minkowski distance.
               angular_rp_forest=False,
               # default False, Whether to use an angular random projection forest to initialise the approximate nearest neighbor search.
               target_n_neighbors=-1,
               # default -1, The number of nearest neighbors to use to construct the target simplcial set. If set to -1 use the ``n_neighbors`` value.
               # target_metric='categorical', # default 'categorical', The metric used to measure distance for a target array is using supervised dimension reduction. By default this is 'categorical' which will measure distance in terms of whether categories match or are different.
               # target_metric_kwds=None, # dict, default None, Keyword argument to pass to the target metric when performing supervised dimension reduction. If None then no arguments are passed on.
               # target_weight=0.5, # default 0.5, weighting factor between data topology and target topology.
               transform_seed=42,
               # default 42, Random seed used for the stochastic aspects of the transform operation.
               verbose=False,  # default False, Controls verbosity of logging.
               unique=False,
               # default False, Controls if the rows of your data should be uniqued before being embedded.
               )


def PCA_svd(X, k, center=True):
    n = X.size()[0]
    ones = torch.ones(n).view([n, 1])
    h = ((1 / n) * torch.mm(ones, ones.t())) if center else torch.zeros(n * n).view([n, n])
    H = torch.eye(n) - h
    # H = H.cuda()
    X_center = torch.mm(H.double(), X.double())
    u, s, v = torch.svd(X_center)
    components = v[:k].t()
    # explained_variance = torch.mul(s[:k], s[:k])/(n-1)
    return components




def plot_embedding_3d(X, title=None):
    # 坐标缩放到[0,1]区间
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)
    # X=X*3
    # 降维后的坐标为（X[i, 0], X[i, 1],X[i,2]），在该位置画出对应的digits
    fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax = Axes3D(fig)
    for i in range(X.shape[0]):
        if i % 10 == 0:
            ax.text(X[i, 0], X[i, 1], X[i, 2], str(1),
                    # color=plt.cm.Set3(i / 5.),
                    color="b",
                    fontdict={'weight': 'bold', 'size': 9})
        elif i % 10 == 1:
            ax.text(X[i, 0], X[i, 1], X[i, 2], str(2),
                    color='g',
                    fontdict={'weight': 'bold', 'size': 9})
        elif i % 10 == 2:
            ax.text(X[i, 0], X[i, 1], X[i, 2], str(3),
                    color='r',
                    fontdict={'weight': 'bold', 'size': 9})
        elif i % 10 == 3:
            ax.text(X[i, 0], X[i, 1], X[i, 2], str(4),
                    color='c',
                    fontdict={'weight': 'bold', 'size': 9})
        elif i % 10 == 4:
            ax.text(X[i, 0], X[i, 1], X[i, 2], str(5),
                    color='m',
                    fontdict={'weight': 'bold', 'size': 9})
        elif i % 10 == 5:
            ax.text(X[i, 0], X[i, 1], X[i, 2], str(6),
                    color='y',
                    fontdict={'weight': 'bold', 'size': 9})
        elif i % 10 == 6:
            ax.text(X[i, 0], X[i, 1], X[i, 2], str(7),
                    color='k',
                    fontdict={'weight': 'bold', 'size': 9})
        elif i % 10 == 7:
            ax.text(X[i, 0], X[i, 1], X[i, 2], str(8),
                    color='w',
                    fontdict={'weight': 'bold', 'size': 9})
        elif i % 10 == 8:
            ax.text(X[i, 0], X[i, 1], X[i, 2], str(9),
                    color="tab:orange",
                    fontdict={'weight': 'bold', 'size': 9})
        elif i % 10 == 9:
            ax.text(X[i, 0], X[i, 1], X[i, 2], str(10),
                    # color=plt.cm.Set3(i / 10.),
                    fontdict={'weight': 'bold', 'size': 9})

    if title is not None:
        plt.title(title)
    plt.show()


def chart(X, y):
    # --------------------------------------------------------------------------#
    # This section is not mandatory as its purpose is to sort the data by label
    # so, we can maintain consistent colors for digits across multiple graphs

    # Concatenate X and y arrays
    arr_concat = np.concatenate((X, y.reshape(y.shape[0], 1)), axis=1)
    # Create a Pandas dataframe using the above array
    df = pd.DataFrame(arr_concat, columns=['x', 'y', 'z', 'label'])
    # Convert label data type from float to integer
    df['label'] = df['label'].astype(int)
    # Finally, sort the dataframe by label
    df.sort_values(by='label', axis=0, ascending=True, inplace=True)
    # --------------------------------------------------------------------------#

    # Create a 3D graph
    fig = px.scatter_3d(df, x='x', y='y', z='z', color=df['label'].astype(str), height=900, width=950)

    # Update chart looks
    fig.update_layout(title_text='UMAP',
                      showlegend=True,
                      legend=dict(orientation="h", yanchor="top", y=0, xanchor="center", x=0.5),
                      scene_camera=dict(up=dict(x=0, y=0, z=1),
                                        center=dict(x=0, y=0, z=-0.1),
                                        eye=dict(x=1.5, y=-1.4, z=0.5)),
                      margin=dict(l=0, r=0, b=0, t=0),
                      scene=dict(xaxis=dict(backgroundcolor='white',
                                            color='black',
                                            gridcolor='#f0f0f0',
                                            title_font=dict(size=10),
                                            tickfont=dict(size=10),
                                            ),
                                 yaxis=dict(backgroundcolor='white',
                                            color='black',
                                            gridcolor='#f0f0f0',
                                            title_font=dict(size=10),
                                            tickfont=dict(size=10),
                                            ),
                                 zaxis=dict(backgroundcolor='lightgrey',
                                            color='black',
                                            gridcolor='#f0f0f0',
                                            title_font=dict(size=10),
                                            tickfont=dict(size=10),
                                            )))
    # Update marker size
    fig.update_traces(marker=dict(size=3, line=dict(color='black', width=0.1)))

    fig.show()

#
# def chart(X, y, round):
#     # --------------------------------------------------------------------------#
#     # This section is not mandatory as its purpose is to sort the data by label
#     # so, we can maintain consistent colors for digits across multiple graphs
#
#     # Concatenate X and y arrays
#     arr_concat = np.concatenate((X, y.reshape(y.shape[0], 1)), axis=1)
#     print(arr_concat)
#     arr_concat = np.concatenate((arr_concat, round.reshape(y.shape[0], 1)), axis=1)
#     print(arr_concat)
#     # Create a Pandas dataframe using the above array
#     df = pd.DataFrame(arr_concat, columns=['x', 'y', 'z', 'label', 'round'])
#     # Convert label data type from float to integer
#     df['label'] = df['label'].astype(int)
#     df['round'] = df['round'].astype(int)
#     # Finally, sort the dataframe by label
#     df.sort_values(by='label', axis=0, ascending=True, inplace=True)
#     # --------------------------------------------------------------------------#
#
#     # Create a 3D graph
#     fig = px.scatter_3d(df, x='x', y='y', z='z', color=df['label'].astype(str), text=df['round'].astype(str),
#                         height=900, width=950)
#
#     # Update chart looks
#     fig.update_layout(title_text='UMAP',
#                       showlegend=True,
#                       legend=dict(orientation="h", yanchor="top", y=0, xanchor="center", x=0.5),
#                       scene_camera=dict(up=dict(x=0, y=0, z=1),
#                                         center=dict(x=0, y=0, z=-0.1),
#                                         eye=dict(x=1.5, y=-1.4, z=0.5)),
#                       margin=dict(l=0, r=0, b=0, t=0),
#                       scene=dict(xaxis=dict(backgroundcolor='white',
#                                             color='black',
#                                             gridcolor='#f0f0f0',
#                                             title_font=dict(size=10),
#                                             tickfont=dict(size=10),
#                                             ),
#                                  yaxis=dict(backgroundcolor='white',
#                                             color='black',
#                                             gridcolor='#f0f0f0',
#                                             title_font=dict(size=10),
#                                             tickfont=dict(size=10),
#                                             ),
#                                  zaxis=dict(backgroundcolor='lightgrey',
#                                             color='black',
#                                             gridcolor='#f0f0f0',
#                                             title_font=dict(size=10),
#                                             tickfont=dict(size=10),
#                                             )))
#     # Update marker size
#     fig.update_traces(marker=dict(size=3, line=dict(color='black', width=0.1)))
#
#     fig.show()



def get_mse():
    re_assemble = []

    for roundnum in range(1, 99):  # 控制将多少轮的数据加在一起
        matrix = []
        result = []

        temp_round_path = "round_" + str(roundnum) + str("\\")
        for i in range(1, 11):
            temp_model_path = str(i) + "_model.pth"
            model = torch.load("E:\\BaiduNetdiskDownload\\Fed_pca\local\\" + temp_round_path + temp_model_path)
            v = torch.nn.utils.parameters_to_vector(model.parameters()).tolist()
            # print(type(matrix))
            matrix.append(v)

        matrix = np.array(matrix)
        umap_result = reducer.fit_transform(matrix)

        umap_result = torch.tensor(umap_result)

        avg_parm = umap_result.sum(dim=0)
        # print(avg_parm)
        avg_parm = torch.divide(avg_parm, 10)
        # print(avg_parm)
        cos_similary = torch.nn.functional.cosine_similarity(umap_result, avg_parm)
        # print(cos_similary)
        result.append(cos_similary.argmin().item() + 1)
        cos_similary[cos_similary.argmin().item()] = 1
        result.append(cos_similary.argmin().item() + 1)
        cos_similary[cos_similary.argmin().item()] = 1
        result.append(cos_similary.argmin().item() + 1)
        print(result)
        re_assemble.extend(result)
    prob = (re_assemble.count(1) + re_assemble.count(2) + re_assemble.count(3)) / len(re_assemble)
    print(prob)


def Kmeans_cluster():
    X_assemble = []

    for roundnum in range(1, 100):  # 控制将多少轮的数据加在一起
        matrix = []
        result = []
        temp_round_path = "round_" + str(roundnum) + str("\\")
        for i in range(1, 11):
            temp_model_path = str(i) + "_model.pth"
            model = torch.load("E:\\BaiduNetdiskDownload\\Fed_pca\local\\" + temp_round_path + temp_model_path)
            v = torch.nn.utils.parameters_to_vector(model.parameters()).tolist()
            # print(type(matrix))
            matrix.append(v)

        matrix = np.array(matrix)
        umap_result = reducer.fit_transform(matrix)
        # print(umap_result.shape)
        y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(umap_result)
        print(y_pred)
        count_0 = 0
        count_1 = 0


def umap_kmeans(matrix, sampled_client_indices):
    benign_result = []
    # matrix=matrix.to('cpu')
    umap_result = reducer.fit_transform(matrix)

    tensors=umap_result
    B_n_values = []
    Slope_result = []
    intercept_result = []

    # 对每个维度分别进行升序排序，并计算B_n
    for n in range(3):
        # 从每个张量中提取第n个维度的值并排序
        values_n = [tensor[n] for tensor in tensors]
        values_n = torch.tensor(values_n)
        sorted_values, indices = torch.sort(values_n)
        temp = [i for i in range(10)]
        sorted_indices = torch.tensor(temp)

        # 计算B(i, j)的值
        B_values = []
        for i in range(10):
            for j in range(10):
                if i != j:
                    B_ij = (sorted_values[j] - sorted_values[i]) / (sorted_indices[j] - sorted_indices[i])
                    B_values.append(B_ij)
            median_B_values = torch.median(torch.stack(B_values), dim=0).values
            B_values.clear()
            B_n_values.append(median_B_values)

        Slope = torch.median(torch.stack(B_n_values), dim=0).values
        Slope_result.append(Slope)
        intercept_vector = []
        for k in range(len(values_n)):
            intercept_vector.append(sorted_values[k] - Slope * sorted_indices[k])
        intercept = torch.median(torch.stack(intercept_vector), dim=0).values
        intercept_result.append(intercept)

        x = sorted_indices
        y = Slope * x + intercept

        # plt.figure()
        # plt.scatter(x, sorted_values)
        # plt.plot(x, y, label=f'Line {i + 1}', color='r')
        # plt.xlabel('X-axis')
        # plt.ylabel('Y-axis')
        # plt.title(f'Line {i + 1}')
        # plt.legend()
        # plt.show()



    sampled_client_indices=np.array(sampled_client_indices)
    # chart(umap_result,sampled_client_indices)
    y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(umap_result)
    v0 = []
    v1 = []
    for i in range(len(y_pred)):
        if y_pred[i] == 0:
            v0.append(sampled_client_indices[i])
        else:
            v1.append(sampled_client_indices[i])
    print(y_pred)
    y_pred_list=y_pred.tolist()
    print(sampled_client_indices)
    if len(v0) < len(v1):
        benign_result = v1
    elif len(v0) > len(v1):
        benign_result = v0
    else:
        benign_result =[4,5,6,7,8,9,10]
    print(benign_result)


    return benign_result,y_pred_list


