import os
import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from .metrics import masked_mape_np
from scipy.sparse.linalg import eigs


import pickle

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj(adj_filename, num_of_vertices=None):
    """加载邻接矩阵
    
    Args:
        adj_filename: 邻接矩阵文件路径 (CSV格式)
        num_of_vertices: 站点数量（可选，如果不提供则从文件推断）
    
    Returns:
        adj: 邻接矩阵 (N, N)
    """
    import pandas as pd
    import numpy as np

    # 读取邻接矩阵文件
    if adj_filename.endswith('.csv'):
        # CSV格式：直接读取为矩阵
        adj = np.loadtxt(adj_filename, delimiter=',')
        
        # 验证矩阵是否为方阵
        if len(adj.shape) != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError(f"邻接矩阵必须是方阵，当前形状: {adj.shape}")
        
        # 如果提供了num_of_vertices，验证一致性
        if num_of_vertices is not None:
            if adj.shape[0] != num_of_vertices:
                print(f"⚠️  警告: 配置文件中的num_of_vertices={num_of_vertices}与邻接矩阵大小{adj.shape[0]}不一致")
                print(f"   将使用邻接矩阵的实际大小: {adj.shape[0]}")
        
        print(f"✅ 邻接矩阵加载成功: shape={adj.shape}")
        return adj
    
    elif adj_filename.endswith('.pkl') or adj_filename.endswith('.pickle'):
        # Pickle格式：原有的逻辑
        df = pd.read_csv(adj_filename, header=None)
        
        # 动态获取节点数
        if num_of_vertices is None:
            # 从数据中推断最大节点ID
            num_nodes = max(df[0].max(), df[1].max()) + 1
        else:
            num_nodes = num_of_vertices
            
        adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

        # 计算标准差 (用来衡量距离远近的尺度)
        distances = df[2].values
        std = distances.std()

        # 填充矩阵：使用高斯核函数转换
        for _, row in df.iterrows():
            i, j, dist = int(row[0]), int(row[1]), float(row[2])
            if i < num_nodes and j < num_nodes:
                adj[i, j] = np.exp(-(dist ** 2) / (std ** 2))

        # 必须加自连接
        for i in range(num_nodes):
            adj[i, i] = 1.0
        
        print(f"✅ 邻接矩阵加载成功 (Pickle格式): shape={adj.shape}")
        return adj
    
    else:
        raise ValueError(f"不支持的邻接矩阵文件格式: {adj_filename}")


def re_normalization(x, mean, std):
    """
    反归一化函数，支持多种数据格式
    
    Args:
        x: 归一化后的数据
        mean: 均值，形状可能是 (1,1,1,F) 或 (1,1,F,1)
        std: 标准差，形状可能是 (1,1,1,F) 或 (1,1,F,1)
    
    Returns:
        反归一化后的数据
    """
    # 处理不同的数据格式
    if len(x.shape) == 4:
        # 情况1: x shape = (B, N, F, T) - 多变量输入
        # mean/std shape = (1, 1, 1, F) 需要转换为 (1, 1, F, 1)
        if mean.shape == (1, 1, 1, x.shape[2]):
            # 转换归一化参数以匹配 (B, N, F, T)
            mean_reshaped = mean.reshape(1, 1, x.shape[2], 1)
            std_reshaped = std.reshape(1, 1, x.shape[2], 1)
            x = x * std_reshaped + mean_reshaped
        elif mean.shape == (1, 1, x.shape[2], 1):
            # 已经匹配，直接使用
            x = x * std + mean
        else:
            raise ValueError(f"mean形状 {mean.shape} 与数据形状 {x.shape} 不匹配")
    
    elif len(x.shape) == 3:
        # 情况2: x shape = (B, N, T) - 单变量
        # mean/std shape = (1, 1, 1, F) 取第一个特征
        if len(mean.shape) == 4:
            mean_val = mean[0, 0, 0, 0] if mean.shape[3] > 0 else mean[0, 0, 0]
            std_val = std[0, 0, 0, 0] if std.shape[3] > 0 else std[0, 0, 0]
            x = x * std_val + mean_val
        else:
            x = x * std + mean
    else:
        # 其他情况，直接广播
        x = x * std + mean
    
    return x


def max_min_normalization(x, _max, _min):
    x = 1. * (x - _min)/(_max - _min)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x


def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)

        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA

        else:

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    #lambda_max = eigs(L, k=1, which='LR')[0].real
    try:
        from scipy.sparse.linalg import eigs
        lambda_max = eigs(L, k=1, which='LR')[0].real
    except:
        # 如果 ARPACK 报错，说明矩阵有问题，我们换用最稳的 numpy 全量计算
        lambda_max = max(np.linalg.eigvals(L)).real
    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


def generate_data(dataset_dir):
    """加载并处理ASTGCN格式的数据（多变量版本）
    
    期望的.npz文件格式:
    - x: (B, N, F, T_in) 输入特征
    - y: (B, N, F, T_out) 预测目标（多变量）
    - mean: (1, 1, 1, F) 归一化均值
    - std: (1, 1, 1, F) 归一化标准差
    
    Returns:
        train_x, train_target, val_x, val_target, test_x, test_target, mean, std
    """
    data = {}
    mean = None
    std = None
    
    for category in ['train', 'val', 'test']:
        file_path = os.path.join(dataset_dir, category + '.npz')
        print(f"加载 {category} 数据: {file_path}")
        
        cat_data = np.load(file_path)
        
        # 检查可用的键
        print(f"  可用键: {list(cat_data.keys())}")
        
        # 加载数据
        x = cat_data['x']  # (B, N, F, T_in)
        y = cat_data['y']  # (B, N, F, T_out) - 多变量目标
        
        if mean is None:
            mean = cat_data['mean']
            std = cat_data['std']
            print(f"  归一化参数 - mean shape: {mean.shape}, std shape: {std.shape}")
        
        data['x_' + category] = x
        data['y_' + category] = y
        
        print(f"  {category}_x shape: {x.shape}")
        print(f"  {category}_y shape: {y.shape}")
    
    # ASTGCN 期望的返回格式:
    # x 已经是归一化的，直接返回
    # y 是真实值（未归一化），现在是多变量 (B, N, F, T_out)
    train_x = data['x_train']
    train_target = data['y_train']
    val_x = data['x_val']
    val_target = data['y_val']
    test_x = data['x_test']
    test_target = data['y_test']
    
    return train_x, train_target, val_x, val_target, test_x, test_target, mean, std


def load_data_new(graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size, shuffle=True):
    '''
    这个是为PEMS的数据准备的函数
    将x,y都处理成归一化到[-1,1]之前的数据;
    每个样本同时包含所有监测点的数据，所以本函数构造的数据输入时空序列预测模型；
    该函数会把hour, day, week的时间串起来；
    注： 从文件读入的数据，x是最大最小归一化的，但是y是真实值
    这个函数转为mstgcn，astgcn设计，返回的数据x都是通过减均值除方差进行归一化的，y都是真实值
    :param graph_signal_matrix_filename: str
    :param num_of_hours: int
    :param num_of_days: int
    :param num_of_weeks: int
    :param DEVICE:
    :param batch_size: int
    :return:
    three DataLoaders, each dataloader contains:
    test_x_tensor: (B, N_nodes, in_feature, T_input)
    test_decoder_input_tensor: (B, N_nodes, T_output)
    test_target_tensor: (B, N_nodes, T_output)

    '''

    # file = os.path.basename(graph_signal_matrix_filename).split('.')[0]
    #
    # dirpath = os.path.dirname(graph_signal_matrix_filename)
    #
    # filename = os.path.join(dirpath,
    #                         file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks)) +'_astcgn'
    #
    # print('load file:', filename)
    #
    # file_data = np.load(filename + '.npz')
    # train_x = file_data['train_x']  # (10181, 307, 3, 12)
    # train_x = train_x[:, :, 0:1, :]
    # train_target = file_data['train_target']  # (10181, 307, 12)
    #
    # val_x = file_data['val_x']
    # val_x = val_x[:, :, 0:1, :]
    # val_target = file_data['val_target']
    #
    # test_x = file_data['test_x']
    # test_x = test_x[:, :, 0:1, :]
    # test_target = file_data['test_target']
    #
    # mean = file_data['mean'][:, :, 0:1, :]  # (1, 1, 3, 1)
    # std = file_data['std'][:, :, 0:1, :]  # (1, 1, 3, 1)

    train_x, train_target, val_x, val_target, test_x, test_target, mean, std = generate_data(graph_signal_matrix_filename)

    # ------- train_loader -------
    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    # ------- val_loader -------
    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    val_target_tensor = torch.from_numpy(val_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_target_tensor)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ------- test_loader -------
    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # print
    print('train:', train_x_tensor.size(), train_target_tensor.size())
    print('val:', val_x_tensor.size(), val_target_tensor.size())
    print('test:', test_x_tensor.size(), test_target_tensor.size())

    return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, mean, std

def load_graphdata_channel1(graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size, shuffle=True):
    '''
    这个是为PEMS的数据准备的函数
    将x,y都处理成归一化到[-1,1]之前的数据;
    每个样本同时包含所有监测点的数据，所以本函数构造的数据输入时空序列预测模型；
    该函数会把hour, day, week的时间串起来；
    注： 从文件读入的数据，x是最大最小归一化的，但是y是真实值
    这个函数转为mstgcn，astgcn设计，返回的数据x都是通过减均值除方差进行归一化的，y都是真实值
    :param graph_signal_matrix_filename: str
    :param num_of_hours: int
    :param num_of_days: int
    :param num_of_weeks: int
    :param DEVICE:
    :param batch_size: int
    :return:
    three DataLoaders, each dataloader contains:
    test_x_tensor: (B, N_nodes, in_feature, T_input)
    test_decoder_input_tensor: (B, N_nodes, T_output)
    test_target_tensor: (B, N_nodes, T_output)

    '''

    file = os.path.basename(graph_signal_matrix_filename).split('.')[0]

    dirpath = os.path.dirname(graph_signal_matrix_filename)

    filename = os.path.join(dirpath,
                            file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks)) +'_astcgn'

    print('load file:', filename)

    file_data = np.load(filename + '.npz')
    train_x = file_data['train_x']  # (10181, 307, 3, 12)
    train_x = train_x[:, :, 0:1, :]
    train_target = file_data['train_target']  # (10181, 307, 12)

    val_x = file_data['val_x']
    val_x = val_x[:, :, 0:1, :]
    val_target = file_data['val_target']

    test_x = file_data['test_x']
    test_x = test_x[:, :, 0:1, :]
    test_target = file_data['test_target']

    mean = file_data['mean'][:, :, 0:1, :]  # (1, 1, 3, 1)
    std = file_data['std'][:, :, 0:1, :]  # (1, 1, 3, 1)

    # ------- train_loader -------
    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    # ------- val_loader -------
    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    val_target_tensor = torch.from_numpy(val_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_target_tensor)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ------- test_loader -------
    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # print
    print('train:', train_x_tensor.size(), train_target_tensor.size())
    print('val:', val_x_tensor.size(), val_target_tensor.size())
    print('test:', test_x_tensor.size(), test_target_tensor.size())

    return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, mean, std


def compute_val_loss_mstgcn(net, val_loader, criterion, sw, epoch, limit=None):
    '''
    for rnn, compute mean loss on validation set
    :param net: model
    :param val_loader: torch.utils.data.utils.DataLoader
    :param criterion: torch.nn.MSELoss
    :param sw: tensorboardX.SummaryWriter
    :param global_step: int, current global_step
    :param limit: int,
    :return: val_loss
    '''

    net.train(False)  # ensure dropout layers are in evaluation mode

    with torch.no_grad():

        val_loader_length = len(val_loader)  # nb of batch

        tmp = []  # 记录了所有batch的loss

        for batch_index, batch_data in enumerate(val_loader):
            encoder_inputs, labels = batch_data
            outputs = net(encoder_inputs)
            loss = criterion(outputs, labels)  # 计算误差
            tmp.append(loss.item())
            if batch_index % 100 == 0:
                print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))
            if (limit is not None) and batch_index >= limit:
                break

        validation_loss = sum(tmp) / len(tmp)
        sw.add_scalar('validation_loss', validation_loss, epoch)
    return validation_loss


def evaluate_on_test_mstgcn(net, test_loader, test_target_tensor, sw, epoch, _mean, _std):
    '''
    for rnn, compute MAE, RMSE, MAPE scores of the prediction for every time step on testing set.

    :param net: model
    :param test_loader: torch.utils.data.utils.DataLoader
    :param test_target_tensor: torch.tensor (B, N_nodes, T_output, out_feature)=(B, N_nodes, T_output, 1)
    :param sw:
    :param epoch: int, current epoch
    :param _mean: (1, 1, 3(features), 1)
    :param _std: (1, 1, 3(features), 1)
    '''

    net.train(False)  # ensure dropout layers are in test mode

    with torch.no_grad():

        test_loader_length = len(test_loader)

        test_target_tensor = test_target_tensor.cpu().numpy()

        prediction = []  # 存储所有batch的output

        for batch_index, batch_data in enumerate(test_loader):

            encoder_inputs, labels = batch_data

            outputs = net(encoder_inputs)

            prediction.append(outputs.detach().cpu().numpy())

            if batch_index % 100 == 0:
                print('predicting testing set batch %s / %s' % (batch_index + 1, test_loader_length))

        prediction = np.concatenate(prediction, 0)  # (batch, T', 1)
        prediction_length = prediction.shape[2]

        for i in range(prediction_length):
            assert test_target_tensor.shape[0] == prediction.shape[0]
            print('current epoch: %s, predict %s points' % (epoch, i))
            mae = mean_absolute_error(test_target_tensor[:, :, i], prediction[:, :, i])
            rmse = mean_squared_error(test_target_tensor[:, :, i], prediction[:, :, i]) ** 0.5
            mape = masked_mape_np(test_target_tensor[:, :, i], prediction[:, :, i], 0)
            print('MAE: %.2f' % (mae))
            print('RMSE: %.2f' % (rmse))
            print('MAPE: %.2f' % (mape))
            print()
            if sw:
                sw.add_scalar('MAE_%s_points' % (i), mae, epoch)
                sw.add_scalar('RMSE_%s_points' % (i), rmse, epoch)
                sw.add_scalar('MAPE_%s_points' % (i), mape, epoch)


def predict_and_save_results_mstgcn(net, data_loader, data_target_tensor, global_step, _mean, _std, params_path, type):
    '''
    多变量版本的预测和结果保存
    
    :param net: nn.Module
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor (B, N, F, T_out) - 多变量目标
    :param global_step: int
    :param _mean: (1, 1, 1, F) - 归一化均值
    :param _std: (1, 1, 1, F) - 归一化标准差
    :param params_path: the path for saving the results
    :param type: string ('test' or 'val')
    :return:
    '''
    net.train(False)  # ensure dropout layers are in test mode

    with torch.no_grad():

        data_target_tensor = data_target_tensor.cpu().numpy()  # (B, N, F, T_out)

        loader_length = len(data_loader)  # nb of batch

        prediction = []  # 存储所有batch的output

        input_data = []  # 存储所有batch的input

        for batch_index, batch_data in enumerate(data_loader):

            encoder_inputs, labels = batch_data

            # 保存完整的输入数据（包含所有特征）
            input_data.append(encoder_inputs.cpu().numpy())  # (batch, N, F, T_in)

            outputs = net(encoder_inputs)  # (batch, N, F, T_out)

            prediction.append(outputs.detach().cpu().numpy())

            if batch_index % 100 == 0:
                print('predicting data set batch %s / %s' % (batch_index + 1, loader_length))

        input_data = np.concatenate(input_data, 0)  # (B, N, F, T_in)
        
        # 反归一化输入数据
        input_normalized = re_normalization(input_data, _mean, _std)

        prediction = np.concatenate(prediction, 0)  # (B, N, F, T_out)

        print('input shape:', input_data.shape)
        print('input after normalization shape:', input_normalized.shape)
        print('prediction shape:', prediction.shape)
        print('data_target_tensor shape:', data_target_tensor.shape)
        
        output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (global_step, type))
        np.savez(output_filename, 
                 input=input_normalized, 
                 prediction=prediction, 
                 data_target_tensor=data_target_tensor)

        # 计算每个特征的误差指标
        feature_names = ['小客车上行', '小客车下行', '非小客车上行', '非小客车下行']
        num_features = prediction.shape[2]
        
        print(f"\n{'='*60}")
        print(f"按特征分类评估结果")
        print(f"{'='*60}")
        
        excel_list = []
        prediction_length = prediction.shape[3]

        # 对每个时间步进行单独评估
        for i in range(prediction_length):
            print(f'\n当前epoch: {global_step}, 预测第 {i+1} 个时间点')
            
            # 对每个特征进行单独评估
            for feat_idx in range(num_features):
                feat_pred = prediction[:, :, feat_idx, i]  # (B, N)
                feat_true = data_target_tensor[:, :, feat_idx, i]  # (B, N)
                
                mae = mean_absolute_error(feat_true.flatten(), feat_pred.flatten())
                rmse = mean_squared_error(feat_true.flatten(), feat_pred.flatten()) ** 0.5
                mape = masked_mape_np(feat_true.flatten(), feat_pred.flatten(), 0)
                
                feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f'Feature_{feat_idx}'
                print(f'  {feat_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%')
                
                excel_list.extend([mae, rmse, mape])

        # 计算全局指标（所有特征和时间步的平均）
        print(f"\n{'='*60}")
        print(f"全局评估结果（所有特征和时间步）")
        print(f"{'='*60}")
        
        mae_all = mean_absolute_error(data_target_tensor.flatten(), prediction.flatten())
        rmse_all = mean_squared_error(data_target_tensor.flatten(), prediction.flatten()) ** 0.5
        mape_all = masked_mape_np(data_target_tensor.flatten(), prediction.flatten(), 0)
        
        print(f'全局 MAE: {mae_all:.2f}')
        print(f'全局 RMSE: {rmse_all:.2f}')
        print(f'全局 MAPE: {mape_all:.2f}%')
        
        excel_list.extend([mae_all, rmse_all, mape_all])
        print(f"\n完整指标列表: {excel_list}")
