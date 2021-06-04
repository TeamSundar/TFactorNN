# Import package
import numpy as np
import random, torch, os, argparse
from numpy.core.numeric import ones_like
from torch_geometric.data import Data, DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, confusion_matrix
from torch_geometric.data import DataLoader
from utils.model_gcn import GCN
from utils.model_gen import GEN

def to_device(data, device):
    return data.to(device, non_blocking=True)

def train():
    model.train()
    for data in train_loader:
        a = to_device(data.x, device)
        b = to_device(data.edge_index, device)
        c = to_device(data.batch, device)
        d = to_device(data.y.to(torch.float32) , device)
        
        out = flatten(model(a, b, c))
        loss = loss_func(out, d)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
def evaluate(loader, epoch, t):
    model.eval()
    with torch.no_grad():
        with tqdm(loader, unit="batch") as tepoch:
            conf_mat = np.zeros((2,2))
            acc, aupr, auroc, loss=[], [], [], []
            for data in tepoch:
                tepoch.set_description(f"Epoch {epoch} ({t})")
                a = to_device(data.x, device)
                b = to_device(data.edge_index, device)
                c = to_device(data.batch, device)
                d = to_device(data.y.to(torch.float32), device)

                out = flatten(model(a, b, c))            
                out_np, d_np = out.detach().cpu().clone().numpy(), d.detach().cpu().clone().numpy()

                # AUPR
                precision, recall, thresholds = precision_recall_curve(d_np, out_np)
                auc_score = auc(recall, precision)
                aupr.append(auc_score)

                # AUROC
                try:
                    roc_score = roc_auc_score(d_np, out_np)
                    auroc.append(roc_score)
                except:
                    pass

                # LOSS
                loss_ = loss_func(out, d)
                loss.append(loss_.detach().cpu().clone().numpy())

                # ACCURACY
                acc_score = binary_acc(out_np, d_np)
                acc.append(acc_score)

                # Compute confusion
                conf_mat = conf_mat + confusion_matrix(d_np, np.round(out_np))
                
                tepoch.set_postfix(loss=np.nanmean(loss),
                                   acc=np.nanmean(acc),
                                   aupr=np.nanmean(aupr),
                                   auroc=np.nanmean(auroc))

    return np.nanmean(acc), np.nanmean(aupr), np.nanmean(auroc), np.nanmean(loss), conf_mat

def binary_acc(y_pred, y_test):
    y_pred_tag = np.round(y_pred)
    correct_results_sum = (y_pred_tag == y_test).sum()
    acc = correct_results_sum/y_test.shape[0]
    acc = np.round(acc * 100)
    return acc

def split_data(data_, split, i):
    torch.manual_seed(i)
    random.shuffle(data_)
    split_n = int(split*(len(data_)))
    train_data = data_[:split_n]
    test_data = data_[split_n:]
    return train_data, test_data

def flatten(t):
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t

def save_dict(obj, name):
    with open(DATAPATH+'results/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(DATAPATH+'results/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='GNN Pytorch Geometrics')
    
    parser.add_argument('--device', type=int, default=1,
                    help='which gpu to use if any (default: 0)')
    parser.add_argument('--tf', type=str, default='ctcf',
                    help='Current TF (default: ctcf)')
    parser.add_argument('--model_type', type=str, default='gcn',
                    help='gcn, gnn etc (default: gcn)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                    help='dropout ratio (default: 0)')
    parser.add_argument('--emb_dim', type=int, default=32,
                    help='dimensionality of hidden units in GNNs (default: 32)')
    parser.add_argument('--batch_size', type=int, default=256,
                    help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
    parser.add_argument('--log_dir', type=str, default="",
                    help='log directory')
    parser.add_argument('--checkpoint_dir', type=str, default = '', help='directory to save checkpoint')
    parser.add_argument('--save_test_dir', type=str, default = '', help='directory to save test submission file')
    args = parser.parse_args()
    
    TF = args.tf
    channels = args.emb_dim
    model_type = args.model_type
    
    DATAPATH = '/DATA/yogesh/encodeDream/%s/'%(TF)
    POSITIVE = DATAPATH+'processed_out/positive/'
    NEGATIVE = DATAPATH+'processed_out/negative/'

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.cuda.set_device(1)
    print('Current cuda device ID:',torch.cuda.current_device())
    print('Current cuda device name:', torch.cuda.get_device_name())
    print()
    
    # Train/test
    if model_type=='gcn':
        model = GCN(hidden_channels=channels)
    elif model_type=='gen':
        model = GEN(hidden_channels=channels)
    model.to(device)
    print('Model architecture:')
    print(model)
    print()
    
    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    loss_func = torch.nn.BCELoss()
    
    # Initialize dict to save results
    results = {'train': {'accuracy': [], 'auroc': [], 'aupr': []},
              'test': {'accuracy': [], 'auroc': [], 'aupr': []}}
    
    # Import positive dataset
    pos_set = torch.load(POSITIVE+os.listdir(POSITIVE)[0])
    np.seterr(divide='ignore', invalid='ignore')
    
    # Start training and evaluation
    for epoch in range(1, len(os.listdir(NEGATIVE))):
        # Import negative dataset
        neg_set = torch.load(NEGATIVE+os.listdir(NEGATIVE)[epoch])
        DATALIST = pos_set+neg_set

        # Split data
        train_dataset, test_dataset = split_data(DATALIST, 0.8, epoch)

        # Prepare dataloader
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=True)

        # Start training
        train()

        # Evaluate
        train_acc, train_aupr, train_auroc, train_loss, train_conf = evaluate(train_loader, epoch, 'train')
        test_acc, test_aupr, test_auroc, test_loss, test_conf = evaluate(test_loader, epoch ,'test')

        # save metrics to list
        results['train']['accuracy'].append(train_acc)
        results['train']['auroc'].append(train_auroc)
        results['train']['aupr'].append(train_aupr)
        results['test']['accuracy'].append(test_acc)
        results['test']['auroc'].append(test_auroc)
        results['test']['aupr'].append(test_aupr)

#         # Print results
#         print(f'Epoch:{epoch:03d}')
#         print()
#         print('Train:\n', train_conf)
#         print('Test:\n', test_conf)
#         print()
#         print(f'Train::ACC:{train_acc} AUPR:{train_aupr} AUROC:{train_auroc} LOSS:{train_loss}')
#         print(f'Test ::ACC:{test_acc} AUPR:{test_aupr} AUROC:{test_auroc} LOSS:{test_loss}')

#         print()
    
    np.save(DATAPATH+'results/'+TF+'_'+model_type+'_'+str(channels)+'.npy', results)