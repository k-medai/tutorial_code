import argparse
from utils.seed import seed
import torch
from utils.dataset.load_ptbxl import load_dataloader
from models.CNN import SimpleCNN
import torch.nn as nn
import torch.optim as optim
from utils.train import train
from utils.evaluate import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--seed',
                    default= 7,
                    type= int)
parser.add_argument('--device',
                    default= 'cuda',
                    type= str)
parser.add_argument('--csv-path',
                    default= './data/ptbxl/super/',
                    type= str,
                    help= 'split을 위한 csv file이 있는 경로 입력')
parser.add_argument('--signal-path',
                    default= '../../physionet.org/files/ptb-xl/1.0.3/',
                    type= str,
                    help= 'ECG signal(raw data)이 있는 경로 입력')
parser.add_argument('--sampling-rate',
                    default= 500,
                    type= int)
parser.add_argument('--batch-size',
                    default= 256,
                    type= int)
parser.add_argument('--model',
                    default= 'SimpleCNN',
                    type= str)
parser.add_argument('--learning-rate',
                    default= 1e-3,
                    type= float)
parser.add_argument('--epochs',
                    default= 30,
                    type= int)

def main() :
    args = parser.parse_args()
    
    # seed
    seed(args.seed)
    
    # device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # data
    trainloader, validloader, testloader = load_dataloader(args.csv_path, args.signal_path, args.sampling_rate, args.batch_size)
    
    # model
    if args.model == 'SimpleCNN':
        model = SimpleCNN()
    
    # train
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr= args.learning_rate)
    train(
        model,
        trainloader, validloader,
        device,
        criterion, optimizer, args.epochs
        )
    
    # result
    model.load_state_dict(torch.load('./results/best_state_dict.pth'))
    testscore = evaluate(testloader, model, device)
    print('Result of testset')
    print('AUC: {:.4f}, F1-score: {:.4f}'.format(testscore['auc'], testscore['f1']))

if __name__ == '__main__' : main()