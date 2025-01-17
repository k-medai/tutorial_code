import os
import time
from tqdm import tqdm
import torch
from utils.evaluate import evaluate
import matplotlib.pyplot as plt

def train(model, trainloader, validloader, device, criterion, optimizer, epochs):
    train_losses = [];train_aucs = [];train_f1s = []
    valid_losses = [];valid_aucs = [];valid_f1s = []
    
    best_f1 = float('-inf')
    os.makedirs('./results', exist_ok= True)
    best_state_dict = './results/best_state_dict.pth'
    
    model.to(device)
    for epoch in range(epochs) :
        start_time = time.time()
        
        model.train()
        epoch_loss = 0.0
        train_progress_bar = tqdm(trainloader, unit= 'batch', desc= f'Epoch {epoch+1}')
        for inputs, labels in train_progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        train_losses.append(epoch_loss/len(trainloader))
        trainscore = evaluate(trainloader, model, device)
        train_aucs.append(trainscore['auc'])
        train_f1s.append(trainscore['f1'])
        
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                valid_loss += loss.item()
        valid_losses.append(valid_loss/len(validloader))
        validscore = evaluate(validloader, model, device)
        valid_aucs.append(validscore['auc'])
        valid_f1s.append(validscore['f1'])
        
        if best_f1 < validscore['f1']:
            best_f1 = validscore['f1']
            torch.save(model.state_dict(), best_state_dict)
        
        end_time = time.time()
        total_time = end_time-start_time
        if (epoch==0) or (((epoch+1)%5)==0):
            print('Time: {:.0f}m {:.0f}s Loss: {:3.4f}\nTrain | AUC: {:.4f}, F1: {:.4f}\nValid | AUC: {:.4f}, F1: {:.4f}\n'.format(
                total_time//60, total_time%60, train_losses[-1],
                trainscore['auc'], trainscore['f1'],
                validscore['auc'], validscore['f1']))
        else :
            print('Time: {:.0f}m {:.0f}s Loss: {:3.4f}\n'.format(total_time//60, total_time%60, train_losses[-1]))
    
    # loss curve
    plt.figure(figsize= (10,5))
    plt.plot(range(1,epochs+1), train_losses, label= 'Train Loss', color= 'r')
    plt.plot(range(1,epochs+1), valid_losses, label= 'Valid Loss', color= 'b')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('./results/loss_curve.png')
    plt.close
    
    # AUC curve
    plt.figure(figsize= (10,5))
    plt.plot(range(1,epochs+1), train_aucs, label= 'Train AUC', color= 'r')
    plt.plot(range(1,epochs+1), valid_aucs, label= 'Valid AUC', color= 'b')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('AUC curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('./results/auc_curve.png')
    plt.close
    
    # AUC curve
    plt.figure(figsize= (10,5))
    plt.plot(range(1,epochs+1), train_f1s, label= 'Train F1', color= 'r')
    plt.plot(range(1,epochs+1), valid_f1s, label= 'Valid F1', color= 'b')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.title('F1 curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('./results/f1_curve.png')
    plt.close
