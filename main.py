from argparse import ArgumentParser
import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn.utils import resample
from torchmetrics.classification import MulticlassF1Score,MulticlassAccuracy,MulticlassConfusionMatrix
import matplotlib.pyplot as plt
from network import CreateModel



def main():
    parser = ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train", required="true", help="Execution step")

    args = parser.parse_args()

    # Setting variables

    num_epochs = 300
    learning_rate = 0.01

    # Setting device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Import DataFrames
    train_df = pd.read_parquet("x_train.parquet")
    validation = pd.read_parquet("x_valid.parquet")
    test = pd.read_parquet("x_test.parquet")

    # Split dataset

    x_train = train_df.iloc[:,:-1].values
    y_train = train_df.iloc[:,-1].values

    x_valid = validation.iloc[:,:-1].values
    y_valid = validation.iloc[:,-1].values

    x_test = test.iloc[:,:-1].values
    y_test = test.iloc[:,-1].values

    # Tranform to tensor

    x_train = torch.from_numpy(x_train).type(torch.float)
    y_train = torch.from_numpy(y_train).type(torch.long)

    x_valid = torch.from_numpy(x_valid).type(torch.float)
    y_valid = torch.from_numpy(y_valid).type(torch.long)

    x_test = torch.from_numpy(x_test).type(torch.float)
    y_test = torch.from_numpy(y_test).type(torch.long)

    x_train = x_train.to(device)
    y_train = y_train.to(device)

    x_valid = x_valid.to(device)
    y_valid = y_valid.to(device)

    x_test = x_test.to(device)
    y_test = y_test.to(device)
    
    # Doing Resampling in DataFrame Train
    
    df_1=train_df[train_df['target']==1]
    df_2=train_df[train_df['target']==2]
    df_3=train_df[train_df['target']==3]
    df_4=train_df[train_df['target']==4]
    df_0=(train_df[train_df['target']==0]).sample(n=20000,random_state=42)
    
    df_1_upsample=resample(df_1,replace=True,n_samples=20000,random_state=123)
    df_2_upsample=resample(df_2,replace=True,n_samples=20000,random_state=124)
    df_3_upsample=resample(df_3,replace=True,n_samples=20000,random_state=125)
    df_4_upsample=resample(df_4,replace=True,n_samples=20000,random_state=126)
    
    train=pd.concat([df_0,df_1_upsample,df_2_upsample,df_3_upsample,df_4_upsample])

    # Create Metrics and auxiliar items
    f1 = MulticlassF1Score(num_classes=5,average='weighted').to(device)
    acc = MulticlassAccuracy(num_classes=5).to(device)
    confuse_matrix = MulticlassConfusionMatrix(num_classes=5).to(device)
    softmax = nn.Softmax(dim=1)
    loss_fn = nn.CrossEntropyLoss()

    if args.mode == "train":

        # Creating model
        model = CreateModel().to(device)
        
        #Creating Loss Function and Optimizer
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        


        for epoch in range(num_epochs):

              softmax = nn.Softmax(dim=1)
              model.train()
              # 1 - Foward
              y_logits = model(x_train)
              y_labels = softmax(y_logits).argmax(dim=1).type(torch.float).requires_grad_(True).unsqueeze(dim=1)
              
            
              # 2 - Calculate Loss
              loss = loss_fn(y_logits,y_train)
              
              # 3 - Zero Grad
            
              optimizer.zero_grad()
            
              # 4 - backpropagation
            
              loss.backward()
            
              # 5 - optimizer step
            
              optimizer.step()
            
              # 6 - Validate Set
            
              model.eval()
              with torch.inference_mode():
                y_logits_valid = model(x_valid)
                loss_valid = loss_fn(y_logits_valid, y_valid)
                f1_score = f1(softmax(y_logits_valid), y_valid)
              
              # 6 - Print 
              if epoch % 10 == 0:
                  print(f'Epoch {epoch} - Train Loss - {loss} - Valid Loss - {loss_valid} - F1 Score valid - {f1_score}')
                        


        torch.save(model.state_dict(), "networkfullyconected")

    else:
        # Test
        model = CreateModel().to(device)
        model.load_state_dict(torch.load("networkfullyconected", map_location=torch.device(device)))
        model.eval()
        with torch.inference_mode():
            print("Running model with test dataset ...")
            y_logits_test = model(x_test)
            f1_score = f1(softmax(y_logits_test), y_test)
            acc_score = acc(softmax(y_logits_test), y_test)
            confuse_matrix_score  = confuse_matrix(softmax(y_logits_test), y_test)
            print(f'F1-Weighted : {f1_score :.2f}')
            print(f'Acurracy : {acc_score :.2f}')
            print("Matriz de Confusão")
            print(confuse_matrix_score)



if __name__ == "__main__":
    main()
