import os
import time
import copy
import torch


def train(model, data, optimizer, criterion, device, num_epochs=10, scheduler=None):
    """
    function for training the neural network
    """
    time_start = time.time()
    loss_values_train = list()
    loss_values_val = list()
    best_weights = None
    best_val_loss = 1e10
    model.to(device)
    
    #Iterate over epochs
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-'*15)
                
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                actual_loss = 0.0
            else :
                model.eval()
                val_loss = 0.0
                
            #iterate over batches
            for images, _ in data["train"]:
                images = images.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(images)
                    loss = criterion(outputs, images)
                    loss = loss.sum(dim=[1,2,3]).mean(dim=[0])
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        actual_loss += loss.detach().cpu().item()
                    else :
                        val_loss += loss.detach().cpu().item()
            #update loss over batches 
            if phase == "train":
                epoch_loss = actual_loss / len(data[phase].dataset)
                print("training loss value is : {}".format(round(epoch_loss, 5)))
                loss_values_train.append(epoch_loss)
            else :
                epoch_val_loss = val_loss / len(data[phase].dataset)
                print("validation loss value is : {}".format(round(epoch_val_loss, 5)))
                loss_values_val.append(epoch_val_loss)
            

        #save if best is found !
        if loss_values_val[-1] < best_val_loss :
            best_val_loss = loss_values_val[-1]
            best_weights = copy.deepcopy(model.state_dict())
            print("Improvemet ... model saved !")
        
        #scheduler
        if scheduler != None:
            scheduler.step(loss_values_val[-1])
    time_end = time.time()
    elapsed_time = time_end - time_start
    print('Training complete in {:.0f}m {:.0f}s'.format(elapsed_time // 60, elapsed_time % 60))
    return model, loss_values_train, loss_values_val


