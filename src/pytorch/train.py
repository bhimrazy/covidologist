import os
import time
import copy
import torch
from tqdm import tqdm
from src.pytorch.utils import save_checkpoint
from src.pytorch.dataloader import dataloaders, dataset_sizes
from src.pytorch.config import LOGS_FILE_NAME, DEVICE
from src.pytorch.model import model, criterion, optimizer, exp_lr_scheduler


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    logs = {"train": {"loss": [], "acc": []}, "val": {"loss": [], "acc": []}}

    for epoch in tqdm(range(num_epochs), "Training..."):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            log = f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}'
            logs[phase]["loss"].append(epoch_loss)
            logs[phase]["acc"].append(epoch_acc)
            print(log)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    print("Writing logs....")
    with open(LOGS_FILE_NAME, "w+") as f:
        f.write(str(logs))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def train_and_save_model():
    """This function trains and saves model
    """
    model_conv = train_model(model, criterion, optimizer,
                             exp_lr_scheduler, num_epochs=5)

    save_checkpoint(state=model_conv.state_dict())
