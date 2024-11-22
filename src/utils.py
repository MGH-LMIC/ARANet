import torch
import torchvision
from dataset import MRIDataset
from torch.utils.data import DataLoader
import numpy as np
from torchmetrics import Dice

device  = 'cuda' if torch.cuda.is_available() else 'cpu'

def save_checkpoint(state, filename ):
    print('=> Saving checkpoint ---------------------------------------------------------')
    print('')
    torch.save(state, filename)
    
def load_checkpoint(checkpoint, model):
    print('=> Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])

def get_loaders(train_dir,train_mask_dir,val_dir,val_mask_dir,batch_size,train_transform,val_transform,num_workers = 4, pin_memory = True):
    
    train_ds = MRIDataset(image_dir = train_dir, mask_dir = train_mask_dir,transform = train_transform)
    
    train_loader = DataLoader(train_ds,batch_size = batch_size,num_workers = num_workers,pin_memory = pin_memory,shuffle = True)
    
    val_ds = MRIDataset(image_dir = val_dir,mask_dir = val_mask_dir,transform = val_transform)
    
    val_loader = DataLoader(val_ds,batch_size = batch_size,num_workers = num_workers,pin_memory = pin_memory,shuffle = False)
    
    return train_loader, val_loader

def check_accuracy(loader, model, device = 'cuda'):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    # dice_score2 = 0
    model.eval()
    calc_dice = Dice(num_classes=3).to(device)
    
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device).long()#.unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = preds.argmax(dim=1)
            
            dice_score += calc_dice(preds, y)
            

        # print(f'Got {num_correct}/{num_pixels} with acc {num_correct/ num_pixels*100 : 0.2f}')
        print(f'Dice score: {dice_score/len(loader) : 0.3f}')
        # print(f'Dice score2: {dice_score2/len(loader) : 0.3f}')
    
    
    # model.train()
    return dice_score/len(loader)#, num_correct/ num_pixels*100

def save_predictions_as_imgs(loader, model, folder = 'saved_images/', device = 'cuda'):
    model.eval()

    for idx, (x,y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f'{folder}/pred_{idx}.png')
        torchvision.utils.save_image(y.unsqueeze(1),f'{folder}/Ground_truth_{idx}.png')

