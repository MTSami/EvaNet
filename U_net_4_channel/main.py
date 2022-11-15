import torch
from torch.optim import Adam, SGD

import numpy as np

from data.data import *
from unet_model import *
from metrics import *

from tqdm import tqdm

import matplotlib.pyplot as plt
import cv2

import argparse

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)



def run_pred(model, data_loader):
    
    ## Model gets set to evaluation mode
    model.eval()
    pred_patches_dict = dict()

    for data_dict in tqdm(data_loader):
        
        ## RGB data
        rgb_data = data_dict['rgb_data'].float().to(DEVICE)

        ## Data labels
        labels = data_dict['labels'].long().to(DEVICE)

        ## Get filename
        filename = data_dict['filename']
        # print("filename: ", filename)

        ## Get model prediction
        pred = model(rgb_data)


        ## Remove pred and GT from GPU and convert to np array
        pred_labels_np = pred.detach().cpu().numpy() 
        gt_labels_np = labels.detach().cpu().numpy()
        
        ## Save Image and RGB patch
        for idx in range(rgb_data.shape[0]):
            pred_patches_dict[filename[idx]] = pred_labels_np[idx, :, :, :]
        
    return pred_patches_dict
    

def find_patch_meta(pred_patches_dict):
    y_max = 0
    x_max = 0

    for item in pred_patches_dict:

        ##print(item)

        temp = int(item.split("_")[3])
        if temp>y_max:
            y_max = temp

        temp = int(item.split("_")[5])
        if temp>x_max:
            x_max = temp


    y_max+=1
    x_max+=1
    # print(f"y_max: {y_max}, x_max: {x_max}")
    # print(f"Number of patches: {y_max*x_max}")

    # print(f"rgb_pred_patches len: {len(pred_patches_dict)}")
    
    return y_max, x_max


def stitch_patches(pred_patches_dict, cropped_val_test_data_path):
    y_max, x_max = find_patch_meta(pred_patches_dict)
    
    for i in range(y_max):
        for j in range(x_max):
            dict_key = f"{args.test_region[:-5]}_y_{i}_x_{j}_features.npy"
            #print(dict_key)
        
            pred_patch = pred_patches_dict[dict_key]
            pred_patch = np.transpose(pred_patch, (1, 2, 0))

            rgb_patch = np.load(os.path.join(cropped_val_test_data_path, dict_key))[:, :, :3]


            if j == 0:
                rgb_x_patches = rgb_patch
                pred_x_patches = pred_patch
            else:
                rgb_x_patches = np.concatenate((rgb_x_patches, rgb_patch), axis = 1)
                pred_x_patches = np.concatenate((pred_x_patches, pred_patch), axis = 1)

            ## rgb_patches.append(rgb_patch)
            ## pred_patches.append(pred_patch)
    
        if i == 0:
            rgb_y_patches = rgb_x_patches
            pred_y_patches = pred_x_patches
        else:
            rgb_y_patches = np.vstack((rgb_y_patches, rgb_x_patches))
            pred_y_patches = np.vstack((pred_y_patches, pred_x_patches))
        

    rgb_stitched = rgb_y_patches.astype('uint8')
    pred_stitched = np.argmax(pred_y_patches, axis = -1)
    
    return rgb_stitched, pred_stitched


def center_crop(stictched_data, image, meta):
    
    dict_key = f"{args.test_region[:-5]}_Features7Channel.npy"
    
    if image:
        current_height, current_width, _ = stictched_data.shape
    else:
        current_height, current_width = stictched_data.shape    
    # print("current_height: ", current_height)
    # print("current_width: ", current_width)
    
    original_height = meta[dict_key]['height']
    original_width = meta[dict_key]['width']
    # print("original_height: ", original_height)
    # print("original_width: ", original_width)
    
    height_diff = current_height-original_height
    width_diff = current_width-original_width
    
    # print("height_diff: ", height_diff)
    # print("width_diff: ", width_diff)
    
    
    cropped = stictched_data[height_diff//2:current_height-height_diff//2, width_diff//2: current_width-width_diff//2]
    
    return cropped


def plot_curve(train_loss_dict, val_loss_dict):
    plt.figure(figsize = (8, 8))

    val_epochs = list(val_loss_dict.keys())
    val_losses = list(val_loss_dict.values())
    val_len = len(val_epochs)

    train_epochs = list(train_loss_dict.keys())
    train_losses = list(train_loss_dict.values())
    # train_losses = train_losses[:val_len]

    print(len(val_epochs))
    print(len(train_losses))

    plt.plot(val_epochs, val_losses, "bs-.", label = "Val Loss")
    plt.plot(train_epochs, train_losses, "rp-.", label = "Train Loss")

    plt.xlabel("EPOCH")  
    plt.ylabel("LOSS")
    plt.title("Training & Validation Curve")
    plt.legend(loc = 'upper right', prop={'size' : 15})
    # plt.savefig(f'{metric}.pdf', format = 'pdf')

    plt.show()



def get_meta_data(DATASET_PATH):
    
    DATASET = os.listdir(DATASET_PATH)
    DATASET = [file for file in DATASET if  file.endswith(".npy") and re.search("Features", file)]

    META_DATA = dict()

    for file_name in DATASET:
        file = np.load(os.path.join(DATASET_PATH, file_name))
        #print(file.shape)
        file_height, file_width, _ = file.shape
        #print(file_height)
        #print(file_width)

        elev_data = file[:, :, 3]
        file_elev_max = np.max(elev_data)
        file_elev_min = np.min(elev_data)
        # print(file_elev_max)
        # print(file_elev_min)

        if file_elev_max>config.GLOBAL_MAX:
            config.GLOBAL_MAX = file_elev_max
        if file_elev_min<config.GLOBAL_MIN:
            config.GLOBAL_MIN = file_elev_min


        META_DATA[file_name] = {"height": file_height,
                                "width": file_width}
        
    return META_DATA



def main(args):
    
    ## Meta Data
    META_DATA = get_meta_data(args.data_path)
    # print("Dataset Meta Data: ", META_DATA)
    print("Maximum Elevation of dataset: ", config.GLOBAL_MAX)
    print("Minimum Elevation of dataset: ", config.GLOBAL_MIN)
    
    
    ## Create torch dataloaders
    cropped_train_data_path = f"./data/{args.train_region}_{args.test_region}/cropped_data_train"
    cropped_val_test_data_path = f"./data/{args.train_region}_{args.test_region}/cropped_data_val_test"
    
    elev_train_dataset = get_dataset(cropped_train_data_path)
    elev_val_test_dataset = get_dataset(cropped_val_test_data_path)
    
    train_seq = np.arange(0, len(elev_train_dataset), dtype=int)
    d_len = len(elev_val_test_dataset)
    val_idx = int(0.5*d_len)
    val_seq = np.arange(0, val_idx, 1, dtype=int)
    test_seq = np.arange(0, d_len, 1, dtype=int)
    print(f"Size of Training data: {len(train_seq)}")
    print(f"Size of Testing data: {d_len}")
    
    train_dataset = torch.utils.data.Subset(elev_train_dataset, train_seq)
    val_dataset = torch.utils.data.Subset(elev_val_test_dataset, val_seq)
    test_dataset = torch.utils.data.Subset(elev_val_test_dataset, test_seq)
    
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size)
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size)
    
    
    ## Setup model and its parameters
    model = UNet(args.input_channel, args.pred_channel, ultrasmall = args.small_model).to(DEVICE)
    # print(model)
    optimizer = SGD(model.parameters(), lr = args.lr)
    criterion = torch.nn.CrossEntropyLoss(reduction = 'sum', ignore_index = 0)
    
    ## Model evaluator
    elev_eval = Evaluator()

    
    if args.mode == 'training':
        print("Model in Training mode")
        model_path = f"./{args.save_model_dir}/{args.test_region}/saved_model_{args.saved_model_epoch}.ckpt"

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            resume_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
        else:
            resume_epoch = 0
        print("Strating from epoch: ", resume_epoch)
        
        ################################## Training Loop#####################################    
        train_loss_dict = dict()
        val_loss_dict = dict()
        min_val_loss = 1e10
        
        for epoch in range(resume_epoch, args.epochs):

            ## Model gets set to training mode
            model.train()
            train_loss = 0 
            
            for data_dict in tqdm(train_loader):
                
                ## Retrieve data from data dict and send to deivce
                ## RGB data
                rgb_data = data_dict['rgb_data'].float().to(DEVICE)
                rgb_data.requires_grad = True

                labels = data_dict['labels'].long().to(DEVICE)
                labels.requires_grad = False  


                ## Get model prediction
                pred = model(rgb_data)

                ## Backprop Loss
                optimizer.zero_grad() 
                loss = criterion.forward(pred, labels)
                ##print("Loss: ", loss.item())

                loss.backward()
                optimizer.step()

                ## Record loss for batch
                train_loss += loss.item()
                    
            train_loss /= len(train_loader)
            train_loss_dict[epoch+1] = train_loss
            print(f"Epoch: {epoch+1} Training Loss: {train_loss}" )
                
            
            #=====================================================================================
            
                
            ## Do model validation for epochs that match VAL_FREQUENCY
            if (epoch+1)%args.val_freq == 0:    
                
                ## Model gets set to evaluation mode
                model.eval()
                val_loss = 0 
                
                print("Starting Evaluation")
                
                for data_dict in tqdm(val_loader):
                    
                    ## RGB data
                    rgb_data = data_dict['rgb_data'].float().to(DEVICE)

                    ## Data labels
                    labels = data_dict['labels'].long().to(DEVICE)

                    ## Get model prediction
                    pred = model(rgb_data)            

                    ## Backprop Loss
                    loss = criterion.forward(pred, labels)
                    ##print("Loss: ", loss.item())

                    ## Record loss for batch
                    val_loss += loss.item()

                    ## Remove pred and GT from GPU and convert to np array
                    pred_labels_np = pred.detach().cpu().numpy() 
                    gt_labels_np = labels.detach().cpu().numpy()
                
                val_loss /= len(val_loader)
                val_loss_dict[epoch+1] = val_loss
                print(f"Epoch: {epoch+1} Validation Loss: {val_loss}" )
                
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    print("Saving Model")
                    torch.save({'epoch': epoch + 1,  # when resuming, we will start at the next epoch
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict()}, 
                                f"./{args.save_model_dir}/{args.test_region}/saved_model_{epoch+1}.ckpt")
        
            
                ### Save model Periodically, every multiple of 2
                if args.save_freq and ((epoch+1)%args.save_freq == 0):
                    torch.save({'epoch': epoch + 1,  # when resuming, we will start at the next epoch
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict()
                                }, 
                                f"./{args.save_model_dir}_{epoch+1}.ckpt")

    else:
        print("Model in Inference mode \n")
        model_path = f"./{args.save_model_dir}/{args.test_region}/saved_model_{args.saved_model_epoch}.ckpt"
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            resume_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
        else:
            resume_epoch = 0
            
        print("Evaluation at epoch: ", resume_epoch)
        
        ## Run prediciton on test dataset
        pred_patches_dict = run_pred(model, test_loader)
        
        ## Stitch image patches back together
        rgb_stitched, pred_stitched = stitch_patches(pred_patches_dict, cropped_val_test_data_path)
        
        ## Remove border padding
        rgb_unpadded = center_crop(stictched_data = rgb_stitched, image = True, meta = META_DATA)
        pred_unpadded = center_crop(stictched_data = pred_stitched, image = False, meta = META_DATA)
        ## Change to gt data label format
        pred_unpadded = np.where(pred_unpadded == 2, 0, pred_unpadded)
        
        ## Load GT data for evaluation
        gt_labels = np.load(f"./data/repo/FloodNetData/{args.test_region[:-5]}_labels.npy")
        
        ## Run Evaluation
        print("############Starting Model Evalution###############")
        elev_eval.run_eval(pred_unpadded, gt_labels)
        print("####################################################")
        
        ## Save model output
        flood = np.where(pred_unpadded == 1, 1, 0)
        flood = np.expand_dims(flood, axis = -1)
        flood = flood*np.array([ [ [255, 0, 0] ] ])
        # print(flood.shape)
        
        combined = (flood).astype('uint8')
        # print(combined.shape)
        blended = cv2.addWeighted(rgb_unpadded, 0.9, combined, 0.22, 0).astype('uint8')
        cv2.imwrite(f"./{args.out_dir}/{args.test_region[:-5]}_overlay.jpg", cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Running EvaNet')
    
    parser.add_argument('--mode', type=str, required = True, help="--mode can 'training' or 'testing' ")
    parser.add_argument('--data_path', default = "./data/repo/FloodNetData" , type=str, help="--data_path is the path to the dataset ")
    parser.add_argument('--train_region', default = "Region_1_3_TRAIN" , type=str, help="Which two regions to use for training")
    parser.add_argument('--test_region', default = "Region_2_TEST" , type=str, help="Which region to use for testing")
    parser.add_argument('--batch_size', default = 4 , type = int, help = "Image patches per batch")
    parser.add_argument('--input_channel', default = 4 , type = int, help = "Input Channel of Image")
    parser.add_argument('--pred_channel', default = 3 , type = int, help = "No. of channels in output prediction")
    parser.add_argument('--epochs', default = 350 , type = int, help = "No. of epochs to train model")
    parser.add_argument('--val_freq', default = 10 , type = int, help = "No. times to run evaluation during training")
    parser.add_argument('--save_freq', default = None , type = int, help = "No. times to save model during training")
    parser.add_argument('--small_model', default = True , type = bool, help = "3-Layer vs 4_Layer model")
    parser.add_argument('--lr', default = 1e-6 , type = float, help = "model learning rate")
    parser.add_argument('--saved_model_epoch', default = 250 , type = int, help = "The epoch of the last saved model")
    parser.add_argument('--out_dir', default = "./output" , type = str, help = "Output directory of model")
    parser.add_argument('--save_model_dir', default = "./saved_models" , type = str, help = "Directory for saved model")
    
    
    args = parser.parse_args()
    main(args)
    
    
    