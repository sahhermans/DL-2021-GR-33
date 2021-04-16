# Deep Learning

from code import probabilities_to_decision
import torch
import numpy as np
import re
from PIL import Image
import glob
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.lines as mlines
from datetime import datetime
import os
from os import path
from skimage.filters import gaussian

def computeShapeTextureBias(model_name, transform, main_folderpath, gaussianFilter=False):
    """
    Computes the shape vs texture bias for a model for different images in multiple subfolders.
    """

    # Load pre-trained model
    model = torch.hub.load('pytorch/vision:v0.9.0', model_name, pretrained=True)
    model.eval()

    # Create list of all subfolders to be computed
    folderpath_list = glob.glob(main_folderpath)

    # Initialise output
    words_pattern = '[a-zA-Z]+'
    shape_bias_cnt = np.zeros(shape = (len(folderpath_list), 1))
    texture_bias_cnt = np.zeros_like(shape_bias_cnt)
    wrong_cnt = np.zeros_like(shape_bias_cnt)
    catagories_list = []

    # Loop over all files in all subfolders and store output
    for i, folder in enumerate(folderpath_list):
        # Save shape of folder
        catagories_list.append(re.findall(words_pattern, folder, flags=re.IGNORECASE)[-1])

        for filename in glob.glob(folder + '/*.png'): 
            # Selects shapes and textures
            string_list = re.findall(words_pattern, filename, flags=re.IGNORECASE)
            shape = string_list[-3]
            texture = string_list[-2]

            # Skip if texture is the same as shape
            if shape == texture:
                continue

            # Open image and write as readable tensor
            im = Image.open(filename)
            # Filtering the image if keyword is given
            if gaussianFilter == True:
                image_arr = np.asarray(im)
                im = gaussian(image_arr, sigma=2, multichannel=True, preserve_range=True)
                im = Image.fromarray(im.astype(np.uint8))
            img_t = transform(im)
            batch_t = torch.unsqueeze(img_t, 0)
            
            # Compute model output and map to one of the sixteen classes
            out = model(batch_t)
            softmax_output = torch.nn.functional.softmax(out, dim=1)[0] 
            softmax_output_numpy = softmax_output.detach().numpy() 
            mapping = probabilities_to_decision.ImageNetProbabilitiesTo16ClassesMapping()
            decision_from_16_classes = mapping.probabilities_to_decision(softmax_output_numpy)

            # Count made decisions
            if decision_from_16_classes == shape:
                shape_bias_cnt[i] += 1
            elif decision_from_16_classes == texture:
                texture_bias_cnt[i] += 1
            else:
                wrong_cnt[i] += 1
                    
        print(catagories_list)
        print(shape_bias_cnt.T)
        print(texture_bias_cnt.T)
        print(wrong_cnt.T)
    
    # Compute shape bias and wronlgy decided images
    shape_bias = shape_bias_cnt/(shape_bias_cnt + texture_bias_cnt)
    correctness = (shape_bias_cnt + texture_bias_cnt)/(shape_bias_cnt + texture_bias_cnt + wrong_cnt)

    return shape_bias, correctness

def saveData(save_name, model_list, shape_bias_arr, correctness_arr):
    # Create directory
    now = datetime.now()
    dir_ = now.strftime(save_name + "-%d_%b_%H_%M_%S")
    os.mkdir(dir_)

    # Saving variables
    np.save(path.join(dir_, "model_list.npy"), model_list)
    np.save(path.join(dir_, "shape_bias_arr.npy"), shape_bias_arr)
    np.save(path.join(dir_, "correctness_arr.npy"), correctness_arr)

    # Feedback
    print("Succesfully saved!")
    # Return path to folder of results
    saved_folder_path = os.path.dirname(os.path.abspath(__file__)) + "\\" + dir_ + "\\"

    return saved_folder_path

def saveDataSimple(model, shape_bias, correctness):
    # Create directory
    dir_ = model
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    # Saving variables
    np.save(path.join(dir_, "shape-bias.npy"), shape_bias)
    np.save(path.join(dir_, "correctness.npy"), correctness)

def computeModelList(model_list, save_folder_name, gaussianFilter=False):

    main_folderpath = r'C:\texture-vs-shape-master\stimuli\style-transfer-preprocessed-512\*'

    # Transform input to be interpretable for models from torchvision
    transform = transforms.Compose([              
    transforms.Resize(256),                    
    transforms.CenterCrop(224),                
    transforms.ToTensor(),                     
    transforms.Normalize(                      
    mean=[0.485, 0.456, 0.406],                
    std=[0.229, 0.224, 0.225]                  
    )])

    # Initialise
    shape_bias_arr = np.zeros(shape=(16, len(model_list)))
    correctness_arr = np.zeros_like(shape_bias_arr)

    # Compute shape bias and correctness
    for i, model_name in enumerate(model_list):
        # Compute model results
        shape_bias, correctness = computeShapeTextureBias(model_name, transform, main_folderpath, gaussianFilter)  

        # Store results in arrays
        shape_bias_arr[:, [i]] = shape_bias
        correctness_arr[:, [i]] = correctness

    saved_folder_path = saveData(save_folder_name, model_list, shape_bias_arr, correctness_arr)

    return saved_folder_path

def plot_figure4(folder_path, saved_fig_name):
    # Load variables
    model_list = np.load(folder_path + "model_list.npy")
    shape_bias_arr = np.load(folder_path + "shape_bias_arr.npy")
    correctness_arr = np.load(folder_path + "correctness_arr.npy")

    # Compute texture bias fraction
    texture_bias_arr = 1 - shape_bias_arr

    # Logo path parameter
    logo_path_png = r"C:\texture-vs-shape-master\data-analysis\category-images\*.png"

    # Colour and shape list "library"
    colour_list = ["b","r","k","g","m"]
    shape_list = ['D','o','s','^','*']

    # Make plotting colour lists apropiate
    colour_list = colour_list[0:len(model_list)]
    shape_list = shape_list[0:len(model_list)]

    # Create framework main figure
    fig = plt.figure()
    gs = fig.add_gridspec(16,18)
    main_fig = fig.add_subplot(gs[:, 1:16])
    main_fig.hlines(np.linspace(1,16,16),0,1, colors='k', linestyles='dotted')
    main_fig.set_xticks(np.linspace(0,1,11))
    main_fig.set_xticks(np.linspace(0,1,11))
    main_fig.set_xlabel(r"Fraction of 'texture' decisions")
    a = plt.gca()
    plt.ylim([0,16])
    plt.xlim([0,1])
    yax = a.axes.get_yaxis()
    yax = yax.set_visible(False)
    plt.grid(True)
    ax2 = main_fig.twiny()
    ax2.set_xlim(main_fig.get_xlim())
    ax2.set_xticks(np.linspace(0,1,11))
    ax2.set_xticklabels([1,.9,.8,.7,.6,.5,.4,.3,.2,.1,0])
    ax2.set_xlabel(r"Fraction of 'shape' decisions")
        
    # Create icons on the left of figure
    for i, filename in enumerate(glob.glob(logo_path_png)):
        fig.add_subplot(gs[15-i, 0])
        plt.axis('off')
        plt.imshow(img.imread(filename))

    # Plot shapebias scatterplot
    for i, model_name, colour_name, shape_name in zip(range(len(model_list)), model_list, colour_list, shape_list):
        main_fig.vlines(np.mean(texture_bias_arr[:, i]),0,len(correctness_arr[:, 0]), colors= colour_name, linewidth=1)
        main_fig.scatter(texture_bias_arr[:, i], np.linspace(0.5,15.5,16),s=100, c = colour_name, marker = shape_name)
    
    # Create corresponding legend
    handles_list = []
    for p in range(len(model_list)):
        handles_list.append(mlines.Line2D([], [], color=colour_list[p], marker=shape_list[p], markersize=10, label=model_list[p], linestyle=''))
        
    main_fig.legend(handles = handles_list, loc='best', fontsize = 9, fancybox=False,  shadow=False, ncol=1)          
    
    # Plot correctness barplots
    for i in range(16):
        fig.add_subplot(gs[i, -1])
        plt.barh(np.arange(len(correctness_arr[0, :]))+1/len(correctness_arr[0, :]),correctness_arr[i, :],color = colour_list)
        plt.hlines(range(len(correctness_arr[0, :])+1),0,1, colors='k', linewidth=0.5)
        plt.vlines([0,1],0,len(correctness_arr[0, :]), colors='k', linewidth=0.5)
        plt.axis('off')
        
    # Save figure
    plt.savefig(path.join(saved_folder_path, saved_fig_name))
    
    # Show plot
    plt.show()

################################################################################

##################################################
# Input parameters 
model_list = ["alexnet", "googlenet", "resnet50", "vgg16"]
saved_folder_name = "sigma-2-original"
saved_fig_name = "Fig4.png"
gaussianFilter = False

##################################################
# Outcome
saved_folder_path = computeModelList(model_list, saved_folder_name, gaussianFilter)
plot_figure4(saved_folder_path, saved_fig_name)
