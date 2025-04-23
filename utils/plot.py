import torch
import matplotlib.pyplot as plt
from utils.utils import load_json
CMAP_VO = plt.cm.get_cmap('jet', 28)
def get_colormap(n_classes=28):
    from matplotlib.colors import ListedColormap
    import numpy as np
    # Define the colors for each class
    colors = [
        [255, 0, 0],  # person (red)
        [0, 0, 255],  # vehicle (blue)
        [160, 139, 19],  # outdoor_things (saddle brown)
        [255, 100, 0],  # animal (orange)
        [128, 0, 128],  # accessory (purple)
        [255, 20, 147],  # sports (deep pink)
        [255, 192, 203],  # kitchen (pink)
        [255, 165, 0],  # food (orange)
        [50, 50, 0],  # indoor (light blue)
        [128, 0, 255],  # appliance (grey)
        [0, 255, 255],  # electronic (cyan)
        [0, 82, 45],  # furniture (sienna)
        [100, 0, 169],  # rawmaterial (dark grey)
        [75, 0, 130],  # textile (indigo)
        [135, 206, 250],  # window (sky blue)
        [255, 228, 181],  # floor (moccasin)
        [255, 240, 245],  # ceiling (lavender blush)
        [128, 128, 0],  # wall (olive)
        [169, 169, 169],  # building (dark grey)
        [112, 128, 144],  # structural (slate grey)
        [0, 128, 0],  # plant (green)
        [135, 206, 235],  # sky (light blue)
        [200, 82, 45],  # ground (sienna)
        [0, 0, 139],  # water (dark blue)
        [50, 105, 30],  # solid (chocolate)
        [192, 0, 192],  # other (silver)
        [50, 69, 19],  # door (saddle brown)
        [10, 10, 10],  # stairs (tomato)
    ]


    # Normalize colors to [0, 1] range for matplotlib
    colors = np.array(colors) / 255.0
    # RGB to BGR
    colors = colors[:, ::-1]
    # Create the colormap
    cmap = ListedColormap(colors)
    return cmap #plt.cm.get_cmap('tab20b_r', n_classes)
def map_colors(image):
    mapped_image = CMAP_VO(image)
    return mapped_image

def create_legend(cmap,  labels = load_json('src/data/cocostuff24.json')): #TODO: remove hardcoding

    # Create a figure and a subplot
    fig, ax = plt.subplots(figsize=(3, 10))  # Adjust the figure size for a vertical legend
    num_labels = len(labels)

    # Generate a range of numbers which will be used for the labels
    data = torch.arange(num_labels).unsqueeze(1)  # Change to vertical by using unsqueeze(1)


    # Display the data
    #cax = ax.imshow(data, cmap=CMAP_VO, aspect="auto")
    ax.axis('off')  # Turn off axis
    norm = plt.Normalize(vmin=0, vmax=28)
    # Create a colorbar with label names, oriented vertically
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', ticks=range(num_labels))

    cbar.set_ticks([i + 0.5 for i in range(num_labels)])  # Adjust tick positions if labels are misaligned
    cbar.ax.set_yticklabels([labels[str(i)] for i in range(num_labels)])  # Map numerical labels to string labels in the JSON
    cbar.ax.tick_params(axis='y', labelrotation=0)  # Ensure labels are horizontal

    cbar.ax.set_position([0.1, 0.1, 0.5, 0.8])  # [left, bottom, width, height]

    # Save the figure
    fig.savefig('legend_cocostuff.png')

def plot_hist(data,bins = 100, title= 'Histogram'):
    min = torch.min(data)
    max = torch.max(data)
    hist = torch.histc(data, bins=bins, min=min, max=max)
    # x axis should be the actual values
    x = torch.linspace(min, max, bins)
    plt.bar(x, hist, align='center')
    plt.xlabel('Bins')
    plt.savefig(title + '.png')

def plot_and_save(data, filename, bins = 100):
    data = data.detach().cpu().squeeze().flatten()
    min = torch.min(data)
    max = torch.max(data)
    hist = torch.histc(data, bins=bins, min=min, max=max)
    # x axis should be the actual values
    x = torch.linspace(min, max, bins)
    # change width to be dependent on the length of the data
    width = (max - min)/ bins
    plt.bar(x, hist, width=width, align='center')
    plt.xlabel('Bins')
    plt.title(filename)
    plt.savefig(filename)
    plt.close()