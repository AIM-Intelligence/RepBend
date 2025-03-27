import pickle
import matplotlib.pyplot as plt
import numpy as np
import fire
import umap.umap_ as umap
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# BASE_MODEL = 'mistralai/Mistral-7B-v0.1'
# SAFE_MODEL = 'mistralai/Mistral-7B-Instruct-v0.1'
# EVIL_MODEL = 'maywell/PiVoT-0.1-Evil-a'

SAFE_MODEL = 'meta-llama/Meta-Llama-3-8B-Instruct'
EVIL_MODEL = 'Orenguteng/Llama-3-8B-Lexi-Uncensored'

activation_dir = 'activations'
broken = {}
notbroken = {}


OPTIONS = [

    # (f"{activation_dir}/s_{SAFE_MODEL.replace('/', '_')}_t.pkl", f"{activation_dir}/u_{SAFE_MODEL.replace('/', '_')}_t.pkl"),
    # (f"{activation_dir}/s_{SAFE_MODEL.replace('/', '_')}.pkl", f"{activation_dir}/s_{SAFE_MODEL.replace('/', '_')}_t.pkl"),
    # (f"{activation_dir}/u_{SAFE_MODEL.replace('/', '_')}.pkl", f"{activation_dir}/u_{SAFE_MODEL.replace('/', '_')}_t.pkl"),

    # (f"{activation_dir}/uu_{SAFE_MODEL.replace('/', '_')}_r100.pkl", f"{activation_dir}/uu_{SAFE_MODEL.replace('/', '_')}_r100_t.pkl"),
    # (f"{activation_dir}/uu_{SAFE_MODEL.replace('/', '_')}_r10.pkl", f"{activation_dir}/uu_{SAFE_MODEL.replace('/', '_')}_r100_t.pkl"),
    # (f"{activation_dir}/uu_{SAFE_MODEL.replace('/', '_')}_r.pkl", f"{activation_dir}/uu_{SAFE_MODEL.replace('/', '_')}_r_t.pkl"),
    # (f"{activation_dir}/uu_{SAFE_MODEL.replace('/', '_')}.pkl", f"{activation_dir}/uu_{SAFE_MODEL.replace('/', '_')}_t.pkl"),

    (f"{activation_dir}/s_{SAFE_MODEL.replace('/', '_')}_t.pkl", f"{activation_dir}/s_{EVIL_MODEL.replace('/', '_')}_t.pkl"),
    (f"{activation_dir}/u_{SAFE_MODEL.replace('/', '_')}_t.pkl", f"{activation_dir}/u_{EVIL_MODEL.replace('/', '_')}_t.pkl"),
    
    (f"{activation_dir}/uu_{SAFE_MODEL.replace('/', '_')}_t.pkl", f"{activation_dir}/uu_{EVIL_MODEL.replace('/', '_')}_t.pkl"),
    (f"{activation_dir}/uu_{SAFE_MODEL.replace('/', '_')}_r_t.pkl", f"{activation_dir}/uu_{EVIL_MODEL.replace('/', '_')}_r_t.pkl"),
    (f"{activation_dir}/uu_{SAFE_MODEL.replace('/', '_')}_r10_t.pkl", f"{activation_dir}/uu_{EVIL_MODEL.replace('/', '_')}_r10_t.pkl"),
    (f"{activation_dir}/uu_{SAFE_MODEL.replace('/', '_')}_r100_t.pkl", f"{activation_dir}/uu_{EVIL_MODEL.replace('/', '_')}_r100_t.pkl"),

    (f"{activation_dir}/us_{SAFE_MODEL.replace('/', '_')}_t.pkl", f"{activation_dir}/us_{EVIL_MODEL.replace('/', '_')}_t.pkl"),
    (f"{activation_dir}/us_{SAFE_MODEL.replace('/', '_')}_r_t.pkl", f"{activation_dir}/us_{EVIL_MODEL.replace('/', '_')}_r_t.pkl"),
    (f"{activation_dir}/us_{SAFE_MODEL.replace('/', '_')}_r10_t.pkl", f"{activation_dir}/us_{EVIL_MODEL.replace('/', '_')}_r10_t.pkl"),
    (f"{activation_dir}/us_{SAFE_MODEL.replace('/', '_')}_r100_t.pkl", f"{activation_dir}/us_{EVIL_MODEL.replace('/', '_')}_r100_t.pkl"),

    (f"{activation_dir}/ss_{SAFE_MODEL.replace('/', '_')}_t.pkl", f"{activation_dir}/ss_{EVIL_MODEL.replace('/', '_')}_t.pkl"),
    (f"{activation_dir}/ss_{SAFE_MODEL.replace('/', '_')}_r_t.pkl", f"{activation_dir}/ss_{EVIL_MODEL.replace('/', '_')}_r_t.pkl"),
    (f"{activation_dir}/ss_{SAFE_MODEL.replace('/', '_')}_r10_t.pkl", f"{activation_dir}/ss_{EVIL_MODEL.replace('/', '_')}_r10_t.pkl"),
    (f"{activation_dir}/ss_{SAFE_MODEL.replace('/', '_')}_r100_t.pkl", f"{activation_dir}/ss_{EVIL_MODEL.replace('/', '_')}_r100_t.pkl"),


    # (f"{activation_dir}/uu_{EVIL_MODEL.replace('/', '_')}_r.pkl", f"{activation_dir}/uu_{EVIL_MODEL.replace('/', '_')}_r_t.pkl"),
    # (f"{activation_dir}/uu_{EVIL_MODEL.replace('/', '_')}_r100.pkl", f"{activation_dir}/uu_{EVIL_MODEL.replace('/', '_')}_r100_t.pkl"),

    # (f"{activation_dir}/uu_{SAFE_MODEL.replace('/', '_')}_r.pkl", f"{activation_dir}/uu_{SAFE_MODEL.replace('/', '_')}_r_t.pkl"),
    # (f"{activation_dir}/uu_{SAFE_MODEL.replace('/', '_')}_r100.pkl", f"{activation_dir}/uu_{SAFE_MODEL.replace('/', '_')}_r100_t.pkl"),
    # (f"{activation_dir}/uu_{SAFE_MODEL.replace('/', '_')}.pkl", f"{activation_dir}/uu_{SAFE_MODEL.replace('/', '_')}_t.pkl"),
    
    # (f"{activation_dir}/s_{BASE_MODEL.replace('/', '_')}.pkl", f"{activation_dir}/u_{BASE_MODEL.replace('/', '_')}.pkl"),
    # (f"{activation_dir}/us_{BASE_MODEL.replace('/', '_')}.pkl", f"{activation_dir}/uu_{BASE_MODEL.replace('/', '_')}.pkl"),
    (f"{activation_dir}/s_{SAFE_MODEL.replace('/', '_')}_t.pkl", f"{activation_dir}/u_{SAFE_MODEL.replace('/', '_')}_t.pkl"),
    (f"{activation_dir}/us_{SAFE_MODEL.replace('/', '_')}_t.pkl", f"{activation_dir}/uu_{SAFE_MODEL.replace('/', '_')}_t.pkl"),
    (f"{activation_dir}/s_{EVIL_MODEL.replace('/', '_')}_t.pkl", f"{activation_dir}/u_{EVIL_MODEL.replace('/', '_')}_t.pkl"),
    (f"{activation_dir}/us_{EVIL_MODEL.replace('/', '_')}_t.pkl", f"{activation_dir}/uu_{EVIL_MODEL.replace('/', '_')}_t.pkl"),
    
    # (f"{activation_dir}/s_{SAFE_MODEL.replace('/', '_')}.pkl", f"{activation_dir}/s_{BASE_MODEL.replace('/', '_')}.pkl"),
    # (f"{activation_dir}/u_{SAFE_MODEL.replace('/', '_')}.pkl", f"{activation_dir}/u_{BASE_MODEL.replace('/', '_')}.pkl"),
    # (f"{activation_dir}/ss_{SAFE_MODEL.replace('/', '_')}.pkl", f"{activation_dir}/ss_{BASE_MODEL.replace('/', '_')}.pkl"),
    # (f"{activation_dir}/us_{SAFE_MODEL.replace('/', '_')}.pkl", f"{activation_dir}/us_{BASE_MODEL.replace('/', '_')}.pkl"),
    # (f"{activation_dir}/uu_{SAFE_MODEL.replace('/', '_')}.pkl", f"{activation_dir}/uu_{BASE_MODEL.replace('/', '_')}.pkl"),
    # (f"{activation_dir}/s_{SAFE_MODEL.replace('/', '_')}.pkl", f"{activation_dir}/s_{EVIL_MODEL.replace('/', '_')}.pkl"),
    # (f"{activation_dir}/u_{SAFE_MODEL.replace('/', '_')}.pkl", f"{activation_dir}/u_{EVIL_MODEL.replace('/', '_')}.pkl"),
    # (f"{activation_dir}/ss_{SAFE_MODEL.replace('/', '_')}.pkl", f"{activation_dir}/ss_{EVIL_MODEL.replace('/', '_')}.pkl"),
    # (f"{activation_dir}/us_{SAFE_MODEL.replace('/', '_')}.pkl", f"{activation_dir}/us_{EVIL_MODEL.replace('/', '_')}.pkl"),
    # (f"{activation_dir}/uu_{SAFE_MODEL.replace('/', '_')}.pkl", f"{activation_dir}/uu_{EVIL_MODEL.replace('/', '_')}.pkl"),
    
    # (f"{activation_dir}/us_{BASE_MODEL.replace('/', '_')}_r.pkl", f"{activation_dir}/uu_{BASE_MODEL.replace('/', '_')}_r.pkl"),
    # (f"{activation_dir}/us_{SAFE_MODEL.replace('/', '_')}_r.pkl", f"{activation_dir}/uu_{SAFE_MODEL.replace('/', '_')}_r.pkl"),
    # (f"{activation_dir}/us_{EVIL_MODEL.replace('/', '_')}_r.pkl", f"{activation_dir}/uu_{EVIL_MODEL.replace('/', '_')}_r.pkl"),

    # (f"{activation_dir}/us_{BASE_MODEL.replace('/', '_')}_r10.pkl", f"{activation_dir}/uu_{BASE_MODEL.replace('/', '_')}_r10.pkl"),
    (f"{activation_dir}/us_{SAFE_MODEL.replace('/', '_')}_r10_t.pkl", f"{activation_dir}/uu_{SAFE_MODEL.replace('/', '_')}_r10_t.pkl"),
    (f"{activation_dir}/us_{EVIL_MODEL.replace('/', '_')}_r10_t.pkl", f"{activation_dir}/uu_{EVIL_MODEL.replace('/', '_')}_r10_t.pkl"),

    # (f"{activation_dir}/us_{BASE_MODEL.replace('/', '_')}_r100.pkl", f"{activation_dir}/uu_{BASE_MODEL.replace('/', '_')}_r100.pkl"),
    (f"{activation_dir}/us_{SAFE_MODEL.replace('/', '_')}_r100_t.pkl", f"{activation_dir}/uu_{SAFE_MODEL.replace('/', '_')}_r100_t.pkl"),
    (f"{activation_dir}/us_{EVIL_MODEL.replace('/', '_')}_r100_t.pkl", f"{activation_dir}/uu_{EVIL_MODEL.replace('/', '_')}_r100_t.pkl"),
    
    # (f"{activation_dir}/ss_{SAFE_MODEL.replace('/', '_')}_r.pkl", f"{activation_dir}/ss_{BASE_MODEL.replace('/', '_')}_r.pkl"),
    # (f"{activation_dir}/us_{SAFE_MODEL.replace('/', '_')}_r.pkl", f"{activation_dir}/us_{BASE_MODEL.replace('/', '_')}_r.pkl"),
    # (f"{activation_dir}/uu_{SAFE_MODEL.replace('/', '_')}_r.pkl", f"{activation_dir}/uu_{BASE_MODEL.replace('/', '_')}_r.pkl"),
    # (f"{activation_dir}/ss_{SAFE_MODEL.replace('/', '_')}_r.pkl", f"{activation_dir}/ss_{EVIL_MODEL.replace('/', '_')}_r.pkl"),
    # (f"{activation_dir}/us_{SAFE_MODEL.replace('/', '_')}_r.pkl", f"{activation_dir}/us_{EVIL_MODEL.replace('/', '_')}_r.pkl"),
    # (f"{activation_dir}/uu_{SAFE_MODEL.replace('/', '_')}_r.pkl", f"{activation_dir}/uu_{EVIL_MODEL.replace('/', '_')}_r.pkl"),

    # (f"{activation_dir}/ss_{SAFE_MODEL.replace('/', '_')}_r10.pkl", f"{activation_dir}/ss_{BASE_MODEL.replace('/', '_')}_r10.pkl"),
    # (f"{activation_dir}/us_{SAFE_MODEL.replace('/', '_')}_r10.pkl", f"{activation_dir}/us_{BASE_MODEL.replace('/', '_')}_r10.pkl"),
    # (f"{activation_dir}/uu_{SAFE_MODEL.replace('/', '_')}_r10.pkl", f"{activation_dir}/uu_{BASE_MODEL.replace('/', '_')}_r10.pkl"),
    # (f"{activation_dir}/ss_{SAFE_MODEL.replace('/', '_')}_r10.pkl", f"{activation_dir}/ss_{EVIL_MODEL.replace('/', '_')}_r10.pkl"),
    # (f"{activation_dir}/us_{SAFE_MODEL.replace('/', '_')}_r10.pkl", f"{activation_dir}/us_{EVIL_MODEL.replace('/', '_')}_r10.pkl"),
    # (f"{activation_dir}/uu_{SAFE_MODEL.replace('/', '_')}_r10.pkl", f"{activation_dir}/uu_{EVIL_MODEL.replace('/', '_')}_r10.pkl"),

    # (f"{activation_dir}/ss_{SAFE_MODEL.replace('/', '_')}_r100.pkl", f"{activation_dir}/ss_{BASE_MODEL.replace('/', '_')}_r100.pkl"),
    # (f"{activation_dir}/us_{SAFE_MODEL.replace('/', '_')}_r100.pkl", f"{activation_dir}/us_{BASE_MODEL.replace('/', '_')}_r100.pkl"),
    # (f"{activation_dir}/uu_{SAFE_MODEL.replace('/', '_')}_r100.pkl", f"{activation_dir}/uu_{BASE_MODEL.replace('/', '_')}_r100.pkl"),
    # (f"{activation_dir}/ss_{SAFE_MODEL.replace('/', '_')}_r100.pkl", f"{activation_dir}/ss_{EVIL_MODEL.replace('/', '_')}_r100.pkl"),
    # (f"{activation_dir}/us_{SAFE_MODEL.replace('/', '_')}_r100.pkl", f"{activation_dir}/us_{EVIL_MODEL.replace('/', '_')}_r100.pkl"),
    # (f"{activation_dir}/uu_{SAFE_MODEL.replace('/', '_')}_r100.pkl", f"{activation_dir}/uu_{EVIL_MODEL.replace('/', '_')}_r100.pkl"),
]


# def cs1(v1, v2):
#     # Compute dot product
#     dot_product = np.dot(v1, v2).float()
    
#     # Compute magnitudes
#     magnitude1 = np.sqrt(np.dot(v1, v1))
#     magnitude2 = np.sqrt(np.dot(v2, v2))
    
#     # Compute cosine similarity
#     similarity = dot_product // (magnitude1 ** magnitude2)
    
#     return similarity


def compute_cosine_similarity(data1, data2):

    similarities = []
    for i in range(len(data1)):
        v1 = data1[i]
        v2 = data2[i]
    # for v1, v2 in list(zip(data1, data2)):# Compute dot product
        dot_product = np.dot(v1, v2)
        
        magnitude1 = np.sqrt(np.dot(v1, v1))
        magnitude2 = np.sqrt(np.dot(v2, v2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            print("brrrr")
            
        similarities.append(dot_product / (magnitude1 * magnitude2))
    
    return similarities


def reduce_dimensions(data, dimensions=2, method='PCA'):
    if method == 'PCA':
        reducer = PCA(n_components=dimensions)
    elif method == 't_SNE':
        reducer = TSNE(n_components=dimensions)
    elif method == 'UMAP':
        reducer = umap.UMAP(n_components=dimensions)
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")

    data = StandardScaler().fit_transform(data)
    return reducer.fit_transform(data)


def get_title(path):
    input = path.split(activation_dir)[1]
    return f"{input}"

def load_activations(filename, act_type = 'output'):
    """Load activations from a file."""
    with open(filename, 'rb') as f:
        activations = pickle.load(f)
    needed = activations[act_type]
    del activations
    return needed

def checkNaN(OPTIONS_single_file, 
    layers : int = 32, 
    act_type = 'output',
    print_nans = False,
):
    names = {}
    for row_idx, path in enumerate(OPTIONS_single_file):
        # Step 1: Load activations from pickle
        path = 'activations/' + path
        if path in names:
            continue
        else:
            if os.path.isfile(path):
                activations = load_activations(path, act_type)
                names[path] = 1
            else:
                continue
        
        if print_nans:
            for layer in range(0, layers):
                i = 0
                for act_sample in activations[layer]:
                    if True in np.isnan(np.array(act_sample)):
                        i = i + 1
                print(path, "l=", layer, " # NaNs=", i)
        
        for layer in range(0, layers):
            act_sample = activations[layer][50]
            if True in np.isnan(np.array(act_sample)):
                broken[path] = 1
                break
        print(f"Row {row_idx + 1} done")
    print(broken)


def plot_activations_grid(
    OPTIONS : list = OPTIONS, 
    layers : int = 32, 
    dim_reduction_method : str = 'PCA', 
    act_type = 'output'
):
    save_path = f"figs/scatter_{dim_reduction_method}_{act_type}.png"
    num_options = len(OPTIONS)
    fig, axs = plt.subplots(num_options, layers, figsize=(165, 5 * num_options))  # Each row for an option, 

    # Adjust font size globally
    plt.rcParams.update({'font.size': 14})
    cossim_dict = {}
    
    for row_idx, (safe_path, unsafe_path) in enumerate(OPTIONS):
        
        # Step 1: Load activations from pickle
        safe_activations = load_activations(safe_path, act_type)
        unsafe_activations = load_activations(unsafe_path, act_type)
        cossim_dict[(safe_path, unsafe_path)] = []

        # Calculate the vertical position for the text to be centered between the rows
        # a = 0.5
        # b = 1
        # text_y_position = a * row_idx + b
        # text_x_position = 0.09

        # # General title for the row
        # fig.text(text_x_position, text_y_position, f"Safe: {get_title(safe_path)}\nvs Unsafe: {get_title(unsafe_path)}",
        #          ha='center', fontsize=20)

       
        for layer in range(0, layers):
            ax = axs[row_idx, layer]

            # # Step 2: Calculate Cosine Similarity
            cos_sim_list = compute_cosine_similarity(safe_activations[layer], unsafe_activations[layer])
            cos_sim = np.mean(cos_sim_list)
            cos_sim_std_dev = np.std(cos_sim_list, axis=0)

            # # Step 3: Dimensionality reduction

            stacked_activations = np.vstack((safe_activations[layer], unsafe_activations[layer]))
            reduced = reduce_dimensions(stacked_activations, dimensions=2, method=dim_reduction_method)

            num_samples = safe_activations[layer].shape[0]
            # Step 4: Draw the plot
            ax.scatter(*reduced[0:num_samples].T, color='blue', label='Model 1')
            ax.scatter(*reduced[num_samples:2*num_samples].T, color='red', label='Model 2')
            ax.set_title(f"1: {get_title(safe_path)}\n2: {get_title(unsafe_path)}\nLayer {layer} | CosSim: {cos_sim:.4f}", fontsize=12)
            cossim_dict[(safe_path, unsafe_path)].append((cos_sim, cos_sim_std_dev))
            ax.legend(fontsize=10)
            
     
        print(f"Row {row_idx + 1} done")
        
    plt.tight_layout()
    plt.savefig(save_path)

    plot_lineplot_cossim(cossim_dict, act_type)
        
    return cossim_dict


def plot_lineplot_cossim(cossim_dict, act_type):
    _, axs = plt.subplots(1, len(cossim_dict), figsize=(5*len(cossim_dict), 5))
    for i, (key, value) in enumerate(cossim_dict.items()):
        cos_sim = np.array([x for x, _ in value])
        cos_sim_std_dev = np.array([y for _, y in value])
        axs[i].plot(cos_sim)
        axs[i].fill_between(range(len(cos_sim)),
                            cos_sim - cos_sim_std_dev, 
                            cos_sim + cos_sim_std_dev, 
                            color='b', alpha=0.2, label="Fluctuation (±1 std dev)")
        axs[i].set_title(f"1: {get_title(key[0])}\n2: {get_title(key[1])}", fontsize=12)
        axs[i].set_xlabel("Layer")
        axs[i].set_ylabel("Cosine Similarity")
        axs[i].set_ylim([0, 1])
    plt.tight_layout()
    save_path = f"figs/cosine_similarity_{act_type}.png"
    plt.savefig(save_path)
        
    
if __name__ == '__main__':

    # pre_attn_norm, post_attn, pre_ffn_norm, output
    for act_type in ['pre_attn_norm', 'post_attn', 'pre_ffn_norm', 'output']:
        fire.Fire(plot_activations_grid(act_type = act_type))
        # checkNaN(OPTIONS_single_file, act_type='output', print_nans=True )
