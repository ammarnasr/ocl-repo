import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
from tqdm.auto import tqdm

def get_representations(network, dataloader, num_samples=-1, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = dataloader.batch_size
    num_batches = len(dataloader)
    total_images = num_batches * batch_size

    if num_samples == -1:
        num_samples = total_images

    network.eval()
    
    all_representations = []
    all_labels = []

    with torch.no_grad():
        batches_bar = tqdm(dataloader, desc=f'Getting Representations', position=0, leave=True)
        counter = 0
        for batch in batches_bar:

            # Get batch data
            images = batch[0].to(device)
            labels = batch[1].to(device)

            # Forward pass
            [kasami_preds, outputs] = network(images)

            all_representations.append(kasami_preds)
            all_labels.append(labels)
            counter += batch_size

            batches_bar.set_description(f'Getting Representations: {counter}/{num_samples}')
            batches_bar.refresh()

            if counter >= num_samples:
                break

    all_representations = torch.cat(all_representations, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_representations, all_labels


def plot_representations_tsne_2d(representations, labels, title=None, save_path=None, show=True):

    # Convert to numpy arrays
    representations = representations.cpu().numpy()
    labels = labels.cpu().numpy()

    # Get 2D representation
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(representations)


    # Plot
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x=tsne_results[:,0], y=tsne_results[:,1],
        hue=labels,
        palette=sns.color_palette("hls", len(np.unique(labels))),
        legend="full",
        alpha=0.3
    )
    plt.title(title)
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()



