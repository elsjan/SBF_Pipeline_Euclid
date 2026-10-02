import os
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_same_name_pngs(
    base_dir,
    filename="H.png",
    output_png="combined.png",
    ncols=10,              # 10 columns → good for ~90 images
    panel_size=2.2,        # inches per panel
    dpi=300,
    title_fontsize=8
):
    image_paths = []
    labels = []

    for folder in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, folder, filename)
        if os.path.isfile(path):
            image_paths.append(path)
            labels.append(folder)

    n_images = len(image_paths)
    if n_images == 0:
        raise ValueError("No matching PNG images found.")

    nrows = math.ceil(n_images / ncols)
    figsize = (ncols * panel_size, nrows * panel_size)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for ax, img_path, label in zip(axes, image_paths, labels):
        ax.text(
            0.5, 0.98, label,
            transform=ax.transAxes,
            fontsize=title_fontsize,
            va="top", ha="center",
            color="black"
        )
        img = mpimg.imread(img_path)
        ax.imshow(img)
#         ax.set_title(label, fontsize=title_fontsize)
        
        ax.axis("off")

    for ax in axes[n_images:]:
        ax.axis("off")

    plt.subplots_adjust(
        left=0.005, right=0.995,
        bottom=0.005, top=0.99,
        wspace=0.0,
        hspace=0.02
    )
#     plt.subplots_adjust(
#     left=0.005, right=0.995,
#     bottom=0.005, top=0.96,
#     wspace=0.0,
#     hspace=0.08
#     )

    plt.savefig(output_png, dpi=dpi)#,layout="constrained")
    plt.close(fig)
