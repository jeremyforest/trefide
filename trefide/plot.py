import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable


def pixelwise_ranks(rank_vec,
                    fov_height,
                    fov_width,
                    num_frames,
                    bheight,
                    bwidth,
                    figsize=(16, 16),
                    save_fig=False,
                    save_fig_path=None,
                    dataset=None):
    """ rank_vec should be column major ordered

    Parameter:
        rank_vec:  rank of each patch
        fov_height: height of field of view
        fov_width: width of field of view
        num_frames: number of frames
        bheight: block height
        bwidth: block width
        figsize: fig size
        save figure: save plot
        save_fig_path: path to save plot to
        dataset: title to prepend
    """

    # Assert Valid Dimensions
    assert fov_height % bheight == 0, "FOV height must be evenly divisible by block width"
    assert fov_width % bwidth == 0, "FOV width must be evenly divisible by block width"

    # Create Figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(np.kron(np.reshape(rank_vec,
                                      (int(fov_height/bheight),
                                       int(fov_width/bwidth)),
                                      order='F'),
                           np.ones((bheight, bwidth))),
                   cmap=cm.Greys_r)
    comp_factor = (fov_height*fov_width*num_frames) /\
        ((bheight*bwidth + num_frames)*np.sum(rank_vec))
    if dataset is None:
        ax.set_title("Pixel-Wise Ranks, Compression Factor: %.1f" % comp_factor)
    else:
        ax.set_title(dataset + " Pixel-Wise Ranks, Compression Factor: %.1f" % comp_factor)
    plt.colorbar(im, cax, orientation='vertical')
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'{save_fig_path}/pixelwise_ranks_plot.png')
    else:
        plt.show()
