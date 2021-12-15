from PIL import Image, ImageOps
from util import util
import functools
import numpy as np


def get_color_count(img, color_palette):
    """ Returns the number of colors found in the color palette

    Parameters: 
        img (image)     -- Pillow image

    Returns:
        colors (list(int, (int,int,int)))   -- list of colors as a tuple and the number of occurrences of the color
        count (int)                         -- number of colors in the image that are found in the color palette
   
    """
    w, h = img.size
    # get all unique colors in image and their count 
    colors = img.getcolors(maxcolors=w * h)

    # count how many of the colors are in the nes palette
    return colors, functools.reduce(lambda x, y: (x + (1 if y[1] in color_palette else 0)), colors, 0)


def _get_distance_between_palettes(img, console_palette):
    from scipy.spatial.distance import cdist
    import sys
    w, h = img.size
    img_colors = np.asarray(img).astype(int)
    img_colors = img_colors.reshape(w * h, 3)

    cm = 255
    console_palette = np.array(console_palette).astype(int)
    img_colors = img_colors
    max_points = np.array([[0, 0, 0], [0, 0, cm], [0, cm, 0], [0, cm, cm],
                           [cm, 0, 0], [cm, 0, cm], [cm, cm, 0], [cm, cm, cm]]).astype(int)

    min_dists = cdist(img_colors, console_palette)
    pct_exact = np.sum(min_dists < 1.) / float(len(min_dists))
    cf = 1.0 / len(console_palette)
    exact_match_factor = max(0., min(1.0, (cf + pct_exact * (1.-cf))))

    min_dist_sum = np.sum(np.min(min_dists, axis=1))
    max_dists = cdist(img_colors, max_points)
    max_dist_sum = np.sum(np.max(max_dists, axis=1))
    score = (max_dist_sum - min_dist_sum) / max_dist_sum

    return score * exact_match_factor


def get_color_distance_score_from_nes_palette(img):
    w, h = img.size
    # get all unique colors in image and their count
    colors = img.getcolors(maxcolors=w * h)
    colors = [[c, (r/255.), (g/255.), (b/255.)] for c, (r, g, b) in colors]
    nes_colors = np.array([(r/255.), (g/255.), (b/255.)] for (r, g, b) in util.get_nes_color_palette())
    x_colors = []
    for [c, cr, cg, cb] in colors:
        x_colors.extend([cr, cg, cb] * c)

    x_colors = np.array(x_colors)
    from scipy.spatial.distance import cdist
    dists = cdist(x_colors, nes_colors)
    min_dists = np.min(dists, axis=1)
    score = np.sum(min_dists)
    return score
        #for (nr, nb, ng) in nes_colors:


            #score += = np.sqrt((cr-nr)**2 + (cg-ng)**2 + (cg-ng)**2)


def compute_nes_color_score(img):
    """ Returns the ratio of NES colors to the total number of colors in the image

    Parameters: 
        img (image)     -- Pillow image

    Returns:
        count (float)   -- ratio of NES colors
   
    """

    score = _get_distance_between_palettes(img, util.get_nes_color_palette())
    return score
    """
    colors, nes_color_count = get_color_count(img, util.get_nes_color_palette())
    total_color_count = len(colors)
    return nes_color_count / total_color_count
    """


def compute_snes_color_score(img):
    """ Returns the ratio of SNES colors to the total number of colors in the image

    Parameters: 
        img (image)     -- Pillow image

    Returns:
        count (float)   -- ratio of SNES colors
   
    """
    score = _get_distance_between_palettes(img, util.get_snes_color_palette())
    return score

    # colors, snes_color_count = get_color_count(img, util.get_snes_color_palette())
    w, h = img.size
    colors = np.array(img.getcolors(maxcolors=w * h))
    total_color_count = len(colors)
    invalid_color_count = np.sum([((r & 0x03) & (g & 0x03) & (b & 0x03)) for (_, (r, g, b)) in colors])  # zero out valid bits, leaving only invalid bits
    snes_color_count = total_color_count - invalid_color_count  # count remaining colors with invalid bits
    return snes_color_count / total_color_count
