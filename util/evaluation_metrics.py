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


def compute_nes_color_ratio(img):
    """ Returns the ratio of NES colors to the total number of colors in the image

    Parameters: 
        img (image)     -- Pillow image

    Returns:
        count (float)   -- ratio of NES colors
   
    """

    colors, nes_color_count = get_color_count(img, util.get_nes_color_palette())
    total_color_count = len(colors)
    return nes_color_count / total_color_count


def compute_snes_color_ratio(img):
    """ Returns the ratio of SNES colors to the total number of colors in the image

    Parameters: 
        img (image)     -- Pillow image

    Returns:
        count (float)   -- ratio of SNES colors
   
    """
    # colors, snes_color_count = get_color_count(img, util.get_snes_color_palette())
    w, h = img.size
    colors = np.array(img.getcolors(maxcolors=w * h))
    total_color_count = len(colors)
    invalid_color_count = np.sum([((r & 0x03) & (g & 0x03) & (b & 0x03)) for (_, (r, g, b)) in colors])  # zero out valid bits, leaving only invalid bits
    snes_color_count = total_color_count - invalid_color_count  # count remaining colors with invalid bits
    return snes_color_count / total_color_count
