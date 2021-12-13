import argparse
from PIL import Image
from util.evaluation_metrics import compute_nes_color_score, compute_snes_color_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Parse a single run from checkpoints/loss_log.txt')
    parser.add_argument('file', type=str, help='input image filepath')
    parser.add_argument('score_type', type=str, help='snes or nes (depending on which color score to calculate))')
    
    args = parser.parse_args()

    im = Image.open(args.file)
    if args.score_type == 'nes':
        # convert to pillow image
        score = compute_nes_color_score(im)
        print(score)
    elif args.score_type == 'snes':
        score = compute_snes_color_score(im)
        print(score)
    else:
        print('score_type must be nes or snes')
    
