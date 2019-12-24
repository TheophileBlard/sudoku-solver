from sudoku_solver.grid.extract import extract_grid, plot_digits
from sudoku_solver.utils import load_img, plot_img


# Prepare pipeline


# Launch pipeline

img = load_img('datasets/mathworks/segmentation_data/raw_data/images/0010_05.jpg')
plot_img(img)

digits = extract_grid(img) # we also return img because it may have i have been resized.
plot_digits(digits)
