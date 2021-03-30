import numpy as np
from random import randrange
from random import choice


class ArrowDataset:
    """
    Represents a dataset of small arrow images
    The actual data points are randomly generated
    """

    def __init__(self, h, w, arrow_size, n_samples):
        self.h = h
        self.w = w
        self.arrow_size = arrow_size
        self.arrow_bb = 2 * arrow_size + 1
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.gen_arrow()

    def gen_arrow(self):
        y = randrange(0, self.h - self.arrow_bb)
        x = randrange(0, self.w - self.arrow_bb)
        arrow = np.zeros((self.h, self.w), dtype=bool)

        if choice(['rotate', 'no_rotate']) == 'rotate':
            for i in range(self.arrow_bb):
                arrow[x + i][y + abs(i - self.arrow_size)] = True
                arrow[x + self.arrow_size][y + i] = True
        else:
            for i in range(self.arrow_bb):
                arrow[x + abs(i - self.arrow_size)][y + i] = True
                arrow[x + i][y + self.arrow_size] = True

        return arrow

def show_arrow(arrow, h, w):
    for i in range(h):
        for j in range(w):
            print('x' if arrow[i][j] else 'o', end='')
        print('')
    print('')

if __name__ == "__main__":
    h = 10
    w = 10
    arrow_ds = ArrowDataset(h, w, 2, 5)
    for i in range(len(arrow_ds)):
        arrow = arrow_ds[i]
        show_arrow(arrow, h, w)
