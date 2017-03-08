'''
Show all different interpolation methods for imshow
'''

import matplotlib.pyplot as plt
import numpy as np

# from the docs:

# If interpolation is None, default to rc image.interpolation. See also
# the filternorm and filterrad parameters. If interpolation is 'none', then
# no interpolation is performed on the Agg, ps and pdf backends. Other
# backends will fall back to 'nearest'.
#
# http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.imshow

methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
           'catrom', 'gaussian', 'bessel', 'mitchell']

np.random.seed(0)
grid = np.random.rand(4, 4)

fig, axes = plt.subplots(4, 3, figsize=(12, 6),
                         subplot_kw={'xticks': [], 'yticks': []})


fig.subplots_adjust(hspace=0.3)#, wspace=0.05)

for ax, interp_method in zip(axes.flat, methods):
    img = ax.imshow(grid, interpolation=interp_method, cmap='seismic')
    #fig.colorbar(img, ticks=[-1, 0, 1])
    ax.set_title(interp_method)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(img, cax=cbar_ax)



plt.show()