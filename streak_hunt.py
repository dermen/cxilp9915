
import numpy as np
import pylab as plt
from astride import Streak
import h5py


# load some images
f = h5py.File('hits4.hdf5', 'r') 
imgs = f['images']
img_sh = imgs[0].shape

# define coordinates of each pixel for checking later
Y,X = np.indices( img_sh)
pix_pts = np.array( zip(X.ravel(), Y.ravel() ) )

def mask_streak( img):
#   make the streak detector
    streak = Streak(img, output_path='.')
    streak.detect()
     
#   make a mask from the streaks
    edges = streak.streaks

#   these vertices define polygons surrounding each streak
    verts = [ np.vstack(( edge['x'], edge['y'])).T 
        for edge in edges]

#   make a path object corresponging to each polygon
    paths = [ plt.mpl.path.Path(v) 
        for v in verts ]

#   check if pixel is contained in ANY of the streak polygons
    contains = np.vstack( [ p.contains_points(points) 
        for p in paths ])
    mask = np.any( contains,0).reshape( img_sh)

    return np.logical_not(mask)

fig, ax = plt.subplots( 1,2, figsize=(12,12))



