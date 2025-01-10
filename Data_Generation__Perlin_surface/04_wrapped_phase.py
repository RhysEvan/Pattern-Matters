import image, pattern, surface, projection

import cv2
import numpy as np

# height: x axis, width: y axis
height = 128
width = 128
n_periods = 8
depth = width/8
pattern_class = pattern.Lines

patt = pattern_class(height, width, n_periods=n_periods)

surf = surface.Waves(height, width, n_periods=3) + surface.Waves(height, width, n_periods=5)
surf.normalise()
surf.plot(z_max=2)

# disc_proj = projection.shift_pixels(patt, surf)

Ycoords = projection.shifted_Ycoords(surf.Z, surf.Y, depth=depth)
proj1 = pattern_class(height, width, n_periods=n_periods, Ycoords=Ycoords, phase_deg=0)
proj2 = pattern_class(height, width, n_periods=n_periods, Ycoords=Ycoords, phase_deg=90)
proj3 = pattern_class(height, width, n_periods=n_periods, Ycoords=Ycoords, phase_deg=180)
proj4 = pattern_class(height, width, n_periods=n_periods, Ycoords=Ycoords, phase_deg=270)

num = proj4-proj2
den = proj1-proj3
wrapped_phase = np.arctan(num.array/den.array)
wp = image.Image.from_array(wrapped_phase)
num.normalise()
den.normalise()
wp.normalise()

patt.imshow("Pattern", size = 256)
proj1.imshow("Projection", size = 256)
num.imshow("Numerator", size = 256)
den.imshow("Denominator", size = 256)
wp.imshow("Wrapped phase", size = 256)

cv2.waitKey()
cv2.destroyAllWindows()
