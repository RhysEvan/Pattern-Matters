import pattern, surface, projection

import cv2

# height: x axis, width: y axis
height = 72
width = 128
n_periods = 8

depth = height/8

patt1 = pattern.Lines(height, width, n_periods=n_periods)
patt2 = pattern.LinesSaw(height, width, n_periods=n_periods)

surf = surface.Waves(height, width, n_periods=3)
# surf.plot(z_max=2)

# proj = projection.shift_pixels(patt, surf, depth=depth)
# proj_alt = projection.shift_pixels(patt_alt, surf, depth=depth)
Ycoords = projection.shifted_Ycoords(surf.Z, surf.Y, depth=depth)
proj1 = pattern.Lines(height, width, n_periods=n_periods, Ycoords=Ycoords)
proj2 = pattern.LinesSaw(height, width, n_periods=n_periods, Ycoords=Ycoords)

patt1.imshow("Pattern 1", size = 512)
patt2.imshow("Pattern 2", size = 512)
proj1.imshow("Projection 1", size = 512)
proj2.imshow("Projection 2", size = 512)

cv2.waitKey()
cv2.destroyAllWindows()