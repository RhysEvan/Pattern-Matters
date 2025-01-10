import pattern, surface, projection

import cv2


# height: x axis, width: y axis
height = 512
width = 512
n_periods = 8
depth = width/8

patt = pattern.Lines(height, width, n_periods=n_periods)

surf = surface.Paraboloid(height, width)
surf.plot(z_max=2)

# disc_proj = projection.shift_pixels(patt, surf)

Ycoords = projection.shifted_Ycoords(surf.Z, surf.Y, depth=depth)
proj = pattern.Lines(height, width, n_periods=n_periods, Ycoords=Ycoords)

patt.imshow("Pattern")
proj.imshow("Projection")
# disc_proj.imshow("Discrete Projection")

cv2.waitKey()
cv2.destroyAllWindows()

# patt.imwrite("lines.png")
# disc_proj.imwrite("crude_projection512.png")
# proj.imwrite("smooth_projection.png")