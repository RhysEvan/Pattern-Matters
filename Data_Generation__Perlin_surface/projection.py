import image, pattern, surface

import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2


def shift_pixels(patt: pattern.Pattern, surf: surface.Surface, depth: float = None,
                 angle_rad: float = None, angle_deg: float = 30) -> image.Image:
    """Shift the pixels of a pattern to simulate a projection on a surface"""
    # Set angle in radians
    angle_rad = angle_rad or angle_deg/180*np.pi
    
    # Set depth for z-axis
    depth = depth or surf.array.shape[1]/4
    
    # Calculate projection shift and round to an integer value
    dy = ((surf.Z-1/2)*np.sin(angle_rad)*depth).astype(int)
    
    # Create new image
    img = image.Image(*surf.shape)
    
    # Shift pixels of the original image (modulo size_y)
    img.array[patt.X, patt.Y] = patt.array[patt.X, (patt.Y+dy) % patt.size_y]
    return img

def shifted_Ycoords(Z: np.ndarray, Y: np.ndarray, depth: float = None,
                    angle_rad: float = None, angle_deg: float = 30) -> np.ndarray:
    """Create a meshgrid of shifted y-coordinates to simulate a projection on a surface"""
    # Set angle in radians
    angle_rad = angle_rad or angle_deg/180*np.pi
    
    # Set depth for z-axis
    depth = depth or Y.shape[1]/4
    
    # Calculate projection shift
    dy = (Z-1/2)*np.sin(angle_rad)*depth
    
    # Return a meshgrid with shifted y-coordinates (modulo size_y)
    Ycoords = (Y+dy) % Y.shape[1]
    return Ycoords

def max_depth(surf: surface.Surface, angle_rad: float = None, angle_deg: float = 30):
    """Calculate the maximum depth that can be used for projections without a shadow"""
    # Set angle in radians
    angle_rad = angle_rad or angle_deg/180*np.pi
    
    # Calculate the gradient
    grad_y, _ = np.gradient(surf.array)
    
    # Calculate the maximum depth by the gradient
    max_depth = np.tan(np.pi/2 - angle_rad)/(grad_y.max())
    return max_depth


def plot(surf: surface.Surface, proj: image.Image, z_max: int|float = 1, show: bool = True):
    """Show a projection on a matplotlib plot (laggy)"""
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(surf.X, surf.Y, surf.Z, facecolors=cv2.cvtColor(proj.array.transpose(),cv2.COLOR_GRAY2RGB), rstride=8)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.axis('equal')
    ax.set_zlim(0, z_max)
    if show:
        fig.show()
    return fig, ax