import os
import torch
import numpy as np
import scipy.interpolate

nimages = 2**14
angle = 30
sizex = 256
sizey = 256
data_dir = "data"
labels_name = f"general"
start_set = 4
end_set = 4

def generate_random_surface_model(sizex,sizey,nhills,interpol_method):
    if nhills == 0:
        ZI = np.random.uniform(0,1)*np.ones(sizex,sizey)
    elif nhills == 1: # straight flat tilted plane
        ZI = 1.1
        while max(max(ZI)) > 1 or min(min(ZI)) < 0:
            p1 = [0,0,np.random.uniform(0,1)]
            p2 = [sizex,0,np.random.uniform(0,1)]
            p3 = [sizex/2,sizey,np.random.uniform(0,1)]
            normal = np.cross(p1 - p2, p1 - p3)
            d = p1(1)*normal(1) + p1(2)*normal(2) + p1(3)*normal(3)
            d = -d
            x = [range(1,sizex)]
            y = [range(1,sizex)]
            [X,Y] = np.meshgrid(x,y)
            ZI = (-d - (normal(1)*X) - (normal(2)*Y))/normal(3)
    else:
        Z=np.random.normal(0,1,size=(nhills,nhills))
        points = np.linspace(0, nhills, nhills),np.linspace(0, nhills, nhills)
        [xq,yq] = np.meshgrid(np.linspace(1, nhills-1, sizex),np.linspace(1, nhills-1, sizey))
        XI = np.column_stack((xq.ravel(), yq.ravel()))
        if interpol_method == 'spline':
            ZI = scipy.interpolate.interpn(points, Z, XI, method='splinef2d', bounds_error=False, fill_value = 0)
        elif interpol_method == 'linear':
            ZI= scipy.interpolate.interp2d(x,y,Z,kind='linear')
        elif interpol_method == 'random':
            if np.random.uniform(0,1) > 0.5:
                ZI = scipy.interpolate.interpn(points, Z, XI, method='splinef2d', bounds_error=False, fill_value = 0)
            else:
                ZI=scipy.interpolate.interp2d(x,y,Z,kind='linear')
    ZI = ZI.reshape(xq.shape)
    minZI,maxZI  = ZI.min(), ZI.max()
    ZI = (ZI - minZI)/(maxZI - minZI)
    return ZI

for idx in range(start_set, end_set+1):

    labels = torch.zeros(nimages, sizex, sizey)

    for i in range(nimages):
        nhills = np.random.randint(4,15)
        ZI = generate_random_surface_model(sizex,sizey,nhills,'spline')

        # Add surface to labels tensor
        labels[i] = torch.from_numpy(ZI).unsqueeze(0)

        if i % (nimages//2**5) == 0:
                print(f"Generating data: {i}/{nimages} ({i/nimages:.2%})", end = "\r")

    print(f"Generating data: {nimages}/{nimages} (100.00%)")

    # Save torch tensor to file
    labels = labels.unsqueeze(1)  # Add feature dimension
    os.makedirs(data_dir, exist_ok=True)
    save_path = os.path.join(data_dir, f"{labels_name}_labels_{idx}.pth")
    torch.save(labels, save_path)