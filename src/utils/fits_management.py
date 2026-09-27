from astropy.io import fits
import numpy as np 

REGISTERY_IMAGE_TYPE = {
    "IMAGE_HDU":1

}

def give_image_data(fits_file:fits,image_type = "IMAGE_HDU") -> list[list[float]]: 
    return np.array(fits_file[REGISTERY_IMAGE_TYPE[image_type]].data)

