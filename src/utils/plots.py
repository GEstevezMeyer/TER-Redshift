import matplotlib.pyplot as plt
from src.utils.fits_management import give_image_data
from astropy.io import fits


def plot_fits_image(fits_object:fits,image_type = "IMAGE_HDU") -> None:

    image_data = give_image_data(fits_object,image_type)

    plt.show()
    plt.imshow(image_data,origin="lower")
    plt.colorbar()

    plt.show()


if __name__ == "__main__": 

    test = fits.open("data/arch-260925-104748/988-cutout-HSC-I-8279-pdr3_wide.fits")

    plot_fits_image(test)