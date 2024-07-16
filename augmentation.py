import imgaug.augmenters as iaa
import numpy as np
from imgaug.augmenters import Affine


def applyAugmentation(images):

    # HYPER-PARAMETERS
    # rotation
    rotate_bounds = (0, 10)
    # cropping
    crop_bounds = (0, 0.15)
    # shearing
    tuple = (0.15, 6.0)
    # scale
    scalex_bound = (1.0, 1.3)
    scaley_bound = (1.0, 1.3)
    # blur
    sigma = (np.random.uniform(0.2, 2.5), np.random.uniform(0.2, 2.5))
    # contrast
    gamma = (0.75, 1.1)
    # noise
    scale_noise = (10, 30)
    # hue
    hue_range = (-30, 50)

    output = images

    # flip
    dice = np.random.randint(0, 10) / 10.0
    if dice < 0:
        output = horizontal_flip(output)
    dice = np.random.randint(0, 10) / 10.0
    if dice < 10:
        output = vertical_flip(output)



    # hue and saturation
    dice = np.random.randint(0, 10) / 10.0
    if dice > 0.5:
        output = hue_and_saturation(output, hue_range)


    return convert_back_to_uint(output)


def vertical_flip(image):

    flipped = np.zeros(image.shape)
    for index, img in enumerate(image):

        for index2 in [0, 1, 2]:
            array = img[:, :, index2]
            flipped[index, :, :, index2] = np.flip(array, 0)
    return convert_back_to_uint(flipped)


def horizontal_flip(image):


    flipped = np.zeros(image.shape)
    for index, img in enumerate(image):

        for index2 in [0, 1, 2]:
            array = img[:, :, index2]
            flipped[index, :, :, index2] = np.flip(array, 1)
    return convert_back_to_uint(flipped)


def shear(image, shear_bounds):

    shear_value = np.random.randint(shear_bounds[0], shear_bounds[1])
    seq = iaa.Sequential([iaa.Affine(shear=(shear_value, shear_value))])
    images_aug = seq(images=image)
    return convert_back_to_uint(images_aug)


def gaussian_blur(image, strength):

    res = image.copy()
    gauss = iaa.Sequential([iaa.GaussianBlur(sigma=strength)])
    images_to_modified = image[0:2]
    res[0:2] = convert_back_to_uint(gauss(images=images_to_modified))
    return res


def hue_and_saturation(image, hue_range):

    res = image.copy()
    modifier = iaa.AddToHueAndSaturation(hue_range, per_channel=True)
    res[0:2] = modifier(images=convert_back_to_uint(image[0:2]))
    return res


def rotate(images, bounds):

    rotate_value = np.random.randint(bounds[0], bounds[1])
    rotating = iaa.Affine(rotate=(rotate_value, rotate_value))
    rotated_image = rotating(images=images)
    return convert_back_to_uint(rotated_image)


def gaussian_noise(images, bounds):

    noisy_images = images.copy()
    gaussian_noise_adder = iaa.AdditiveGaussianNoise(loc=0, scale=bounds, per_channel=0.5)
    noisy_images[0:2] = gaussian_noise_adder(images=convert_back_to_uint(images[0:2]))
    return noisy_images


def crop(images, bounds):

    crop_value = np.random.uniform(bounds[0], bounds[1])
    croper = iaa.Crop(percent=(crop_value, crop_value))
    cropped_images = croper(images=images)
    return convert_back_to_uint(cropped_images)


def contrast(images, gamma):

    contrasted_images = images.copy()
    contraster = iaa.GammaContrast(gamma=gamma)
    contrasted_images[0:2] = contraster(images=images[0:2])
    return convert_back_to_uint(contrasted_images)


def scale(images, boundx, boundy):

    scalex = np.random.uniform(boundx[0], boundx[1])
    scaley = np.random.uniform(boundy[0], boundy[1])
    scaler = iaa.Affine(scale={"x": (scalex, scalex), "y": (scaley, scaley)})
    scaled_images = scaler(images=images)
    return convert_back_to_uint(scaled_images)


def convert_back_to_uint(matrix):

    matrix = matrix.astype(np.float64) / 255
    data = 255 * matrix
    img = data.astype(np.uint8)
    return img
