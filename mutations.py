import os
from typing import NamedTuple
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np


# Define the ImageData namedtuple
# ImageData = namedtuple('ImageData', ['original_image_path', 'mutated_image_paths'])
class ImageData(NamedTuple):
    original_image_path: str
    mutated_image_paths: dict[str, str]


def rotate_image(image, angle=5):
    """
    Rotates the image by a specified angle.
    """
    rotated_image = image.rotate(angle, expand=True)
    return rotated_image


def add_gaussian_noise(image, noise_level=5):
    """
    Adds Gaussian noise to the image.
    """
    image_np = np.array(image.convert("RGB"))
    mean = 0
    sigma = noise_level
    gauss = np.random.normal(mean, sigma, image_np.shape).astype("uint8")
    noisy_image_np = image_np + gauss
    noisy_image = Image.fromarray(np.clip(noisy_image_np, 0, 255).astype("uint8"))
    return noisy_image


def adjust_brightness(image, brightness_factor=1.5):
    """
    Adjusts the brightness of the image.
    """
    enhancer = ImageEnhance.Brightness(image)
    bright_image = enhancer.enhance(brightness_factor)
    return bright_image


def blur_image(image, percent=0.01):
    """
    Applies Gaussian blur to the image.
    """

    blur_size = (image.width + image.height) / 2
    blur_pixels = int(blur_size * percent)
    blurred_image = image.filter(ImageFilter.GaussianBlur(blur_pixels))
    return blurred_image


def jpeg_compress_image(image, quality=10):
    """
    Simulates JPEG compression artifacts.
    """
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    compressed_image = Image.open(buffer)
    return compressed_image


def crop_image(image, crop_percent=0.1):
    """
    Crops the image by a certain percentage from each side.
    """
    width, height = image.size
    crop_amount_w = int(crop_percent * width)
    crop_amount_h = int(crop_percent * height)
    cropped_image = image.crop(
        (crop_amount_w, crop_amount_h, width - crop_amount_w, height - crop_amount_h)
    )
    return cropped_image


def flip_image(image):
    """
    Flips the image horizontally.
    """
    flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return flipped_image


def change_color_balance(image, red_factor=1.5, green_factor=1.0, blue_factor=0.3):
    """
    Adjusts the color balance of the image.
    """
    r, g, b = image.convert("RGB").split()
    r = r.point(lambda i: i * red_factor)
    g = g.point(lambda i: i * green_factor)
    b = b.point(lambda i: i * blue_factor)
    color_adjusted_image = Image.merge("RGB", (r, g, b))
    return color_adjusted_image


def affine_transform_image(image):
    """
    Applies a small affine transformation to the image.
    """
    width, height = image.size
    # Define the affine transformation matrix
    m = (1, 0.1, -0.1 * width, 0.1, 1, -0.1 * height)
    transformed_image = image.transform((width, height), Image.AFFINE, m)
    return transformed_image

def passthrough_image(image):
    """
    Passes through the image without any mutations.
    """
    return image

def resize_image_to_max_dimension(image, max_dimension=512):
    """
    Resizes the image to a maximum dimension.
    """
    width, height = image.size
    if width > height:
        new_width = max_dimension
        new_height = int(height * (max_dimension / width))
    else:
        new_height = max_dimension
        new_width = int(width * (max_dimension / height))
    resized_image = image.resize((new_width, new_height))
    return resized_image

all_known_mutations = ['rotate', 'noise', 'brightness', 'blur', 'jpeg_compress', 'crop', 'flip', 'color_balance', 'affine_transform']

def load_and_mutate_image(filename):
    """
    Loads the original image from 'dataset/{filename}', performs mutations if mutated images don't exist,
    saves the mutated images to 'dataset_mutated/{filename}_{mutation}.png',
    and returns an ImageData namedtuple containing the paths.
    """
    # Paths
    original_image_path = os.path.join("dataset", filename)
    mutated_images_folder = "dataset_mutated"

    # Ensure 'dataset_mutated' directory exists
    os.makedirs(mutated_images_folder, exist_ok=True)

    # Prepare the output
    mutated_image_paths = {}

    # Get the base filename without extension
    base_filename, ext = os.path.splitext(filename)

    # Define the mutations
    mutations = {
        "original": passthrough_image,
        "rotate": rotate_image,
        "noise": add_gaussian_noise,
        "brightness": adjust_brightness,
        "blur": blur_image,
        "jpeg_compress": jpeg_compress_image,
        "crop": crop_image,
        "flip": flip_image,
        "color_balance": change_color_balance,
        "affine_transform": affine_transform_image,
    }

    try:
        original_image = resize_image_to_max_dimension(Image.open(original_image_path))
    except Exception as e:
        print(f"Error loading image {original_image_path}: {e}")
        return None

    # Strip the alpha channel if it exists
    if original_image.mode == "RGBA":
        original_image = original_image.convert("RGB")

    # For each mutation, check if the mutated image already exists
    for mutation_name, mutation_function in mutations.items():
        # Prepare the output path
        mutated_image_filename = f"{base_filename}_{mutation_name}.png"
        mutated_image_path = os.path.join(mutated_images_folder, mutated_image_filename)

        # Check if the mutated image already exists
        if os.path.exists(mutated_image_path):
            print(
                f"Mutated image already exists: {mutated_image_path}. Skipping mutation."
            )
        else:
            # Apply the mutation
            mutated_image = mutation_function(original_image)
            # Save the mutated image
            mutated_image.save(mutated_image_path)
            print(f"Mutated image saved: {mutated_image_path}")

        # Add to the dictionary
        mutated_image_paths[mutation_name] = mutated_image_path

    # Extract the original image path and remove it from the dict
    original_image_path = mutated_image_paths["original"]
    del mutated_image_paths["original"]

    # Return the ImageData namedtuple
    return ImageData(
        original_image_path=original_image_path, mutated_image_paths=mutated_image_paths
    )


if __name__ == "__main__":
    files = load_and_mutate_image("PXL_20240823_231254225.jpg")
    print(files)
