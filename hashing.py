from PIL import Image
import imagehash
import cv2
import numpy as np
import pdqhash


def hash_image_phash(image_path: str):
    # Open the image file
    image = Image.open(image_path)

    # Compute the hash of the image
    img_hash = imagehash.phash(image)

    return img_hash


def phash_hamming_distance(hash1, hash2):
    return hash1 - hash2


def hash_image_whash(image_path: str):
    # Open the image file
    image = Image.open(image_path)

    # Compute the hash of the image
    img_hash = imagehash.whash(image)

    return img_hash


def whash_hamming_distance(hash1, hash2):
    return hash1 - hash2


def hash_image_pdq(image_path: str):
    # Open the image file
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Compute the PDQ hash of the image
    pdq_hash, quality = pdqhash.compute(image)

    return pdq_hash


def pdq_hamming_distance(hash1: np.ndarray, hash2: np.ndarray):
    # Check if both hashes have the same length
    if hash1.shape != hash2.shape:
        return "Hashes must have the same shape"

    # XOR the boolean arrays and count the number of differing bits
    differing_bits = np.sum(hash1 != hash2)

    return differing_bits


def compute_sift_descriptors(image_path: str):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise Exception("Image not found or unable to read.")

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute the descriptors
    keypoints, descriptors = sift.detectAndCompute(image, None)

    if descriptors is None:
        raise Exception("No descriptors found in the image: " + image_path)

    return (keypoints, descriptors)


def sift_distance(sift_data1, sift_data2):
    keypoints1, descriptors1 = sift_data1
    keypoints2, descriptors2 = sift_data2

    if descriptors1 is None or descriptors2 is None:
        return 0.0

    # Use BFMatcher with L2 norm
    bf = cv2.BFMatcher(cv2.NORM_L2)

    # Find the two nearest neighbors for each descriptor in descriptors1
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply Lowe's ratio test to find good matches
    good_matches = []
    for mn in matches:
        if len(mn) == 2:
            m, n = mn
        else:
            continue

        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Calculate the similarity score using the Dice coefficient
    total_keypoints = len(keypoints1) + len(keypoints2)
    if total_keypoints == 0:
        return 0.0

    similarity = (2 * len(good_matches)) / total_keypoints

    return 1 / similarity if similarity > 0 else 100


def compute_orb_descriptors(image_path: str):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise Exception("Image not found or unable to read.")

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and compute the descriptors
    keypoints, descriptors = orb.detectAndCompute(image, None)

    if descriptors is None:
        raise Exception("No descriptors found in the image: " + image_path)

    return (keypoints, descriptors)


def orb_distance(orb_data1, orb_data2):
    keypoints1, descriptors1 = orb_data1
    keypoints2, descriptors2 = orb_data2

    if descriptors1 is None or descriptors2 is None:
        return 0.0

    # Use BFMatcher with L2 norm
    bf = cv2.BFMatcher(cv2.NORM_L2)

    # Find the two nearest neighbors for each descriptor in descriptors1
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply Lowe's ratio test to find good matches
    good_matches = []
    for mn in matches:
        if len(mn) == 2:
            m, n = mn
        else:
            continue

        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Calculate the similarity score using the Dice coefficient
    total_keypoints = len(keypoints1) + len(keypoints2)
    if total_keypoints == 0:
        return 0.0

    similarity = (2 * len(good_matches)) / total_keypoints

    return 1 / similarity if similarity > 0 else 100


if __name__ == "__main__":
    # Example usage
    hash_value = compute_orb_descriptors("dataset/PXL_20240901_044727631.jpg")
    hash_value2 = compute_orb_descriptors("dataset/PXL_20240907_064144426.jpg")
    print(f"Hash value of the image: {orb_distance(hash_value, hash_value2)}")
