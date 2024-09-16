from typing import Any, NamedTuple
import hashing as hs


class ImageHashData(NamedTuple):
    image_path: Any
    phash: Any
    whash: Any
    pdq: Any
    sift: Any
    orb: Any


class ImageHashDistances(NamedTuple):
    phash_distance: float
    whash_distance: float
    pdq_distance: float
    sift_distance: float
    orb_distance: float


def compute_hashes(image_path: str):
    try:
        phash = hs.hash_image_phash(image_path)
    except Exception as e:
        phash = None

    try:
        whash = hs.hash_image_whash(image_path)
    except Exception as e:
        whash = None

    try:
        pdq = hs.hash_image_pdq(image_path)
    except Exception as e:
        pdq = None

    try:
        sift = hs.compute_sift_descriptors(image_path)
    except Exception as e:
        sift = None

    try:
        orb = hs.compute_orb_descriptors(image_path)
    except Exception as e:
        orb = None

    return ImageHashData(image_path, phash, whash, pdq, sift, orb)


def compute_distances(image_hash_data: ImageHashData, image_hash_data2: ImageHashData):
    if image_hash_data.phash is None or image_hash_data2.phash is None:
        phash_distance = None
    else:
        phash_distance = hs.phash_hamming_distance(
            image_hash_data.phash, image_hash_data2.phash
        )

    if image_hash_data.whash is None or image_hash_data2.whash is None:
        whash_distance = None
    else:
        whash_distance = hs.whash_hamming_distance(
            image_hash_data.whash, image_hash_data2.whash
        )

    if image_hash_data.pdq is None or image_hash_data2.pdq is None:
        pdq_distance = None
    else:
        pdq_distance = hs.pdq_hamming_distance(image_hash_data.pdq, image_hash_data2.pdq)

    if image_hash_data.sift is None or image_hash_data2.sift is None:
        sift_distance = None
    else:
        sift_distance = hs.sift_distance(image_hash_data.sift, image_hash_data2.sift)

    if image_hash_data.orb is None or image_hash_data2.orb is None:
        orb_distance = None
    else:
        orb_distance = hs.orb_distance(image_hash_data.orb, image_hash_data2.orb)

    return ImageHashDistances(
        phash_distance, whash_distance, pdq_distance, sift_distance, orb_distance
    )


if __name__ == "__main__":
    # Example usage
    image_hash_data = compute_hashes("dataset/PXL_20240901_044727631.jpg")
    image_hash_data2 = compute_hashes("dataset/PXL_20240907_064144426.jpg")
    image_hash_distances = compute_distances(image_hash_data, image_hash_data2)
    print(f"Image hashes: {image_hash_data}")
    print(f"Image hashes: {image_hash_data2}")
    print(f"Distances: {image_hash_distances}")