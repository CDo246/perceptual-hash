from itertools import combinations
import os
from typing import NamedTuple, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import numpy as np

from cache import cache_function, parallel_map_with_cache
from mutations import load_and_mutate_image, ImageData, all_known_mutations
from hash_dataset_compute import (
    ImageHashData,
    ImageHashDistances,
    compute_hashes,
    compute_distances,
)
from plotting import HistogramData, PlotData, plot_bar_charts, plot_histograms


def get_all_image_filenames(dataset_folder="dataset") -> List[str]:
    """
    Returns all image filenames in the dataset folder.
    """
    filenames = []
    for filename in os.listdir(dataset_folder):
        if os.path.isfile(
            os.path.join(dataset_folder, filename)
        ) and filename.lower().endswith((".jpg", ".jpeg", ".png")):
            filenames.append(filename)
    return filenames


class ImageHashResult(NamedTuple):
    original_image_path: str
    # original_image_hash: ImageHashData
    # mutated_image_hashes: Dict[str, ImageHashData]
    mutated_image_hash_distances: Dict[str, ImageHashDistances]


def process_loaded_image(image_data: ImageData) -> ImageHashResult:
    """
    Processes the loaded image data to compute hashes and distances.
    """
    # Compute hash for the original image
    original_image_hash = compute_hashes(image_data.original_image_path)

    mutated_image_hashes = {}
    mutated_image_hash_distances = {}

    # Compute hashes and distances for mutated images
    for mutation_name, mutated_image_path in image_data.mutated_image_paths.items():
        mutated_hash = compute_hashes(mutated_image_path)
        mutated_image_hashes[mutation_name] = mutated_hash

        distances = compute_distances(original_image_hash, mutated_hash)
        mutated_image_hash_distances[mutation_name] = distances

    # Return the result as an ImageHashResult
    return ImageHashResult(
        original_image_path=image_data.original_image_path,
        # original_image_hash=original_image_hash,
        # mutated_image_hashes=mutated_image_hashes,
        mutated_image_hash_distances=mutated_image_hash_distances,
    )


def process_distance_betwen_two_images(image_pair: List[str]) -> ImageHashResult:
    """
    Load both images, get both their hashes, and return the distance

    The weird array argument is for easier parallelizing
    """
    image_hash_data1 = compute_hashes(image_pair[0])
    image_hash_data2 = compute_hashes(image_pair[1])
    distances = compute_distances(image_hash_data1, image_hash_data2)
    return distances


def load_and_prepare_image_data(filename: str) -> ImageData:
    image_data = load_and_mutate_image(filename)
    hashes = process_loaded_image(image_data)
    return hashes


load_and_prepare_image_data_cached = cache_function(load_and_prepare_image_data)


def get_distances_for_hash_results(
    results: List[ImageHashResult], algorithm: str, mutation_kind: Optional[str] = None
) -> List[float]:
    """
    Extracts the distances for a specific algorithm and optionally for a specific mutation kind.

    Parameters:
    - results: List of ImageHashResult objects.
    - algorithm: The algorithm name (e.g., 'phash', 'whash', 'pdq', 'sift', 'orb').
    - mutation_kind: Optionally filter by a specific mutation (e.g., 'rotate', 'blur'). If None, distances for all mutations are returned.

    Returns:
    - A list of floats representing the distances for the specified algorithm and mutation.
    """
    # Validate the algorithm name
    valid_algorithms = {"phash", "whash", "pdq", "sift", "orb"}
    if algorithm not in valid_algorithms:
        raise ValueError(
            f"Invalid algorithm '{algorithm}'. Valid options are: {valid_algorithms}"
        )

    # Initialize an empty list to collect the distances
    distances = []

    # Loop over all ImageHashResult objects
    for result in results:
        # If mutation_kind is specified, collect distances only for that mutation
        if mutation_kind:
            if mutation_kind in result.mutated_image_hash_distances:
                distance_value = getattr(
                    result.mutated_image_hash_distances[mutation_kind],
                    f"{algorithm}_distance",
                )
                distances.append(distance_value)
        # If mutation_kind is not specified, collect distances for all mutations
        else:
            for (
                mutation_name,
                distance_obj,
            ) in result.mutated_image_hash_distances.items():
                distance_value = getattr(distance_obj, f"{algorithm}_distance")
                distances.append(distance_value)

    return distances


def get_distances_for_hash_distance_arrays(
    results: List[ImageHashDistances], algorithm: str
) -> List[float]:
    # Validate the algorithm name
    valid_algorithms = {"phash", "whash", "pdq", "sift", "orb"}
    if algorithm not in valid_algorithms:
        raise ValueError(
            f"Invalid algorithm '{algorithm}'. Valid options are: {valid_algorithms}"
        )

    # Initialize an empty list to collect the distances
    distances = []

    # Loop over all ImageHashResult objects
    for result in results:
        distance_value = getattr(result, f"{algorithm}_distance")
        distances.append(distance_value)

    return distances


def filter_none_distances(distances):
    return [distance for distance in distances if distance is not None]


def count_none_distances(distances):
    return sum([1 for distance in distances if distance is None])


def create_unique_pairs(items: List[str]) -> List[Tuple[str, str]]:
    # Use itertools.combinations to generate all unique pairs
    pairs = list(combinations(items, 2))
    return pairs


if __name__ == "__main__":
    # Step 1: Get all image filenames
    filenames = get_all_image_filenames("dataset")

    # Step 2: Load and mutate images in parallel
    image_hash_results = parallel_map_with_cache(filenames, load_and_prepare_image_data)

    algorithms = ["phash", "whash", "pdq", "sift", "orb"]

    all_original_image_paths = [
        result.original_image_path for result in image_hash_results
    ]
    permutated = create_unique_pairs(all_original_image_paths)

    # Make a seeded random re-ordering of `permutated`
    np.random.seed(0)
    permutated = np.random.permutation(permutated)
    permutated = [(str(left), str(right)) for left, right in permutated]
    permutated = permutated[:10000]

    all_image_pairs_results = parallel_map_with_cache(
        permutated, process_distance_betwen_two_images
    )

    data = []
    data_relative_to_stdev = []
    bars = []

    bar_data = {}
    for algorithm in algorithms:
        mutation_distances = get_distances_for_hash_results(
            image_hash_results, algorithm
        )
        mutation_distances = filter_none_distances(mutation_distances)

        dataset_distances = get_distances_for_hash_distance_arrays(
            all_image_pairs_results, algorithm
        )
        dataset_distances = filter_none_distances(dataset_distances)
        dataset_distances_stdev = np.mean(dataset_distances)

        normalized_mutation_distances = np.divide(
            mutation_distances, dataset_distances_stdev
        )

        data.append(
            HistogramData(
                data={
                    "Mutated": mutation_distances,
                    "Dataset": dataset_distances,
                },
                name=algorithm,
            )
        )

        data_relative_to_stdev.append(
            HistogramData(
                data={
                    "Mutated": normalized_mutation_distances,
                },
                name=algorithm,
            )
        )

        bar_data[algorithm] = 1 / np.mean(normalized_mutation_distances)
    bars.append(PlotData(data=bar_data, name="All Mutations"))

    for mutation in all_known_mutations:
        bar_data = {}
        for algorithm in algorithms:
            mutation_distances = get_distances_for_hash_results(
                image_hash_results, algorithm, mutation
            )
            mutation_distances = filter_none_distances(mutation_distances)

            dataset_distances = get_distances_for_hash_distance_arrays(
                all_image_pairs_results, algorithm
            )
            dataset_distances = filter_none_distances(dataset_distances)
            dataset_distances_stdev = np.mean(dataset_distances)

            normalized_mutation_distances = np.divide(
                mutation_distances, dataset_distances_stdev
            )

            data.append(
                HistogramData(
                    data={
                        f"{mutation} Mutation": mutation_distances,
                        "Dataset": dataset_distances,
                    },
                    name=f"{algorithm} {mutation}",
                )
            )

            data_relative_to_stdev.append(
                HistogramData(
                    data={
                        f"{mutation} Mutation": normalized_mutation_distances,
                    },
                    name=f"{algorithm} {mutation}",
                )
            )

            bar_data[algorithm] = 1 / np.mean(normalized_mutation_distances)
        bars.append(PlotData(data=bar_data, name=mutation))

    plot_histograms(data, filename="distances.png")
    plot_histograms(
        data_relative_to_stdev, filename="distances_relative_to_stdev.png", x_max=1.5
    )
    plot_bar_charts(bars, filename="bar_charts.png")
