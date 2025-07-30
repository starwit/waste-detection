from skimage.metrics import structural_similarity as ssim
from collections import defaultdict
from PIL import Image
import imagehash
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Set


class DisjointSet:
    """
    DisjointSet (Union-Find) data structure for efficiently grouping elements into disjoint sets.
    Used in the image duplicate detection to cluster similar images together.

    Key Operations:
    - find(): Determines which set an element belongs to
    - union(): Merges two sets together

      # Example: We have images: A, B, C
      # And we found these are similar:
      # A similar to B
      # B similar to C

      # Initially:
      # A → A
      # B → B
      # C → C

      ds.union('A', 'B')  # Now B points to A
      # A → As
      # B → A
      # C → C

      ds.union('B', 'C')  # Now C points to A (through B)
      # A → A
      # B → A
      # C → A

      # Now if we call find() on any image, it returns the root:
      ds.find('C')  # Returns 'A'

      # This creates cluster:
      # Cluster 1: A, B, C (with A as root)
    """

    def __init__(self):
        self.parent = {}

    def find(self, item):
        """
        Find the root/representative element of the set containing 'item'.
        """
        # If item isn't in any set yet, create a new set with item as its own parent
        if item not in self.parent:
            self.parent[item] = item

        # If item isn't its own parent, recursively find the root and compress the path by making all nodes point to root
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])

        return self.parent[item]

    def union(self, item1, item2):
        """
        Merge the sets containing item1 and item2.
        After union, both items will be in the same set.
        """
        # Find the roots of both items
        root1 = self.find(item1)
        root2 = self.find(item2)

        # If items are in different sets (have different roots) merge them by making root2 point to root1
        if root1 != root2:
            self.parent[root2] = root1


class DuplicateDetector:
    def __init__(self, phash_threshold: int = 2, ssim_threshold: float = 0.95):
        self.phash_threshold = phash_threshold
        self.ssim_threshold = ssim_threshold

    def compute_perceptual_hash(self, file_path: Path) -> str:
        try:
            with Image.open(file_path) as img:
                return str(imagehash.phash(img))
        except Exception:
            return None

    def compute_ssim(self, image1_path: Path, image2_path: Path) -> float:
        try:
            img1 = cv2.imread(str(image1_path), cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(str(image2_path), cv2.IMREAD_GRAYSCALE)

            img1 = cv2.resize(img1, (256, 256))
            img2 = cv2.resize(img2, (256, 256))

            score, _ = ssim(img1, img2, full=True)
            return score
        except Exception:
            return 0

    def find_duplicates(
        self, image_paths: List[Path]
    ) -> Tuple[Dict[Path, List[Path]], Dict[Tuple[Path, Path], Tuple[int, float]]]:
        phash_dict = []
        disjoint_set = DisjointSet()
        # Compute perceptual hashes
        for path in image_paths:
            phash = self.compute_perceptual_hash(path)
            if phash:
                phash_dict.append((phash, path))

        # Compare images and cluster similar ones
        for i, (hash1, path1) in enumerate(phash_dict):
            for j, (hash2, path2) in enumerate(phash_dict):
                if i < j:
                    hamming_distance = imagehash.hex_to_hash(
                        hash1
                    ) - imagehash.hex_to_hash(hash2)
                    if hamming_distance <= self.phash_threshold:
                        ssim_score = self.compute_ssim(path1, path2)
                        if ssim_score >= self.ssim_threshold:
                            disjoint_set.union(path1, path2)

        # Group images into clusters
        clusters = defaultdict(list)
        for path in {path for _, path in phash_dict}:
            root = disjoint_set.find(path)
            clusters[root].append(path)

        # Filter out non-duplicate clusters
        clusters = {k: v for k, v in clusters.items() if len(v) > 1}

        return clusters

    def print_duplicate_clusters(self, clusters: Dict[Path, List[Path]]):
        print(f"\nFound {len(clusters)} duplicate clusters:")
        for i, (cluster_root, cluster_images) in enumerate(clusters.items(), start=1):
            print(f"\nCluster {i}:")
            for img in cluster_images:
                print(f"  - {img}")

    def get_unique_images(self, image_paths: List[Path]) -> Set[Path]:
        clusters = self.find_duplicates(image_paths)

        # Keep only one image from each cluster (the first one)
        duplicates = set()
        for cluster_images in clusters.values():
            keep = min(cluster_images)            # lexicographically first
            duplicates.update(set(clusters) - {keep})

        # Return set of unique images
        return set(image_paths) - duplicates

    def compare_folders(
        self, folder1: Path, folder2: Path
    ) -> Dict[Path, List[Tuple[Path, float, int]]]:
        """
        Compare images between two folder structures and find similar images.
        """
        # Get all image paths from both folders
        folder1_images = (
            list(folder1.rglob("*.[jJ][pP][gG]"))
            + list(folder1.rglob("*.[pP][nN][gG]"))
            + list(folder1.rglob("*.[jJ][pP][eE][gG]"))
        )
        folder2_images = (
            list(folder2.rglob("*.[jJ][pP][gG]"))
            + list(folder2.rglob("*.[pP][nN][gG]"))
            + list(folder2.rglob("*.[jJ][pP][eE][gG]"))
        )

        # Compute hashes for all images
        folder1_hashes = [
            (self.compute_perceptual_hash(path), path) for path in folder1_images
        ]
        folder2_hashes = [
            (self.compute_perceptual_hash(path), path) for path in folder2_images
        ]

        # Remove None entries (failed hash computations)
        folder1_hashes = [(h, p) for h, p in folder1_hashes if h is not None]
        folder2_hashes = [(h, p) for h, p in folder2_hashes if h is not None]

        # Dictionary to store matches
        matches = defaultdict(list)

        # Compare images between folders
        total_comparisons = len(folder1_hashes) * len(folder2_hashes)
        completed_comparisons = 0

        for hash1, path1 in folder1_hashes:
            for hash2, path2 in folder2_hashes:
                completed_comparisons += 1
                if completed_comparisons % 1000 == 0:
                    progress = (completed_comparisons / total_comparisons) * 100
                hamming_distance = imagehash.hex_to_hash(hash1) - imagehash.hex_to_hash(
                    hash2
                )
                if hamming_distance <= self.phash_threshold:
                    ssim_score = self.compute_ssim(path1, path2)
                    if ssim_score >= self.ssim_threshold:
                        matches[path1].append((path2, ssim_score, hamming_distance))

        return matches

    def print_folder_comparison_results(
        self, matches: Dict[Path, List[Tuple[Path, float, int]]]
    ):
        """
        Print the results of folder comparison in a readable format.
        """
        if not matches:
            print("\nNo similar images found between the folders.")
            return

        print(f"\nFound similar images between folders:")
        for i, (source_img, similar_images) in enumerate(matches.items(), start=1):
            print(f"\nImage {i} from source folder:")
            print(f"  Source: {source_img}")
            print("  Similar images in target folder:")
            for target_img, ssim_score, hamming_dist in similar_images:
                print(f"    - {target_img}")
                print(f"      SSIM: {ssim_score:.3f}, Hamming distance: {hamming_dist}")
