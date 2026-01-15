"""
Edge Detection Performance Comparison
Compares Canny, Sobel, and Laplacian edge detectors on synthetic images
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List


class SyntheticImageGenerator:
    """Generate synthetic images with known edge locations"""

    def __init__(self, size: Tuple[int, int] = (512, 512)):
        self.size = size
        self.edge_pixels = set()  # Store exact edge locations

    def create_image(self, bg_intensity: int = 50,
                     square_intensity: int = 200,
                     circle_intensity: int = 150) -> np.ndarray:
        """
        Create synthetic image with one square and one circle

        Args:
            bg_intensity: Background pixel intensity (0-255)
            square_intensity: Square fill intensity (0-255)
            circle_intensity: Circle fill intensity (0-255)

        Returns:
            Synthetic image as numpy array
        """
        # Create blank image
        img = np.full(self.size, bg_intensity, dtype=np.uint8)
        self.edge_pixels.clear()

        # Add filled square (top-left quadrant)
        square_top_left = (100, 100)
        square_size = 150
        cv2.rectangle(img,
                     square_top_left,
                     (square_top_left[0] + square_size, square_top_left[1] + square_size),
                     square_intensity,
                     -1)  # -1 for filled

        # Record square edge pixels
        x1, y1 = square_top_left
        x2, y2 = x1 + square_size, y1 + square_size

        # Top and bottom edges
        for x in range(x1, x2 + 1):
            self.edge_pixels.add((y1, x))
            self.edge_pixels.add((y2, x))

        # Left and right edges
        for y in range(y1, y2 + 1):
            self.edge_pixels.add((y, x1))
            self.edge_pixels.add((y, x2))

        # Add filled circle (bottom-right quadrant)
        circle_center = (350, 350)
        circle_radius = 80
        cv2.circle(img, circle_center, circle_radius, circle_intensity, -1)

        # Record circle edge pixels (approximate using bresenham-like algorithm)
        cx, cy = circle_center
        for angle in np.linspace(0, 2*np.pi, 360*4):  # High resolution
            x = int(cx + circle_radius * np.cos(angle))
            y = int(cy + circle_radius * np.sin(angle))
            if 0 <= y < self.size[0] and 0 <= x < self.size[1]:
                self.edge_pixels.add((y, x))

        return img

    def get_ground_truth_edges(self) -> np.ndarray:
        """Create binary ground truth edge map"""
        gt = np.zeros(self.size, dtype=np.uint8)
        for y, x in self.edge_pixels:
            if 0 <= y < self.size[0] and 0 <= x < self.size[1]:
                gt[y, x] = 255
        return gt


class EdgeDetector:
    """Wrapper for different edge detection methods"""

    @staticmethod
    def canny(img: np.ndarray, low_threshold: int = 50,
              high_threshold: int = 150) -> np.ndarray:
        """Apply Canny edge detection"""
        return cv2.Canny(img, low_threshold, high_threshold)

    @staticmethod
    def sobel(img: np.ndarray, threshold: int = 50) -> np.ndarray:
        """Apply Sobel edge detection"""
        # Compute gradients in x and y directions
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

        # Compute magnitude
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # Normalize to 0-255
        magnitude = np.uint8(magnitude / magnitude.max() * 255)

        # Apply threshold
        _, edges = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)

        return edges

    @staticmethod
    def laplacian(img: np.ndarray, threshold: int = 30) -> np.ndarray:
        """Apply Laplacian edge detection"""
        # Apply Laplacian
        laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=3)

        # Take absolute value
        laplacian = np.abs(laplacian)

        # Normalize to 0-255
        laplacian = np.uint8(laplacian / laplacian.max() * 255)

        # Apply threshold
        _, edges = cv2.threshold(laplacian, threshold, 255, cv2.THRESH_BINARY)

        return edges


class PerformanceMetrics:
    """Calculate edge detection performance metrics"""

    @staticmethod
    def calculate_metrics(detected: np.ndarray, ground_truth: np.ndarray,
                         tolerance: int = 2) -> Dict[str, float]:
        """
        Calculate performance metrics with spatial tolerance

        Args:
            detected: Binary edge map from detector
            ground_truth: Binary ground truth edge map
            tolerance: Pixel distance tolerance for matching

        Returns:
            Dictionary with precision, recall, F1 score, and accuracy
        """
        # Convert to binary
        detected_bin = (detected > 0).astype(np.uint8)
        gt_bin = (ground_truth > 0).astype(np.uint8)

        # Create dilated ground truth for tolerance
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (2*tolerance+1, 2*tolerance+1))
        gt_dilated = cv2.dilate(gt_bin, kernel, iterations=1)
        detected_dilated = cv2.dilate(detected_bin, kernel, iterations=1)

        # True Positives: detected edges near ground truth
        tp = np.sum((detected_bin > 0) & (gt_dilated > 0))

        # False Positives: detected edges far from ground truth
        fp = np.sum((detected_bin > 0) & (gt_dilated == 0))

        # False Negatives: ground truth edges not detected
        fn = np.sum((gt_bin > 0) & (detected_dilated == 0))

        # True Negatives: non-edges correctly identified
        tn = np.sum((detected_bin == 0) & (gt_bin == 0))

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }


def add_gaussian_noise(img: np.ndarray, mean: float = 0,
                       sigma: float = 25) -> np.ndarray:
    """Add Gaussian noise to image"""
    noise = np.random.normal(mean, sigma, img.shape)
    noisy_img = img.astype(np.float32) + noise
    noisy_img = np.clip(noisy_img, 0, 255)
    return noisy_img.astype(np.uint8)


def visualize_results(img: np.ndarray, results: Dict[str, np.ndarray],
                     ground_truth: np.ndarray, metrics: Dict[str, Dict],
                     title_prefix: str = ""):
    """Visualize edge detection results and metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original image
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title(f'{title_prefix}Original Image')
    axes[0, 0].axis('off')

    # Ground truth
    axes[0, 1].imshow(ground_truth, cmap='gray')
    axes[0, 1].set_title('Ground Truth Edges')
    axes[0, 1].axis('off')

    # Placeholder for metrics summary
    axes[0, 2].axis('off')
    metrics_text = "Performance Metrics (F1 Score):\n\n"
    for method, metric in metrics.items():
        metrics_text += f"{method}: {metric['f1_score']:.3f}\n"
        metrics_text += f"  Precision: {metric['precision']:.3f}\n"
        metrics_text += f"  Recall: {metric['recall']:.3f}\n\n"
    axes[0, 2].text(0.1, 0.5, metrics_text, fontsize=10,
                    verticalalignment='center', family='monospace')
    axes[0, 2].set_title('Metrics Summary')

    # Edge detection results
    methods = ['Canny', 'Sobel', 'Laplacian']
    for idx, method in enumerate(methods):
        axes[1, idx].imshow(results[method], cmap='gray')
        f1 = metrics[method]['f1_score']
        axes[1, idx].set_title(f'{method} (F1: {f1:.3f})')
        axes[1, idx].axis('off')

    plt.tight_layout()
    return fig


def run_experiment(bg_intensity: int = 50, square_intensity: int = 200,
                   circle_intensity: int = 150, noise_sigma: float = 0,
                   canny_low: int = 50, canny_high: int = 150,
                   sobel_thresh: int = 50, laplacian_thresh: int = 30,
                   title_prefix: str = ""):
    """
    Run complete edge detection experiment

    Args:
        bg_intensity: Background intensity
        square_intensity: Square fill intensity
        circle_intensity: Circle fill intensity
        noise_sigma: Gaussian noise standard deviation (0 = no noise)
        canny_low: Canny low threshold
        canny_high: Canny high threshold
        sobel_thresh: Sobel threshold
        laplacian_thresh: Laplacian threshold
        title_prefix: Prefix for plot titles

    Returns:
        Tuple of (figure, metrics_dict, results_dict)
    """
    # Generate synthetic image
    generator = SyntheticImageGenerator()
    img = generator.create_image(bg_intensity, square_intensity, circle_intensity)
    ground_truth = generator.get_ground_truth_edges()

    # Add noise if specified
    if noise_sigma > 0:
        img = add_gaussian_noise(img, sigma=noise_sigma)

    # Apply edge detection methods
    detector = EdgeDetector()
    results = {
        'Canny': detector.canny(img, canny_low, canny_high),
        'Sobel': detector.sobel(img, sobel_thresh),
        'Laplacian': detector.laplacian(img, laplacian_thresh)
    }

    # Calculate metrics
    metric_calc = PerformanceMetrics()
    metrics = {}
    for method, edges in results.items():
        metrics[method] = metric_calc.calculate_metrics(edges, ground_truth)

    # Visualize
    fig = visualize_results(img, results, ground_truth, metrics, title_prefix)

    return fig, metrics, results, img, ground_truth


def main():
    """Run comprehensive edge detection experiments"""

    print("="*70)
    print("EDGE DETECTION PERFORMANCE COMPARISON EXPERIMENT")
    print("="*70)

    # Experiment 1: Clean image with default parameters
    print("\n[Experiment 1] Clean image with default parameters")
    print("-" * 70)
    fig1, metrics1, _, img1, gt1 = run_experiment(
        title_prefix="Exp 1: "
    )
    plt.savefig('/Users/eli/experiment_1_clean.png', dpi=150, bbox_inches='tight')

    for method, metric in metrics1.items():
        print(f"{method:12s} - Precision: {metric['precision']:.3f}, "
              f"Recall: {metric['recall']:.3f}, F1: {metric['f1_score']:.3f}")

    # Experiment 2: With Gaussian noise
    print("\n[Experiment 2] With Gaussian noise (sigma=25)")
    print("-" * 70)
    fig2, metrics2, _, img2, gt2 = run_experiment(
        noise_sigma=25,
        title_prefix="Exp 2: Noisy - "
    )
    plt.savefig('/Users/eli/experiment_2_noisy.png', dpi=150, bbox_inches='tight')

    for method, metric in metrics2.items():
        print(f"{method:12s} - Precision: {metric['precision']:.3f}, "
              f"Recall: {metric['recall']:.3f}, F1: {metric['f1_score']:.3f}")

    # Experiment 3: Different intensities (low contrast)
    print("\n[Experiment 3] Low contrast (smaller intensity differences)")
    print("-" * 70)
    fig3, metrics3, _, img3, gt3 = run_experiment(
        bg_intensity=100,
        square_intensity=150,
        circle_intensity=130,
        title_prefix="Exp 3: Low Contrast - "
    )
    plt.savefig('/Users/eli/experiment_3_low_contrast.png', dpi=150, bbox_inches='tight')

    for method, metric in metrics3.items():
        print(f"{method:12s} - Precision: {metric['precision']:.3f}, "
              f"Recall: {metric['recall']:.3f}, F1: {metric['f1_score']:.3f}")

    # Experiment 4: Different thresholds on clean image
    print("\n[Experiment 4] Clean image with adjusted thresholds")
    print("-" * 70)
    fig4, metrics4, _, img4, gt4 = run_experiment(
        canny_low=30,
        canny_high=100,
        sobel_thresh=30,
        laplacian_thresh=20,
        title_prefix="Exp 4: Lower Thresholds - "
    )
    plt.savefig('/Users/eli/experiment_4_low_thresholds.png', dpi=150, bbox_inches='tight')

    for method, metric in metrics4.items():
        print(f"{method:12s} - Precision: {metric['precision']:.3f}, "
              f"Recall: {metric['recall']:.3f}, F1: {metric['f1_score']:.3f}")

    # Experiment 5: Noisy image with adjusted thresholds
    print("\n[Experiment 5] Noisy image with higher thresholds")
    print("-" * 70)
    fig5, metrics5, _, img5, gt5 = run_experiment(
        noise_sigma=25,
        canny_low=70,
        canny_high=180,
        sobel_thresh=70,
        laplacian_thresh=50,
        title_prefix="Exp 5: Noisy + High Thresh - "
    )
    plt.savefig('/Users/eli/experiment_5_noisy_high_thresh.png', dpi=150, bbox_inches='tight')

    for method, metric in metrics5.items():
        print(f"{method:12s} - Precision: {metric['precision']:.3f}, "
              f"Recall: {metric['recall']:.3f}, F1: {metric['f1_score']:.3f}")

    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY: F1 Score Comparison Across All Experiments")
    print("="*70)

    experiments = {
        'Exp 1 (Clean, Default)': metrics1,
        'Exp 2 (Noisy)': metrics2,
        'Exp 3 (Low Contrast)': metrics3,
        'Exp 4 (Low Thresholds)': metrics4,
        'Exp 5 (Noisy + High Thresh)': metrics5
    }

    for exp_name, metrics in experiments.items():
        print(f"\n{exp_name}:")
        for method, metric in metrics.items():
            print(f"  {method:12s}: {metric['f1_score']:.3f}")

    # Create summary comparison plot
    fig_summary, ax = plt.subplots(figsize=(12, 6))

    methods = ['Canny', 'Sobel', 'Laplacian']
    x = np.arange(len(experiments))
    width = 0.25

    for idx, method in enumerate(methods):
        f1_scores = [experiments[exp][method]['f1_score'] for exp in experiments.keys()]
        ax.bar(x + idx*width, f1_scores, width, label=method)

    ax.set_xlabel('Experiment')
    ax.set_ylabel('F1 Score')
    ax.set_title('Edge Detection Performance Comparison Across Experiments')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"Exp {i+1}" for i in range(len(experiments))], rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig('/Users/eli/experiment_summary.png', dpi=150, bbox_inches='tight')

    print("\n" + "="*70)
    print("All experiments completed! Results saved as PNG files.")
    print("="*70)

    plt.show()


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    main()
