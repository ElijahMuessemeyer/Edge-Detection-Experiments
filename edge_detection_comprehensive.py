import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List

class EdgeDetectionExperiment:
    """
    Comprehensive edge detection experiment comparing Canny, Sobel, and Laplacian methods.
    """

    def __init__(self, image_size: Tuple[int, int] = (512, 512)):
        self.image_size = image_size
        self.ground_truth_edges = None

    def create_synthetic_image(self,
                              square_intensity: int = 200,
                              circle_intensity: int = 150,
                              background_intensity: int = 50,
                              noise_level: float = 0.0) -> Tuple[np.ndarray, Dict]:
        """
        Create synthetic image with one filled square and one filled circle.

        Args:
            square_intensity: Grayscale intensity of square (0-255)
            circle_intensity: Grayscale intensity of circle (0-255)
            background_intensity: Background intensity (0-255)
            noise_level: Standard deviation of Gaussian noise to add

        Returns:
            Tuple of (image, shape_info) where shape_info contains edge coordinates
        """
        # Create background
        image = np.full(self.image_size, background_intensity, dtype=np.uint8)

        # Define square parameters (top-left corner)
        square_x, square_y = 100, 100
        square_size = 150

        # Define circle parameters (center)
        circle_x, circle_y = 350, 350
        circle_radius = 80

        # Draw filled square
        cv2.rectangle(image,
                     (square_x, square_y),
                     (square_x + square_size, square_y + square_size),
                     int(square_intensity),
                     -1)  # -1 means filled

        # Draw filled circle
        cv2.circle(image,
                  (circle_x, circle_y),
                  circle_radius,
                  int(circle_intensity),
                  -1)  # -1 means filled

        # Store shape information for ground truth
        shape_info = {
            'square': {
                'x': square_x,
                'y': square_y,
                'size': square_size,
                'intensity': square_intensity
            },
            'circle': {
                'x': circle_x,
                'y': circle_y,
                'radius': circle_radius,
                'intensity': circle_intensity
            },
            'background': background_intensity
        }

        # Add noise if specified
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, image.shape)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)

        return image, shape_info

    def create_ground_truth(self, shape_info: Dict) -> np.ndarray:
        """
        Create ground truth edge map from known shape coordinates.
        """
        ground_truth = np.zeros(self.image_size, dtype=np.uint8)

        # Square edges (4 lines)
        sq = shape_info['square']
        x, y, size = sq['x'], sq['y'], sq['size']

        # Draw square perimeter
        cv2.rectangle(ground_truth, (x, y), (x + size, y + size), 255, 1)

        # Circle edge
        circ = shape_info['circle']
        cv2.circle(ground_truth, (circ['x'], circ['y']), circ['radius'], 255, 1)

        self.ground_truth_edges = ground_truth
        return ground_truth

    def apply_canny(self, image: np.ndarray,
                   threshold1: int = 50,
                   threshold2: int = 150) -> np.ndarray:
        """Apply Canny edge detection."""
        return cv2.Canny(image, threshold1, threshold2)

    def apply_sobel(self, image: np.ndarray,
                   threshold: int = 50) -> np.ndarray:
        """Apply Sobel edge detection."""
        # Calculate gradients in x and y directions
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

        # Calculate gradient magnitude
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # Normalize to 0-255
        magnitude = np.uint8(255 * magnitude / np.max(magnitude))

        # Apply threshold
        _, edges = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)

        return edges

    def apply_laplacian(self, image: np.ndarray,
                       threshold: int = 30) -> np.ndarray:
        """Apply Laplacian edge detection."""
        # Apply Laplacian
        laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=3)

        # Take absolute value and normalize
        laplacian_abs = np.absolute(laplacian)
        laplacian_norm = np.uint8(255 * laplacian_abs / np.max(laplacian_abs))

        # Apply threshold
        _, edges = cv2.threshold(laplacian_norm, threshold, 255, cv2.THRESH_BINARY)

        return edges

    def evaluate_performance(self, detected_edges: np.ndarray) -> Dict[str, float]:
        """
        Evaluate edge detection performance against ground truth.

        Metrics:
        - Precision: TP / (TP + FP) - how many detected edges are true edges
        - Recall: TP / (TP + FN) - how many true edges were detected
        - F1-Score: Harmonic mean of precision and recall
        """
        if self.ground_truth_edges is None:
            raise ValueError("Ground truth not set. Call create_ground_truth first.")

        # Dilate ground truth slightly to account for pixel-level variations
        kernel = np.ones((3, 3), np.uint8)
        gt_dilated = cv2.dilate(self.ground_truth_edges, kernel, iterations=1)

        # Convert to binary
        gt_binary = (gt_dilated > 0).astype(bool)
        detected_binary = (detected_edges > 0).astype(bool)

        # Calculate metrics
        true_positives = np.sum(gt_binary & detected_binary)
        false_positives = np.sum(~gt_binary & detected_binary)
        false_negatives = np.sum(gt_binary & ~detected_binary)

        # Avoid division by zero
        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives)
        }

    def run_single_experiment(self,
                            image: np.ndarray,
                            canny_params: Tuple[int, int],
                            sobel_threshold: int,
                            laplacian_threshold: int) -> Dict:
        """
        Run all three edge detection methods on a single image.
        """
        # Apply edge detectors
        canny_edges = self.apply_canny(image, canny_params[0], canny_params[1])
        sobel_edges = self.apply_sobel(image, sobel_threshold)
        laplacian_edges = self.apply_laplacian(image, laplacian_threshold)

        # Evaluate performance
        canny_metrics = self.evaluate_performance(canny_edges)
        sobel_metrics = self.evaluate_performance(sobel_edges)
        laplacian_metrics = self.evaluate_performance(laplacian_edges)

        return {
            'canny': {'edges': canny_edges, 'metrics': canny_metrics},
            'sobel': {'edges': sobel_edges, 'metrics': sobel_metrics},
            'laplacian': {'edges': laplacian_edges, 'metrics': laplacian_metrics}
        }


def run_comprehensive_experiments():
    """
    Run all experiments and create comprehensive visualization.
    """
    experiment = EdgeDetectionExperiment()

    # Experiment configurations
    experiments = [
        {
            'name': 'Exp 1: Clean Image',
            'square_int': 200,
            'circle_int': 150,
            'bg_int': 50,
            'noise': 0.0,
            'canny_params': (50, 150),
            'sobel_thresh': 50,
            'laplacian_thresh': 30
        },
        {
            'name': 'Exp 2: Noisy Image',
            'square_int': 200,
            'circle_int': 150,
            'bg_int': 50,
            'noise': 15.0,
            'canny_params': (50, 150),
            'sobel_thresh': 50,
            'laplacian_thresh': 30
        },
        {
            'name': 'Exp 3: Low Contrast',
            'square_int': 120,
            'circle_int': 100,
            'bg_int': 80,
            'noise': 0.0,
            'canny_params': (50, 150),
            'sobel_thresh': 50,
            'laplacian_thresh': 30
        },
        {
            'name': 'Exp 4: Low Thresholds',
            'square_int': 200,
            'circle_int': 150,
            'bg_int': 50,
            'noise': 0.0,
            'canny_params': (20, 60),
            'sobel_thresh': 20,
            'laplacian_thresh': 10
        },
        {
            'name': 'Exp 5: Noisy + High Thresh',
            'square_int': 200,
            'circle_int': 150,
            'bg_int': 50,
            'noise': 20.0,
            'canny_params': (80, 200),
            'sobel_thresh': 80,
            'laplacian_thresh': 50
        }
    ]

    # Create figure with subplots
    n_experiments = len(experiments)
    fig = plt.figure(figsize=(20, 4 * n_experiments))

    all_results = []

    for exp_idx, exp_config in enumerate(experiments):
        # Create synthetic image
        image, shape_info = experiment.create_synthetic_image(
            square_intensity=exp_config['square_int'],
            circle_intensity=exp_config['circle_int'],
            background_intensity=exp_config['bg_int'],
            noise_level=exp_config['noise']
        )

        # Create ground truth (only need to do this once, but repeated for clarity)
        ground_truth = experiment.create_ground_truth(shape_info)

        # Run experiment
        results = experiment.run_single_experiment(
            image,
            exp_config['canny_params'],
            exp_config['sobel_thresh'],
            exp_config['laplacian_thresh']
        )

        # Store results
        all_results.append({
            'config': exp_config,
            'results': results
        })

        # Plot results (6 columns: original, ground truth, canny, sobel, laplacian, metrics)
        row = exp_idx

        # Original image
        ax1 = plt.subplot(n_experiments, 6, row * 6 + 1)
        ax1.imshow(image, cmap='gray', vmin=0, vmax=255)
        ax1.set_title(f"{exp_config['name']}\nOriginal")
        ax1.axis('off')

        # Ground truth
        ax2 = plt.subplot(n_experiments, 6, row * 6 + 2)
        ax2.imshow(ground_truth, cmap='gray')
        ax2.set_title('Ground Truth')
        ax2.axis('off')

        # Canny
        ax3 = plt.subplot(n_experiments, 6, row * 6 + 3)
        ax3.imshow(results['canny']['edges'], cmap='gray')
        canny_f1 = results['canny']['metrics']['f1_score']
        ax3.set_title(f"Canny\nF1: {canny_f1:.3f}")
        ax3.axis('off')

        # Sobel
        ax4 = plt.subplot(n_experiments, 6, row * 6 + 4)
        ax4.imshow(results['sobel']['edges'], cmap='gray')
        sobel_f1 = results['sobel']['metrics']['f1_score']
        ax4.set_title(f"Sobel\nF1: {sobel_f1:.3f}")
        ax4.axis('off')

        # Laplacian
        ax5 = plt.subplot(n_experiments, 6, row * 6 + 5)
        ax5.imshow(results['laplacian']['edges'], cmap='gray')
        laplacian_f1 = results['laplacian']['metrics']['f1_score']
        ax5.set_title(f"Laplacian\nF1: {laplacian_f1:.3f}")
        ax5.axis('off')

        # Metrics comparison
        ax6 = plt.subplot(n_experiments, 6, row * 6 + 6)
        methods = ['Canny', 'Sobel', 'Laplacian']
        f1_scores = [
            results['canny']['metrics']['f1_score'],
            results['sobel']['metrics']['f1_score'],
            results['laplacian']['metrics']['f1_score']
        ]
        precision_scores = [
            results['canny']['metrics']['precision'],
            results['sobel']['metrics']['precision'],
            results['laplacian']['metrics']['precision']
        ]
        recall_scores = [
            results['canny']['metrics']['recall'],
            results['sobel']['metrics']['recall'],
            results['laplacian']['metrics']['recall']
        ]

        x = np.arange(len(methods))
        width = 0.25

        ax6.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
        ax6.bar(x, recall_scores, width, label='Recall', alpha=0.8)
        ax6.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)

        ax6.set_ylabel('Score')
        ax6.set_title('Performance Metrics')
        ax6.set_xticks(x)
        ax6.set_xticklabels(methods, rotation=45, ha='right')
        ax6.legend(fontsize=8)
        ax6.set_ylim([0, 1])
        ax6.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/eli/edge_detection_comprehensive_results.png', dpi=150, bbox_inches='tight')
    print("Results saved to: edge_detection_comprehensive_results.png")

    # Print detailed metrics
    print("\n" + "="*80)
    print("DETAILED PERFORMANCE METRICS")
    print("="*80)

    for exp_idx, result in enumerate(all_results):
        config = result['config']
        results = result['results']

        print(f"\n{config['name']}")
        print(f"Parameters: Square={config['square_int']}, Circle={config['circle_int']}, "
              f"BG={config['bg_int']}, Noise={config['noise']}")
        print(f"Thresholds: Canny={config['canny_params']}, Sobel={config['sobel_thresh']}, "
              f"Laplacian={config['laplacian_thresh']}")
        print("-" * 80)

        for method in ['canny', 'sobel', 'laplacian']:
            metrics = results[method]['metrics']
            print(f"{method.upper():10} | "
                  f"Precision: {metrics['precision']:.4f} | "
                  f"Recall: {metrics['recall']:.4f} | "
                  f"F1-Score: {metrics['f1_score']:.4f}")

    print("\n" + "="*80)
    print("\nKEY FINDINGS:")
    print("- F1-Score balances precision (accuracy of detected edges) and recall (completeness)")
    print("- Higher F1-Score indicates better overall edge detection performance")
    print("- Noise and low contrast reduce performance across all methods")
    print("- Threshold values significantly impact the precision-recall tradeoff")
    print("="*80)

    return all_results


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    results = run_comprehensive_experiments()
