"""
Create sample medical image dataset for testing.
Generates synthetic medical images for demonstration purposes.
"""

import os
import numpy as np
from pathlib import Path
import cv2
from PIL import Image, ImageDraw, ImageFont
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_synthetic_xray(image_size=(224, 224), disease_type="normal"):
    """
    Create synthetic X-ray images for demonstration.
    
    Args:
        image_size: Size of the image (height, width)
        disease_type: Type of disease to simulate
        
    Returns:
        Synthetic X-ray image as numpy array
    """
    # Create base image (dark background)
    image = np.zeros((*image_size, 3), dtype=np.uint8)
    
    # Add some noise to simulate X-ray texture
    noise = np.random.normal(0, 20, image.shape).astype(np.uint8)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Add lung-like structures
    center_x, center_y = image_size[1] // 2, image_size[0] // 2
    
    # Left lung
    cv2.ellipse(image, (center_x - 40, center_y), (30, 60), 0, 0, 360, (100, 100, 100), -1)
    # Right lung
    cv2.ellipse(image, (center_x + 40, center_y), (30, 60), 0, 0, 360, (100, 100, 100), -1)
    
    # Add heart shadow
    cv2.ellipse(image, (center_x, center_y + 20), (25, 35), 0, 0, 360, (80, 80, 80), -1)
    
    # Add disease-specific features
    if disease_type == "pneumonia":
        # Add cloudy areas in lungs
        cv2.ellipse(image, (center_x - 30, center_y - 10), (15, 25), 0, 0, 360, (60, 60, 60), -1)
        cv2.ellipse(image, (center_x + 30, center_y - 10), (15, 25), 0, 0, 360, (60, 60, 60), -1)
    elif disease_type == "tuberculosis":
        # Add small nodules
        for _ in range(5):
            x = np.random.randint(center_x - 60, center_x + 60)
            y = np.random.randint(center_y - 40, center_y + 40)
            cv2.circle(image, (x, y), 3, (50, 50, 50), -1)
    elif disease_type == "lung_cancer":
        # Add large mass
        cv2.ellipse(image, (center_x - 20, center_y - 20), (20, 30), 0, 0, 360, (40, 40, 40), -1)
    
    # Convert to grayscale and back to RGB
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    return image

def create_synthetic_skin_lesion(image_size=(224, 224), disease_type="normal"):
    """
    Create synthetic skin lesion images for demonstration.
    
    Args:
        image_size: Size of the image (height, width)
        disease_type: Type of skin disease to simulate
        
    Returns:
        Synthetic skin lesion image as numpy array
    """
    # Create base skin color
    base_color = np.random.randint(200, 255, 3)
    image = np.full((*image_size, 3), base_color, dtype=np.uint8)
    
    # Add skin texture
    noise = np.random.normal(0, 15, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    center_x, center_y = image_size[1] // 2, image_size[0] // 2
    
    if disease_type == "melanoma":
        # Dark, irregular mole
        cv2.ellipse(image, (center_x, center_y), (25, 35), 0, 0, 360, (50, 50, 50), -1)
        # Add irregular borders
        for _ in range(10):
            x = center_x + np.random.randint(-30, 30)
            y = center_y + np.random.randint(-40, 40)
            cv2.circle(image, (x, y), 2, (30, 30, 30), -1)
    elif disease_type == "basal_cell_carcinoma":
        # Pearly bump
        cv2.ellipse(image, (center_x, center_y), (20, 20), 0, 0, 360, (180, 180, 180), -1)
        cv2.ellipse(image, (center_x, center_y), (15, 15), 0, 0, 360, (220, 220, 220), -1)
    elif disease_type == "squamous_cell_carcinoma":
        # Red, scaly lesion
        cv2.ellipse(image, (center_x, center_y), (30, 20), 0, 0, 360, (150, 100, 100), -1)
        # Add scaly texture
        for _ in range(20):
            x = center_x + np.random.randint(-25, 25)
            y = center_y + np.random.randint(-15, 15)
            cv2.circle(image, (x, y), 1, (120, 80, 80), -1)
    
    return image

def create_sample_dataset(output_dir: str = "data/raw/medical_images", 
                         num_samples_per_class: int = 50):
    """
    Create a sample medical image dataset.
    
    Args:
        output_dir: Directory to save the dataset
        num_samples_per_class: Number of samples per disease class
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define disease classes and their types
    disease_classes = {
        'pneumonia': 'xray',
        'tuberculosis': 'xray', 
        'lung_cancer': 'xray',
        'normal_lung': 'xray',
        'melanoma': 'skin',
        'basal_cell_carcinoma': 'skin',
        'squamous_cell_carcinoma': 'skin',
        'normal_skin': 'skin'
    }
    
    logger.info(f"Creating sample dataset with {num_samples_per_class} samples per class")
    
    for disease, image_type in disease_classes.items():
        # Create class directory
        class_dir = output_path / disease
        class_dir.mkdir(exist_ok=True)
        
        logger.info(f"Creating {num_samples_per_class} samples for {disease}")
        
        for i in range(num_samples_per_class):
            # Generate synthetic image
            if image_type == 'xray':
                image = create_synthetic_xray(disease_type=disease)
            elif image_type == 'skin':
                image = create_synthetic_skin_lesion(disease_type=disease)
            else:
                # Default to normal
                image = create_synthetic_xray(disease_type="normal")
            
            # Save image
            image_path = class_dir / f"{disease}_{i:03d}.jpg"
            cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    # Create CSV file with image paths and labels
    create_dataset_csv(output_path)
    
    logger.info(f"Sample dataset created in {output_path}")
    logger.info(f"Classes: {list(disease_classes.keys())}")
    logger.info(f"Total samples: {len(disease_classes) * num_samples_per_class}")

def create_dataset_csv(dataset_dir: Path):
    """Create CSV file with image paths and labels."""
    import pandas as pd
    
    data = []
    
    for class_dir in dataset_dir.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            
            for image_file in class_dir.glob("*.jpg"):
                data.append({
                    'image_path': str(image_file.relative_to(dataset_dir)),
                    'class_name': class_name,
                    'disease_type': 'xray' if 'lung' in class_name or class_name in ['pneumonia', 'tuberculosis', 'lung_cancer'] else 'skin'
                })
    
    df = pd.DataFrame(data)
    csv_path = dataset_dir / "dataset.csv"
    df.to_csv(csv_path, index=False)
    
    logger.info(f"Dataset CSV created: {csv_path}")
    logger.info(f"Dataset contains {len(df)} images")

def create_dicom_sample():
    """Create a sample DICOM file for testing."""
    try:
        import pydicom
        from pydicom.dataset import Dataset, FileDataset
        
        # Create a simple DICOM file
        filename = "data/raw/medical_images/sample.dcm"
        
        # Create the FileDataset
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.1'  # CR Image Storage
        file_meta.MediaStorageSOPInstanceUID = "1.2.3.4.5.6.7.8.9"
        file_meta.ImplementationClassUID = "1.2.3.4.5.6.7.8.9"
        
        ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
        
        # Add required DICOM tags
        ds.PatientName = "Sample^Patient"
        ds.PatientID = "12345"
        ds.StudyInstanceUID = "1.2.3.4.5.6.7.8.9.1"
        ds.SeriesInstanceUID = "1.2.3.4.5.6.7.8.9.2"
        ds.SOPInstanceUID = "1.2.3.4.5.6.7.8.9.3"
        
        # Create a simple image
        image = create_synthetic_xray((256, 256), "normal")
        ds.PixelData = image.tobytes()
        ds.Rows, ds.Columns = image.shape[:2]
        ds.PhotometricInterpretation = "RGB"
        ds.SamplesPerPixel = 3
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        
        # Save the DICOM file
        ds.save_as(filename)
        logger.info(f"Sample DICOM file created: {filename}")
        
    except ImportError:
        logger.warning("pydicom not available, skipping DICOM sample creation")

def main():
    """Main function to create sample dataset."""
    logger.info("Creating sample medical image dataset...")
    
    # Create sample dataset
    create_sample_dataset(
        output_dir="data/raw/medical_images",
        num_samples_per_class=30  # Smaller for demo
    )
    
    # Create DICOM sample
    create_dicom_sample()
    
    logger.info("Sample dataset creation completed!")

if __name__ == "__main__":
    main()
