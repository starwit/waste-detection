import os
import torch
import shutil
import numpy as np
from PIL import Image, ImageOps
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def load_images(image_files, preprocess):
    """Load and preprocess images in batches."""
    images = []
    for image_file in image_files:
        image = Image.open(image_file).convert('RGB')
        image = preprocess(image).unsqueeze(0)  # Add batch dimension
        images.append(image)
    return torch.cat(images, dim=0)  # Concatenate all images into a single batch

def get_embeddings(images, model):
    """Extract embeddings for a batch of images using the model."""
    with torch.no_grad():
        embeddings = model(images).cpu().numpy()
    return embeddings

def save_combined_image(image1_path, image2_path, output_path):
    """Combine two images side by side and save them as a single image."""
    image1 = Image.open(image1_path).convert('RGB')
    image2 = Image.open(image2_path).convert('RGB')

    # Ensure both images are the same size
    image1 = ImageOps.fit(image1, (max(image1.size[0], image2.size[0]), max(image1.size[1], image2.size[1])))
    image2 = ImageOps.fit(image2, (max(image1.size[0], image2.size[0]), max(image1.size[1], image2.size[1])))

    # Combine images side by side
    combined_image = Image.new('RGB', (image1.width + image2.width, image1.height))
    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (image1.width, 0))

    combined_image.save(output_path)


def remove_duplicates(image_path, labels_path, similarity_threshold):
  # Load pre-trained model
  model = models.resnet50(pretrained=True)
  model.eval()

  # Define the transformation to match the model input
  preprocess = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  image_files = [f for f in image_path.glob('*.jpg')]
  retained_images = []
  embeddings = []
  duplicate_pairs = []

  # Process images in batches
  for i in tqdm(range(0, len(image_files), 16), desc="Processing batches"):
      batch_files = image_files[i:i + 16]
      batch_images = load_images(batch_files, preprocess)
      batch_embeddings = get_embeddings(batch_images, model)

      for j, embedding in enumerate(batch_embeddings):
          if len(embeddings) == 0:
              embeddings.append(embedding)
              retained_images.append(batch_files[j])
              continue

          similarities = cosine_similarity([embedding], embeddings)
          max_similarity = np.max(similarities)
          max_index = np.argmax(similarities)

          if max_similarity < similarity_threshold:
              embeddings.append(embedding)
              retained_images.append(batch_files[j])
          else:
              duplicate_pairs.append((batch_files[j], retained_images[max_index]))

  # Delete duplicate images and their corresponding label files
  for image_file, _ in duplicate_pairs:
      image_file.unlink()  # Delete the duplicate image
      label_file = labels_path / image_file.with_suffix('.txt').name
      if label_file.exists():
          label_file.unlink()  # Delete the corresponding label file

  # Print statistics
  print(f"Total images processed: {len(image_files)}")
  print(f"Unique images retained: {len(retained_images)}")
  print(f"Duplicate pairs found and deleted: {len(duplicate_pairs)}")