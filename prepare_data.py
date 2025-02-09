import os
import json
from PIL import Image
from transformers import BertTokenizer
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class VLM_Dataset(Dataset):
    def __init__(self, data_path, annotations_path, max_samples=None):
        self.data_path = data_path
        
        # Verify data directory exists
        if not os.path.exists(data_path):
            raise RuntimeError(f"Data directory not found: {data_path}")
            
        # Load and process COCO annotations
        if not os.path.exists(annotations_path):
            raise RuntimeError(f"Annotations file not found: {annotations_path}")
            
        with open(annotations_path, 'r') as f:
            coco = json.load(f)
            
        # Create image_id to filename mapping
        self.image_to_file = {
            img['id']: img['file_name'] 
            for img in coco['images']
        }
        
        # Store annotations with proper image filenames
        self.annotations = []
        for ann in coco['annotations']:
            img_filename = self.image_to_file[ann['image_id']]
            img_path = os.path.join(data_path, img_filename)
            
            # Only add annotations for images that exist
            if os.path.exists(img_path):
                self.annotations.append({
                    'image': img_filename,
                    'caption': ann['caption']
                })
                
        if len(self.annotations) == 0:
            raise RuntimeError(
                f"No valid images found in {data_path}. "
                "Please download the COCO 2017 training images from "
                "https://cocodataset.org/#download"
            )
            
        # Optionally limit dataset size
        if max_samples is not None:
            self.annotations = self.annotations[:max_samples]
            
        print(f"Loaded dataset with {len(self.annotations)} valid images")
            
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_filename = self.annotations[idx]['image']
        img_path = os.path.join(self.data_path, img_filename)
        
        # Load and process image
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            raise
            
        # Process text
        text = self.annotations[idx]['caption']
        text_tokenized = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=50,
            truncation=True
        )
        
        # Remove the batch dimension that tokenizer adds
        text_tokenized = {k: v.squeeze(0) for k, v in text_tokenized.items()}
        
        return image, text_tokenized