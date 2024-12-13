import json
import os

class CocoLoader:
    def __init__(self, coco_json_file):
        # Load the COCO annotations JSON file
        with open(coco_json_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Preprocess the data for easy lookup
        self.image_info = {image['file_name']: image for image in self.coco_data['images']}
        self.annotations = {ann['image_id']: [] for ann in self.coco_data['annotations']}
        
        for ann in self.coco_data['annotations']:
            self.annotations[ann['image_id']].append(ann)
        
        # Create a reverse mapping for image_id -> filename
        self.image_id_to_filename = {image['id']: image['file_name'] for image in self.coco_data['images']}
    
    def get_annotations_by_filename(self, filename):
        """
        Retrieve all annotations (bounding boxes, category ids, etc.) for a given image filename.
        Args:
            filename (str): The image filename (e.g., '000000000001.jpg').
        Returns:
            List of annotations associated with the image filename.
        """
        if filename not in self.image_info:
            raise ValueError(f"Image with filename {filename} not found in the COCO dataset.")
        
        # Get the image_id corresponding to the filename
        image_id = self.image_info[filename]['id']
        
        # Return all annotations for the image_id
        return self.annotations.get(image_id, [])

    def get_image_info(self, filename):
        """
        Retrieve basic information about an image from the filename.
        Args:
            filename (str): The image filename (e.g., '000000000001.jpg').
        Returns:
            dict: Image metadata (e.g., width, height, image_id).
        """
        if filename not in self.image_info:
            raise ValueError(f"Image with filename {filename} not found in the COCO dataset.")
        
        return self.image_info[filename]
    
    def get_num_annotations(self, filename):
        """
        Retrieve the number of annotations for a given image filename.
        Args:
            filename (str): The image filename (e.g., '000000000001.jpg').
        Returns:
            int: Number of annotations for the image.
        """
        if filename not in self.image_info:
            raise ValueError(f"Image with filename {filename} not found in the COCO dataset.")
        
        # Get the image_id corresponding to the filename
        image_id = self.image_info[filename]['id']
        
        return len(self.annotations.get(image_id, []))

# Example usage
if __name__ == "__main__":
    coco_loader = CocoLoader('../data/coco-stuff-2017/instances_val2017.json')  # Path to your COCO annotation file
    
    image_filename = '000000369675.jpg'  # Example image filename
    
    try:
        annotations = coco_loader.get_annotations_by_filename(image_filename)
        print(f"Annotations for {image_filename}: {annotations}")
        
        image_info = coco_loader.get_image_info(image_filename)
        print(f"Image info for {image_filename}: {image_info}")

        num_annotations = coco_loader.get_num_annotations(image_filename)
        print(f"Number of annotations for {image_filename}: {num_annotations}")

    
    except ValueError as e:
        print(e)
