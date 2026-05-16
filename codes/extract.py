import os
import glob
import numpy as np
import rasterio # THE FIX: Scientific image reader
import site
site_packages = site.getsitepackages()[0]
os.environ["GDAL_DATA"] = os.path.join(site_packages, "rasterio", "gdal_data")
RAW_DATA_FOLDER = "C:/Users/purbe/Downloads/archive/dataset/Sen1Floods11_8Channel/image/" 
OUTPUT_FOLDER = "c:/projects/multimodal 2.0/dataset/flood_dataset_npy/"
PATCH_SIZE = 64

def process_satellite_images():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    image_files = glob.glob(os.path.join(RAW_DATA_FOLDER, "*.tif"))
    print(f"Found {len(image_files)} raw satellite images.")
    
    if len(image_files) == 0:
        print("Error: No images found.")
        return

    patch_count = 0
    
    for img_path in image_files:
        try:
            # THE FIX: Open the file with rasterio
            with rasterio.open(img_path) as src:
                # Read the first band (Channel 1). Note: Rasterio bands are 1-indexed.
                img_array = src.read(1).astype(np.float32)
            
            # Prevent divide-by-zero errors if an image is completely black
            max_val = np.max(img_array)
            if max_val > 0:
                img_array = img_array / max_val 
            
            h, w = img_array.shape
            
            # Slice the image into 64x64 grids
            for y in range(0, h - PATCH_SIZE + 1, PATCH_SIZE):
                for x in range(0, w - PATCH_SIZE + 1, PATCH_SIZE):
                    patch = img_array[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                    
                    # Save the patch
                    filename = f"patch_{patch_count:05d}.npy"
                    save_path = os.path.join(OUTPUT_FOLDER, filename)
                    np.save(save_path, patch)
                    
                    patch_count += 1
                    
        except Exception as e:
            print(f"Skipping {os.path.basename(img_path)} - Error: {e}")

    print(f"\nSUCCESS! Created {patch_count} individual 64x64 .npy patches.")
    print(f"They are saved in: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    process_satellite_images()