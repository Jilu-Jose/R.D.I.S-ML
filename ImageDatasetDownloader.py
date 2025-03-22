import os
import requests
import pandas as pd


csv_path = "resized_dataset.csv"  
df = pd.read_csv(csv_path)


image_folder = "images_100"
os.makedirs(image_folder, exist_ok=True)


for index, row in df.iterrows():
    img_url = row['img']
    img_path = os.path.join(image_folder, f"{row['id']}.jpg")
    
    try:
        response = requests.get(img_url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(img_path, "wb") as file:
                file.write(response.content)
            print(f"Downloaded: {img_path}")
        else:
            print(f"Failed to download {img_url}")
    except Exception as e:
        print(f"Error downloading {img_url}: {e}")

print("Image download completed.")