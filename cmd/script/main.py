from asyncio import sleep
from math import floor
from icrawler.builtin import GoogleImageCrawler
import os
from PIL import Image, ImageEnhance, ImageOps
import random 
import cv2
from pathlib import Path
import uuid
import shutil

# SOURCE_DIR = "dataset"
# TARGET_DIR = "food_classifier/processed"
SEED:int = 42
SPLIT_RATIOS = {"train":0.7, "val":0.15, "test":0.15}
SOURCE_DIR = "food_classifier/processed"
TARGET_DIR = "food_classifier/split"

DATA_DIR= "food_classifier/split"

meat_mapper : dict = {
    "beef_meat": "beef meat",
    "chicken_meat": "chicken meat",
    "fish_meat": "fish meat",
    "mutton_meat": "mutton meat",
    "pork_meat": "pork meat",
    "egg": "egg"
}

# crawls from google images...
def download_images(keyword:str, max_num:int =110, save_dir:str = "dataset/fallback") -> None:
    crawler = GoogleImageCrawler(storage={'root_dir':f'{save_dir}/{keyword}'})
    if (keyword in meat_mapper) :
        keyword = meat_mapper[keyword]

    print(f"Downloading images for {keyword}...")
    crawler.crawl(keyword=keyword, max_num=max_num)

def download_all_ingredients(ingredients:list, save_dir:str = "dataset/fallback") -> None :
    for ingredient in ingredients:
        download_images(ingredient, save_dir=save_dir)
        augment_folder(f"{save_dir}/{ingredient}", f"{save_dir}/{ingredient}/augmented", augment_count=3)



# data augmentation function
'''
- random horizontal flip
- random rotation
- random brightness
- random zoom in / zoom out
- random jitter.
'''
def augment_image(image) :

    # Random horizontal flip
    if (random.random() > 0.5) :
        image = ImageOps.mirror(image)
    
    # Random rotation
    angle = random.randint(-20, 30) # between -20 and 20 degrees
    image = image.rotate(angle)

    # Random brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(0.7, 1.3))

    # Random color jitter
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(random.uniform(0.7, 1.3))


    # random zoom
    if random.random() > 0.5 :
        width, height = image.size
        crop_percent = random.uniform(0.8, 1.0)
        crop_width , crop_height = int(width * crop_percent), int(height * crop_percent)
        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)
        image = image.crop((left, top, left + crop_width, top + crop_height))
        image = image.resize((width, height))
    
    return image


def augment_folder(input_folder:str, output_folder:str, augment_count:int = 3) :
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("Augmenting images...")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")    

    for root, _, files in os.walk(input_folder) :
        for file in files :
            if file.endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(root, file)
                image = Image.open(image_path)

                try: 
                    print(f"Processing image.....")
                    if image.mode== "P" :
                        print("Image mode is P, converting to RGBA")
                        image = image.convert('RGBA')
                    elif image.mode != "RGB" :
                        print("Image mode is not RGB, converting to RGB")
                        image = image.convert('RGB')

                    base_name = os.path.splitext(file)[0]
                    extension = os.path.splitext(file)[1]

                    for i in range(augment_count):
                        aug_image = augment_image(image)

                        print(f"Saving augmented image {i+1}...")
                        aug_image.save(os.path.join(
                            output_folder,
                            f"{base_name}_aug_{i}{extension}"
                        ))
                except Exception as e:
                    print(f"#################Error processing image: {image_path}#################")
                    print(e)


# resizing images to fixed size of 224 * 224 
def resize_image(input_path : str, output_path: str, size=(224, 224)) :
        image = cv2.imread(input_path)

        if image is None : 
            raise FileNotFoundError(f"Image not found at {input_path}")
        
        print("resizing image...")
        resized = cv2.resize(image, size)
        cv2.imwrite(output_path,resized)

def resize_image_in_place(input_root:str, size=(224, 224)) :
    base:str = input_root
    input_root = Path(input_root)
    print("Something is happening")

    for category_folder in input_root.iterdir():

        if (category_folder.is_dir()):
            #category_folder --> eggs, fish_meat, chicken_meat
            print(f"Processing folder: {category_folder.name}")

            # PRocessing images in "datasets/others/eggs*/*"

            for image_file in category_folder.glob("*"):
                if image_file.is_file() and image_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    try :
                        image = cv2.imread(str(image_file))
                        if image is None:
                            print(f"[SKIPPED] not an image : {image_file}")
                            continue
                        
                        resized = cv2.resize(image, size)
                        cv2.imwrite(str(image_file), resized)
                        print(f"[✓] Resized: {image_file}")

                    except Exception as e:
                        print(f"[ERROR] {image_file} : {e}")
                        continue
            
            # resizing images "/augmented" folder
            augmented_folder = category_folder/"augmented"

            if augmented_folder.is_dir():

                for aug_file in augmented_folder.glob("*"):
                    if aug_file.is_file() and aug_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        try :
                            image = cv2.imread(str(aug_file))
                            if image is None:
                                print(f"[SKIPPED] not an image : {aug_file}")
                                continue
                            
                            resized = cv2.resize(image, size)
                            cv2.imwrite(str(aug_file), resized)
                            print(f"[✓] Resized: {aug_file}")
                            sleep(0.3)

                        except Exception as e :
                            print(f"[ERROR] {aug_file} : {e}")
                            

# import all the data to processed
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def copy_images(src_folder:str, dest_folder:str) -> None:
    for file in os.listdir(src_folder):
        if file.lower().endswith((".jpg",".jpeg", ".png")):
            unique_name = f"{uuid.uuid4().hex}_{file}"
            shutil.copy(os.path.join(src_folder, file), os.path.join(dest_folder, unique_name))

# load raw+processed data to processed folder
def prepare_dataset():
    for category in os.listdir(SOURCE_DIR): # vegetables, fruits, others
        category_path = os.path.join(SOURCE_DIR, category)
        if not os.path.isdir(category_path):
            continue

        for class_name in os.listdir(category_path):# orage,apple, banana
            class_path = os.path.join(category_path, class_name)
            if not os.path.isdir(class_path):
                continue

            if class_name not in ['cucumber', 'pumpkin', 'peas']:
                continue

            print(f"Processing classes : {class_name}")
            dst_class_path = os.path.join(TARGET_DIR, class_name)
            ensure_dir(dst_class_path)

            copy_images(class_path, dst_class_path)

            aug_path = os.path.join(class_path, "augmented")
            if os.path.exists(aug_path) and os.path.isdir(aug_path):
                copy_images(aug_path, dst_class_path)
    
    print(f"\n✅ Dataset preparation complete. Images copied to: '{TARGET_DIR}'")

def split_and_copy():
    random.seed(SEED)

    for class_name in os.listdir(SOURCE_DIR) :
        class_dir = os.path.join(SOURCE_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue

        # get all files
        image_files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        random.shuffle(image_files)

        total:int = len(image_files)
        train_end = floor(SPLIT_RATIOS["train"] * total) 
        val_end = train_end + floor(SPLIT_RATIOS["val"] * total)

        splits = {
            "train" : image_files[:train_end],
            "val" : image_files[train_end:val_end],
            "test" : image_files[val_end:]
        }


        for split, files in splits.items() :
            split_dir = os.path.join(TARGET_DIR, split, class_name)
            ensure_dir(split_dir)

            for file in files:
                src = os.path.join(class_dir, file)
                dst = os.path.join(split_dir, file)
                shutil.copy(src, dst)

        print(f"✅ {class_name}: {total} images split and copied.")
    print("\n✅ Dataset successfully split into train, val, and test folders.")



if __name__ == "__main__" :

    # ingredients : list = ['cucumber', 'pumpkin', 'peas']
    # download_all_ingredients(ingredients, save_dir="dataset/vegetables")
    # resize_image_in_place("dataset/vegetables", size=(224, 224))
    # prepare_dataset()
    # resize_image("dataset/others/beef_meat/000003.png","dataset/fallback/resized_img.png")    
    # split_and_copy()
    print("fasdf")