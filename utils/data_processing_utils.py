
#!pip install imagehash

import h5py
import csv
from pathlib import Path
#import imagehash
import itertools
import hashlib
import os
import shutil
import gdown
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files
from PIL import Image
import cv2
from google.colab.patches import cv2_imshow
import tensorflow as tf
from tensorflow import keras

def move_to_directory(folderpath="/content/data_folder"):
  os.makedirs(folderpath, exist_ok=True)
  uploaded = files.upload()
  for filename in uploaded.keys():
      shutil.move(filename, os.path.join(folderpath, filename))


def metadata_review(filename):
  print(filename, "\n", " ".join(["_" for i in range(20)]),"\n")
  df = pd.read_csv(filename)
  print(df.columns,"\n")
  print(df.describe(),"\n")
  print(df.info(), "\n")
  return df


def review_target_col(df, col):
  print(df[col].value_counts(), "\n")
  print( (100*df[col].value_counts(normalize=True) ).apply(lambda x: f'{x:.2f}%'), "\n")
  print(df[col].unique(), "\n")


def collect_filepathes(directory, files_to_find):
  """pick files to directory"""
  files=[]
  for file in os.listdir(directory):
    if file in files_to_find:
      filepath = os.path.join(directory, file)
      if os.path.isfile(filepath):
        files.append(filepath)
  return files


def is_c_hash_function(hash_function):
  """Checks if a function belongs to hashlib"""
  try:
      return callable(hash_function) and hasattr(hash_function(b''), 'digest') and hasattr(hash_function(b''), 'hexdigest')
  except Exception:
      return False

def is_p_hash_function(hash_function):
  """Checks if a function belongs to imagehash"""
  try:
    test_img = Image.new('RGB', (5, 5))
    return callable(hash_function) and isinstance(hash_function(test_img), imagehash.ImageHash)
  except Exception:
      return False


def perceptual_hash(filepath, hash_function):
  """Read file using PIL and compute perceptual hash hex string"""
  with Image.open(filepath) as img:
    if img is not None:
      return str(hash_function(img))
    else:
      print(f'{filepath}: Unable to read image')
      return '0'


def crypto_hash(filepath, hash_function):
  """Read file as byte-like object and compute cryptographic hash as hex string"""
  hash_obj = hash_function() #call to create hash object
  with open(filepath, 'rb') as img:
    if img is not None:
      for chunk in iter(lambda: img.read(4096), b""):  # read 4Kb
        hash_obj.update(chunk)
    else:
      print(f'{filepath}: Unable to read image')
      return '0'
  return hash_obj.hexdigest()


def compute_hash_table(directory, method="c", hash_function=hashlib.md5):
  """computes hash table with perceptual or cryptographic hashes"""

  if (method == "p") and (not is_p_hash_function(hash_function)):
    print("This hash function is not available for computing perceptual hash")
    return
  if (method == "c") and (not is_c_hash_function(hash_function)):
    print("This hash function is not available for computing cryptographic hash")
    return
  if method not in ["p", "c"]:
    print(f'{method}: "Method is incorrect"')
    return


def find_duplicates(hash_table):
  duplicates = hash_table[hash_table['diff']==0].copy()
  duplicate_pairs = []
  for i in duplicates.index:
    duplicate_pairs.append((hash_table.at[i-1, 'image'], hash_table.loc[i, 'image']))
  return duplicate_pairs


def demo_duplicates(directory, pairs_list):
  n_pairs = len(pairs_list)
  plt.figure(figsize=(10, np.ceil(n_pairs/2)*2.5))
  demo_lst = list(itertools.chain(*[collect_filepathes(directory, p) for p in pairs_list]))
  for i, im in enumerate(demo_lst):
    im_bgr = cv2.imread(im)
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    ax = plt.subplot(int(np.ceil(n_pairs/2)), 4, i + 1)
    plt.imshow(im_rgb)
    plt.axis("off")


def remove_files(files_to_find):
  for file in files_to_find:
    os.remove(file)


def duplicates_handling_pipeline(directory, method="c", hash_function=hashlib.md5, slice_len=6):
  """make hash table, collect duplicates pairs, define the number of pais,
     plot examples, collect filepathes and remove duplicates"""

  hash_table = compute_hash_table(pathlib.Path(directory), method="c", hash_function=hashlib.md5)
  dupl_pairs = find_duplicates(hash_table)
  print(f"Number of duplicates pairs: {len(dupl_pairs)}")
  print(f"{dupl_pairs[:slice_len]}")
  demo_duplicates(directory, dupl_pairs[:slice_len])

  filename_to_save = "/content/" + directory.replace("/", "_") + "_dupl.csv"
  df = pd.DataFrame(dupl_pairs, columns=['image_name_1','image_name_2'])
  df.to_csv(filename_to_save, index=False)
  # with open(filename_to_save, mode="w", newline='') as file:
  #   writer = csv.writer(file)
  #   for row in dupl_pairs:
  #     writer.writerow(row)
  return dupl_pairs


def remove_duplicates(directory, dupl_pairs, slice_len=6):
  if len(dupl_pairs)>0:
    names_remove_arr = np.array(dupl_pairs)[:,1]
    d_files = collect_filepathes(directory, names_remove_arr)
    print(f"Examples to remove: {d_files[:5]}")
    remove_files(d_files)
    counter=0
    for file in d_files:
      if not os.path.exists(file):
        counter +=1
    print(f"Removed {counter} files")
  else:
    print(f"No duplicates in {directory}")


def find_target_class_fnames(metadata_path, row):
  """get list of files with target class"""

  df = pd.read_csv(f'{metadata_path}/{row["filename"]}')
  df[row["image_col"]]=df[row["image_col"]].apply(lambda x: x+".jpg")
  targ_col_mask = df[row["target_col_name"]].isin([1, "melanoma"])
  target_im = df[targ_col_mask][row["image_col"]]
  return target_im.tolist()


def separate_dataset_classes(source_dir, dest_dir, files_to_find):
  """copy target class files to separate directory"""

  classes = ["melanoma", "other"]
  for cl in classes:
    sub_dir = f"{dest_dir}/{cl}"
    if not os.path.exists(sub_dir):
      os.makedirs(sub_dir)
  for filename in os.listdir(source_dir):
    # filepath = os.path.join(source_dir, file)
    if filename.endswith(('.jpg', '.jpeg', ".png") ):
      if filename in files_to_find:
        shutil.copyfile(f"{source_dir}/{filename}", f"{dest_dir}/{classes[0]}/{filename}")
      else:
        shutil.copyfile(f"{source_dir}/{filename}", f"{dest_dir}/{classes[1]}/{filename}")


def has_black_vignette(image, center_ratio=0.5, threshold=70):
  """define if image has black background"""

  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  height, width = gray.shape
  mask = np.zeros((height, width), dtype=np.uint8)
  cv2.circle(mask, (width//2, height//2), int(min(height, width)*center_ratio), 255, -1)
  outside_mask = cv2.bitwise_not(mask)
  avg_brightness = cv2.mean(gray, mask=outside_mask)[0]
  return avg_brightness < threshold


def separate_vignette_images(directory, center_ratio=0.5, threshold=70):
  has = []
  not_has = []
  for im in os.listdir(directory):
    image_path = directory / im
    image = cv2.imread(image_path)
    if has_black_vignette(image, center_ratio, threshold):
      has.append(image_path)
    else:
      not_has.append(image_path)
  return {"has": has, "not_has": not_has}


def demo_images(images_pathes_list):
  num_images = len(images_pathes_list)
  plt.figure(figsize=(18, np.ceil(num_images/7)*2.5))
  for i, im in enumerate(images_pathes_list):
    im_bgr = cv2.imread(im)
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    ax = plt.subplot(int(np.ceil(num_images/7)), 7, i + 1)
    plt.imshow(im_rgb)
    plt.axis("off")


def add_vignette(image_path, output_path, blur_kernel_size=31, intensity=0.9, window_radius_ratio=1):
    """
    Додати вин'єтку (чорну рамку) навколо зображення.
    :param intensity: інтенсивність затемнення.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image is not loaded")
    h, w = image.shape[:2]
    y, x = np.mgrid[0:h, 0:w]
    cx, cy = w // 2, h // 2
    # Радіальна градієнтна маска
    window_radius = int(min(cx, cy)*window_radius_ratio)
    distance = np.sqrt((x - cx)**2 + (y - cy)**2)
    mask = np.ones((h, w), dtype=np.float32)
    mask[distance > window_radius] = 1 - intensity
    mask = cv2.GaussianBlur(mask, (blur_kernel_size, blur_kernel_size), 0)
    # mask = np.stack([mask] * 3, axis=-1).astype(np.float32)
    mask = cv2.merge([mask, mask, mask]).astype(np.float32)  # Для 3-х каналів

    vignette_image = (image * mask).astype(np.uint8)
    # масив NumPy, який містить зображення в форматі RGB
    vignette_image = cv2.cvtColor(vignette_image, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(output_path, vignette_image) # зберігає в BGR
    pil_vignette_image = Image.fromarray(vignette_image) #перетворити у PIL
    pil_vignette_image.save(output_path) # зберігає в RGB
    return vignette_image

# !!!повтор
def split_dataset(source_dir, dest_dir, classes, files_to_find):
  """Split dataset in two derictories"""

  for cl in classes:
    sub_dir = f"{dest_dir}/{cl}"
    if not os.path.exists(sub_dir):
      os.makedirs(sub_dir)
  for filename in os.listdir(source_dir):
    if filename.endswith(('.jpg', '.jpeg', ".png") ):
      if filename in files_to_find:
        shutil.copyfile(f"{source_dir}/{filename}", f"{dest_dir}/{classes[0]}/{filename}")
      else:
        shutil.copyfile(f"{source_dir}/{filename}", f"{dest_dir}/{classes[1]}/{filename}")



def show_dirs_len(path):
  for root, dirs, _ in os.walk(path):
    print(f"{os.path.relpath(root, path)}: {len(os.listdir(root))}")


def crop_black_frame(image_np, threshold=20):
  """
  Обрізати чорні края по периметру, ресайз до вихідного розміру.
      image: RGB NumPy image (H, W, 3)
      threshold: Порог яскравості, нижче якого вважається "чорним"
  """

  h, w = image_np.shape[:2]
  original_size = (w, h)
  radius = min(h, w) // 2 - 1 #зменшення на 1

  gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

  mask = gray > threshold

  coords = np.argwhere(mask)

  if coords.shape[0] == 0:
      return image_np

  y1, x1 = coords.min(axis=0)
  y2, x2 = coords.max(axis=0)

  margin_y = int(0.05 * (y2 - y1))
  margin_x = int(0.05 * (x2 - x1))

  y1 = max(0, y1 + margin_y)
  y2 = min(h, y2 - margin_y)
  x1 = max(0, x1 + margin_x)
  x2 = min(w, x2 - margin_x)

  image_tf = tf.convert_to_tensor(image_np, dtype=tf.uint8)
  # image_tf = tf.image.decode_jpeg(image, channels=3)
  cropped_tensor = tf.image.crop_to_bounding_box(image_tf, y1, x1, y2-y1, x2-x1)

  resized = tf.image.resize(cropped_tensor, original_size, method=tf.image.ResizeMethod.BILINEAR)
  resized = tf.cast(resized, tf.uint8)

  # Save
  # cv2.imwrite("filled_image.jpg", filled_image)

  # encoded_image = tf.image.encode_jpeg(resized)
  # tf.io.write_file("processed_image.jpg", encoded_image)

  return resized

def crop_and_save(im_path, threshold=20):
  img = cv2.imread(im_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  processed = crop_black_frame(img, threshold)
  encoded_image = tf.image.encode_jpeg(processed)
  tf.io.write_file(im_path, encoded_image)


def resize_save_image(filename, source_dir, dest_dir, size=(256, 256)):

  height, width = size

  filepath = os.path.join(source_dir, filename)
  image = cv2.imread(filepath)
  if image is None:
    print(f"Couldn't load '{filepath}'.")
  else:
    resized_image = cv2.resize(image, (width, height))
  # resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(dest_dir, filename), resized_image)


def prepocess_to_crop(data_path, center_ratio=0.5, threshold=70):
  for root, dirs, files in os.walk(data_path):
    if files:
      data_dir = Path(root)
      to_crop_dict = separate_vignette_images(data_dir, center_ratio=center_ratio, threshold=threshold)
      print(f"With vignette {data_dir}: {len(to_crop_dict['has'])}")
      print(f"Without vignette {data_dir}: {len(to_crop_dict['not_has'])}")
      print(len(os.listdir(data_dir)))
      for im_path in to_crop_dict['has']:
        im_path = str(im_path)
        crop_and_save(im_path)
