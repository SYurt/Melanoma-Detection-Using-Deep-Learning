
import os
import gdown
import shutil
import numpy as np
import pandas as pd
import pathlib
import csv
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import tensorflow as tf
# import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model
from tensorflow.keras.utils import save_img, img_to_array, array_to_img, load_img, image_dataset_from_directory
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from google.colab import files

def make_sructured_subset(rate, source_path, dest_path, my_seed, copy=True):

  """create folder the same structure as source
     with part of content files"""
  np.random.seed(my_seed)
  for root, dirs, files in os.walk(source_path):
    dirs.sort()
    files.sort()
    relative_path = os.path.relpath(root, source_path)
    new_folder = os.path.join(dest_path, relative_path)
    os.makedirs(new_folder, exist_ok=True)
    if files:
      num_files = int(len(files)*rate)
      files_to_copy = np.random.choice(files, num_files, replace=False)

      for filename in files_to_copy:
        src_file = os.path.join(root, filename)
        dest_file = os.path.join(new_folder, filename)
        if copy:
          shutil.copyfile(src_file, dest_file)
        else:
          shutil.move(src_file, dest_file)


def make_balanced_sructured_subset(source_path, dest_path, my_seed, rates_per_class, copy=True):

  """create folder the same structure as source
     with part of content files, picked using different rate for selected folder:
      rate*k"""
  np.random.seed(my_seed)
  for root, dirs, files in os.walk(source_path):
    dirs.sort()
    files.sort()
    relative_path = os.path.relpath(root, source_path)
    class_name = os.path.basename(relative_path)
    if class_name in rates_per_class:
       rate = rates_per_class[class_name]
    else:
      rate = 0.2
    new_folder = os.path.join(dest_path, relative_path)
    os.makedirs(new_folder, exist_ok=True)
    if files:
      num_files = int(len(files)*rate)
      files_to_copy = np.random.choice(files, num_files, replace=False)

      for filename in files_to_copy:
        src_file = os.path.join(root, filename)
        dest_file = os.path.join(new_folder, filename)
        if copy:
          shutil.copyfile(src_file, dest_file)
        else:
          shutil.move(src_file, dest_file)


def make_custom_sructured_subset(source_path, dest_path, my_seed, rates_per_ds, copy=True):

  """create folder the same structure as source
     with % of source folder files, randomly picked
     using different rate for each ds"""
  np.random.seed(my_seed)

  for root, dirs, files in os.walk(source_path):
    dirs.sort()
    relative_path = os.path.relpath(root, source_path)
    new_folder = os.path.join(dest_path, relative_path)
    os.makedirs(new_folder, exist_ok=True)
    files = sorted(files)
    if files:
      for ds in rates_per_ds.keys():
        if ds in new_folder.split(os.sep):
          rate = rates_per_ds[ds]
          break
      num_files = int(len(files)*rate)
      files_to_copy = np.random.choice(files, num_files, replace=False)

      for filename in files_to_copy:
        src_file = os.path.join(root, filename)
        dest_file = os.path.join(new_folder, filename)
        if copy:
          shutil.copyfile(src_file, dest_file)
        else:
          shutil.move(src_file, dest_file)


def augment_and_save(img_path, data_augmentation, num_aug, save=True, output_dir=None, add_func=None):
  aug_images_set =[]
  img = load_img(img_path) #завантажує зображення як PIL
  img_arr = img_to_array(img) #конвертує у NumPy
  img_arr = np.expand_dims(img_arr, axis=0)  # додати розмірність - batch
  for i in range(num_aug):
    augmented_img_arr = data_augmentation(img_arr, training=True)
    # augmented_img = keras.utils.array_to_img(augmented_img_arr[0])
    augmented_img = array_to_img(augmented_img_arr[0])   #first in batch
    if add_func is not None:
      augmented_img = add_func(augmented_img)
    aug_images_set.append(augmented_img)
    if save:
      os.makedirs(output_dir, exist_ok=True)
      img_name = os.path.basename(img_path)
      output_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_aug_{i}.jpg")
      save_img(output_path, augmented_img)
  return aug_images_set

def run_augmentation(data_augmentation, input_dir, output_dir, num_aug, add_func=None):
  for img_name in os.listdir(input_dir):
    if img_name.endswith((".png", ".jpg", ".jpeg")):
      img_path = os.path.join(input_dir, img_name)
      augment_and_save(img_path, data_augmentation, num_aug, save=True, output_dir=output_dir, add_func=add_func)

def run_selective_augmentation(data_augmentation, input_dir, output_dir, num_aug, ratio=0.3, my_seed=42, add_func=None):
  """augmentation of subset"""
  np.random.seed(my_seed)
  images = os.listdir(input_dir)
  images_to_augm = np.random.choice(images, int(len(images)*ratio), replace=False)
  for img_name in images_to_augm:
    if img_name.endswith((".png", ".jpg", ".jpeg")):
      img_path = os.path.join(input_dir, img_name)
      augment_and_save(img_path, data_augmentation, num_aug, save=True, output_dir=output_dir, add_func=add_func)


def add_files(source_dir, dest_dir, suffix=None):
  os.makedirs(dest_dir, exist_ok=True)
  for file_name in os.listdir(source_dir):
    source_file = os.path.join(source_dir, file_name)
    if os.path.isfile(source_file):
      if suffix is not None:
        file_name = f"{os.path.splitext(file_name)[0]}_{suffix}.jpg"
      destination_file = os.path.join(dest_dir, file_name)
      shutil.copyfile(source_file, destination_file)


def sample_and_copy_files(source_dir, dest_dir, ratio=2/5, suffix=None):
  img_paths = glob(os.path.join(source_dir, "*"))
  selected_images = np.random.choice(img_paths, int(len(img_paths)*ratio), replace=False)
  for im in selected_images:
    if os.path.isfile(im):
      filename = os.path.basename(im)
      if suffix is not None:
        filename = f"{os.path.splitext(filename)[0]}_{suffix}.jpg"
      destination_file = os.path.join(dest_dir, filename)
      shutil.copyfile(im, destination_file)



def generate_model_version_name():

  from datetime import datetime

  now = datetime.now()
  month_day = now.strftime("%m-%d")  # Формат: ММ-ДД
  hours = now.strftime("%H")  # Формат: ЧЧ, тільки години
  return month_day + "-" + hours


# при дисбалансі класів
def binary_class_weights(dataset):
  count_1 = 0
  count_0 = 0
  for items, labels in dataset:
    labels = labels.numpy()
    count_1 += np.sum(labels==1)
    count_0 += np.sum(labels==0)
  weight_1 = (1/count_1) * ((count_1+count_0)/2.0)
  weight_0 = (1/count_0) * ((count_1+count_0)/2.0)
  return (weight_0, weight_1)

def initial_bias_calc(dataset):
  count_1 = 0
  count_0 = 0
  for items, labels in dataset:
    labels = labels.numpy()
    count_1 += np.sum(labels==1)
    count_0 += np.sum(labels==0)
  weight_1 = (1/count_1) * ((count_1+count_0)/2.0)
  weight_0 = (1/count_0) * ((count_1+count_0)/2.0)
  return np.log([count_1/count_0])


# при дисбалансі класів при відомій кількості екземплярів кожного класу
def binary_class_weights_2(count_1, count_0):
  weight_1 = (1/count_1) * ((count_1+count_0)/2.0)
  weight_0 = (1/count_0) * ((count_1+count_0)/2.0)
  return (weight_0, weight_1)

def initial_bias_calc_2(count_1, count_0):
  weight_1 = (1/count_1) * ((count_1+count_0)/2.0)
  weight_0 = (1/count_0) * ((count_1+count_0)/2.0)
  return np.log(count_1/count_0)


# логування
# зібрати під час навчання метрики
class MetricsLogger(Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.metrics = None
        # create file on init
        if not os.path.exists(self.filepath):
          with open(self.filepath, mode='w', newline='') as f:
              writer = csv.writer(f)
    # method Callbak class
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # get logs keys
        if self.metrics is None:
          self.metrics = list(logs.keys())
            # add metrics header to csv
          with open(self.filepath, mode='a', newline='') as f:
              writer = csv.writer(f)
              writer.writerow(['epoch'] + self.metrics)
        with open(self.filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch] + [logs[m] for m in self.metrics])


class CustomCheckpoint(Callback):
  def __init__(self, filepath, monitor_metric='val_accuracy', mode_metric='max', silent_mode=True):
    super(CustomCheckpoint, self).__init__()
    self.filepath = filepath
    self.monitor_metric = monitor_metric
    self.mode_metric = mode_metric
    self.best_metric = -float('inf') if mode_metric == 'max' else float('inf')
    self.silent_mode = silent_mode

  def on_epoch_end(self, epoch, logs=None):
    metric = logs.get(self.monitor_metric)

    save_flag = False

    if (self.mode_metric == 'max' and metric > self.best_metric) or (self.mode_metric == 'min' and metric < self.best_metric):
      self.best_metric = metric
      save_flag = True

    if save_flag:
        self.model.save(self.filepath)
        if not self.silent_mode:
          print(f"\n Model saved on epoch {epoch+1} ({self.monitor_metric}: {metric:.4f})")


# візуалізація
def plot_scores (df, model_name, metrics_list):
  n = len(metrics_list)
  fig = plt.figure(figsize=(16, int(np.ceil(n/3)*4) ))
  fig.suptitle(f'{model_name}')
  for i, metric in enumerate(metrics_list):
    ax = plt.subplot(int(np.ceil(n/3)), 3, i + 1)
    ax.plot(df.index+1, df[metric], label=metric)
    ax.plot(df.index+1, df["val_" + metric], label="val_" + metric)
    ax.legend(loc='lower right')
    ax.grid(True)
  fig2 = plt.figure(figsize=(8, 6))
  for i, metric in enumerate(metrics_list):
    plt.plot(df.index+1, df[metric], label=metric)
    plt.plot(df.index+1, df["val_" + metric], label="val_" + metric)
  plt.legend(loc='lower right')
  plt.show();

def plot_loss(df, model_name):
  fig = plt.figure(figsize=(8, 6))
  fig.suptitle(f'{model_name + " loss"}')
  plt.plot(df.index+1, df['loss'], label='train loss')
  plt.plot(df.index+1, df["val_loss"], label="validation loss")
  plt.legend(loc='lower right')
  plt.grid(True)
  plt.show();


def plot_confusion_matrix(labels, predictions, threshold=0.5):
  cm = confusion_matrix(labels, predictions > threshold)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["other", "melanoma"])
  disp.plot(cmap=plt.cm.Blues)
  plt.xlabel("Predictions")
  plt.ylabel("Actual")
  plt.title('Confusion matrix')
  plt.show()


def fbeta_metric(precision, recall, beta=2.0):
  return precision * recall * (1+beta**2)/( (beta**2) * precision + recall)


# -------functions to build custom train_ds

def scan_dataset(data_dir, class_names):
  """collect pathes and labels
      class_names: dict (class: label)"""
  # class_names = sorted(os.listdir(data_dir))
  all_image_paths = []
  all_labels = []

  # for class_index, class_name in enumerate(class_names):
  for class_name in class_names.keys():
    img_paths = glob(os.path.join(data_dir, class_name, "*"))
    all_image_paths.extend(img_paths)
    all_labels.extend([class_names[class_name]] * len(img_paths))
    print(f"Found {len(img_paths)} files of class {class_name}")
  return np.array(all_image_paths), np.array(all_labels)

def preprocess_image(path, label, size=(256, 256)):
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=3)
  img = tf.image.resize(img, size)
  label = tf.cast(label, tf.int32)
  return img, label

#------- functions to build subset of train_ds

def split_by_class(image_paths, labels):
  classes = np.unique(labels)
  class_indices = {label: np.where(labels == label)[0] for label in classes}
  return class_indices

def stratified_dataset_from_directory(data_dir, class_names, batch_size,  balanced=True, image_size=(256, 256), shuffle=True, class_proportions=None, my_seed=None):
  """creates stratified dataset with random subset of class according to proportions
      class_names: dict (class: label)
      class_proportions: dict (class_label: k of class len)"""

  image_paths, labels = scan_dataset(data_dir, class_names)
  class_indices = split_by_class(image_paths, labels)

  selected_indices = []

  if my_seed is not None:
    np.random.seed(my_seed)

  if balanced:
    # визначити менший клас
    min_class_len = min(len(indices) for indices in class_indices.values())
    for label, indices in class_indices.items():
      if len(indices) > min_class_len:
        sampled = np.random.choice(indices, min_class_len, replace=False)
      else:
        sampled = indices
      selected_indices.extend(sampled)
  else:
    class_len = {label: len(indices) for label, indices in class_indices.items()}
    for label, indices in class_indices.items():
      k = class_proportions[label] if class_proportions[label] > 0 else 0
      sampled = np.random.choice(indices, int(class_len[label]*k), replace = False if (k<=1.0) else True) if (float(k) != 1.0) else indices
      selected_indices.extend(sampled)

  if shuffle:
    np.random.shuffle(selected_indices)

  image_paths = np.array(image_paths)
  labels = np.array(labels)
  sampled_paths = image_paths[selected_indices]
  sampled_labels = labels[selected_indices]

  dataset = tf.data.Dataset.from_tensor_slices((sampled_paths, sampled_labels))
  dataset = dataset.map(lambda path, label: preprocess_image(path, label, size=image_size), num_parallel_calls=tf.data.AUTOTUNE)
  dataset = dataset.shuffle(1000).batch(batch_size)
  # dataset = dataset.prefetch(tf.data.AUTOTUNE)
  return dataset

# --------------створення та компіляція моделі з заданим bias для останнього шару
def make_model(learning_rate=1e-4, optimizer=None, loss_func=keras.losses.BinaryCrossentropy(), metrics=['accuracy'], model=None, output_bias=None):
  model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=learning_rate) if (optimizer is None) else optimizer,
      loss=loss_func,
      metrics=metrics)
  if output_bias is not None:
    model.layers[-1].bias.assign([output_bias])
  return model


def get_predictions_df(model, img_size=(256, 256), batch_size=32, images_folder=None, dataset=None, threshold = 0.5):
  if images_folder is not None:
    dataset = tf.keras.utils.image_dataset_from_directory(
    images_folder,
    label_mode=None,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
    )
  # get filenames, dataset.file_paths
  file_paths = dataset.file_paths

  predictions = model.predict(dataset, verbose=0)
  predicted_labels = (predictions >= threshold).astype(int)
  true_labels = [0 if 'other' in fp else 1 for fp in file_paths]

  predictions_df = pd.DataFrame({
      'file_path': file_paths,
      'true_label': true_labels,
      'predicted_label': np.concatenate(np.array(predicted_labels), axis=0),
      'predicted_prob': np.concatenate(np.array(predictions), axis=0)
  })
  return predictions_df


def get_predictions_df2(model, img_size=(256, 256), batch_size=32, images_folder=None, dataset=None, threshold = 0.5):
  """same as get_predictions_df, but using iteration, accepts tf.Dataset or folder path"""

  if images_folder is not None:
    dataset = tf.keras.utils.image_dataset_from_directory(
    images_folder,
    label_mode=None,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
    )

  predictions_array = []
  true_labels_array = []
  for images in dataset:
      predictions = model.predict(images, verbose=0)
      predictions_array.extend(predictions)
  file_paths = dataset.file_paths
  true_labels = [0 if 'other' in fp else 1 for fp in file_paths]
  predictions_array = np.concatenate(np.array(predictions_array), axis=0)
  # true_labels_array = np.array(true_labels)
  predicted_labels = (predictions_array >= threshold).astype(int)

  predictions_df = pd.DataFrame({
      'file_path': file_paths,
      'true_label': true_labels,
      'predicted_label': predicted_labels,
      'predicted_prob': predictions_array
  })
  return predictions_df

  
def show_predicted(predictions_df, images_pathes):
  num = len(images_pathes)
  plt.figure(figsize=(16, np.ceil(num/5)*3.2))

  for i, im in enumerate(images_pathes):
    ax = plt.subplot(int(np.ceil(num/5)), 5, i + 1)
    image = Image.open(im)
    plt.imshow(img_to_array(image).astype("uint8"))
    plt.title(f"{predictions_df.file_path[predictions_df['file_path']==im].values[0][-16:]}\n \
    True: {predictions_df.true_label[predictions_df['file_path']==im].values[0]}\
    Prediction: {predictions_df.predicted_prob[predictions_df['file_path']==im].values[0]: .2f}")
    plt.axis("off")

def ModelTraining(train_ds, val_ds, backbone_model,
                    show_summary=True, model_name=None, optimizer=None, learning_rate=1e-4,
                    loss_func=keras.losses.BinaryCrossentropy(), metrics=['accuracy'], output_bias=None,
                    class_weights=None, callbacks_list=None, epochs=20, **kwargs):

  if model_name is not None:
    print(model_name)
  if callbacks_list is None:
    callbacks_list = []

  model = make_model(learning_rate=learning_rate, loss_func=loss_func, optimizer=optimizer, metrics=metrics, model=backbone_model, output_bias=output_bias)

  if show_summary:
    model.summary(show_trainable=True)

  metrics_logger = MetricsLogger(filepath=f"metrics_logs_{model_name}.csv")
  callbacks_list.append(metrics_logger)

  history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=callbacks_list,
  class_weight=class_weights
)

  df_history = pd.DataFrame(history.history)
  df_history.to_csv(f"metrics_{model_name}.csv", index=False)

  files.download(f'/content/metrics_logs_{model_name}.csv')
  files.download(f'/content/{model_name}.keras')
  files.download(f'/content/metrics_{model_name}.csv')

  plot_scores (df_history, model_name, ['accuracy', 'recall', 'precision'])
  plot_loss(df_history, model_name)

  return model, df_history


def ModelEvaluating(test_ds, 
                    model, model_name=None,
                    metrics=['accuracy'], **kwargs):

  results = model.evaluate(test_ds, verbose=1)

  precision = results[2]
  recall = results[3]
  fbeta = fbeta_metric(precision, recall)
  results.append(fbeta)

  for metric, value in zip(["loss"] + [m.name for m in metrics] + ["f1_beta"], results):
    print(metric, ': ', value)

  # Зберегти результат оцінки на тествому наборі
  df_eval = pd.DataFrame({'Metric': ["loss"] + [m.name for m in metrics] + ["f1_beta"], 'Value': results})
  df_eval.to_csv(f'evaluate_results_{model_name}.csv', index=False)
  files.download(f'/content/evaluate_results_{model_name}.csv')

  return results


def ModelPredicting(dataset, model, model_name=None):

  predictions_df = get_predictions_df(model, img_size=(256, 256), batch_size=32, dataset=dataset)
  print(classification_report(predictions_df['true_label'], predictions_df['predicted_label']))
  print(precision_recall_fscore_support(predictions_df['true_label'], predictions_df['predicted_label'], beta=2.0, average='weighted'))

  predictions_df.to_csv(f'Predictions_{model_name}.csv', index=False)
  files.download(f'Predictions_{model_name}.csv')

  plot_confusion_matrix(predictions_df['true_label'], predictions_df['predicted_label'], threshold=0.5)

  fp_rate, tp_rate, _ = roc_curve(predictions_df['true_label'], predictions_df['predicted_prob'])
  roc_auc = auc(fp_rate, tp_rate)

  plt.figure(figsize=(8, 6))
  plt.plot(fp_rate, tp_rate, label=f"ROC curve (AUC = {roc_auc:.2f})")
  plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title(f"ROC Curve-{model_name}")
  plt.legend()
  plt.grid(True)
  plt.show()

  return predictions_df
