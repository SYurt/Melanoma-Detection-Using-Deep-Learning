Melanoma Detection Using Deep Learning

This project was developed as a qualification work thesis and focuses on building a deep learning model for automatic melanoma recognition from dermoscopic images.

Problem: Early melanoma detection is critical, but visual diagnosis is challenging and error-prone.

Dataset:
two datasets of ISIC images:
 1) 2020 year dataset + melanoma images of 2019 year dataset;
 2) 2020 + 2019 years datasets.

Approach:
– Data exploration and preprocessing of dermoscopic image datasets
– Image normalization and augmentation
– Training and evaluation of convolutional neural networks using transfer learning with multiple architectures on sample datasets
– Comparison of model architectures, configurations and performance metrics
- Selection of the best-performing architecture and training on the full dataset
- Model evaluation across different datasets, analysis of classification errors, and identification of domain bias
- Experiments aimed at improving model generalization and robustness to domain shifts
- Training and evaluation of Vision Transformer (ViT) model using pretrained weights from Hugging Face
- Development of a custom model architecture, comparison of models trained from scratch with pretrained models
- Training, evaluating and comparison ensemble models

Tools & technologies:
Python, NumPy, Pandas, TensorFlow/Keras, scikit-learn, Matplotlib, Google Colab, OpenCV

Results:
The project demonstrates how deep learning models can be applied to medical image classification and highlights practical challenges such as class imbalance and model generalization.

Note: This project was completed in an academic setting and reflects my hands-on experience with deep learning workflows and experimentation. This project extends into a research work currently under preparation for publication. The focus includes model generalization across datasets and domain bias analysis.