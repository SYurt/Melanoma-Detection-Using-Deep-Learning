import cv2
import os

def resize_save_image(filename, source_dir, dest_dir, size=(256, 256)):

	height, width = size
	# path = 'test/ISIC_0024306.jpg'
	filepath = os.path.join(source_dir, filename)

	image = cv2.imread(filepath)

	resized_image = cv2.resize(image, (width, height))

	cv2.imwrite(os.path.join(dest_dir, filename), resized_image)


source_dir = ""

dest_dir = ""

size = (256, 256)

for file in os.listdir(source_dir):
	resize_save_image(file, source_dir, dest_dir, size)

print("Done")