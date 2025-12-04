import cv2
import os


def sort_by_prefix(item):
    return int(item[:-4])

image_folder = '/home/lyq/Dataset/BS3D/dining/color'

video_name = '/home/lyq/Videos/BS3D-dining.mp4'

if 'BS3D' in image_folder:
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
else:
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

images.sort(key=sort_by_prefix)

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

if 'scannet' in image_folder:
    frame = cv2.resize(frame, (640, 480))
    height, width, layers = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_name, fourcc, 30, (width, height))

for image in images:
    print(image)
    if 'scannet' not in image_folder:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    else:
        img =cv2.imread(os.path.join(image_folder, image))
        img = cv2.resize(img, (640, 480))
        video.write(img)

video.release()