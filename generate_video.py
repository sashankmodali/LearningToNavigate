import cv2
import os
from natsort import natsorted

env_folder = 'tmp/dump/exp1/episodes/1/6'
image_folder = env_folder
video_name = env_folder+'video.avi'
import re
def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

images = [img for img in natsorted(os.listdir(image_folder)) if img.endswith(".png")]

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# video = cv2.VideoWriter(video_name, 0, 15, (width,height))
video = cv2.VideoWriter(video_name, 0, 15, (width//2,height//2))
for image in images:
    image_ = cv2.imread(os.path.join(image_folder, image))
    image_ = cv2.resize(image_,(width//2,height//2),interpolation=cv2.INTER_AREA)
    video.write(image_)
    # video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()