'''
Usage: python age_estimation.py --video "<video file's name with extension>" (The file MUST be within the "video_files" folder)
'''

import os
import cv2
from wide_resnet import WideResNet
import numpy as np
from keras.utils.data_utils import get_file
import argparse


HAARCASSCADE_PATH = ".\\pretrained_models\\haarcascade_frontalface_alt.xml"
WEIGHTS_PATH = ".\\pretrained_models\\weights.18-4.06.hdf5"  #depth=16, width=8 in init function under class AgeGender


def labeling(cropped_frame, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(cropped_frame, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(cropped_frame, label, point, font, font_scale, (255, 255, 255), thickness)


def cropping(img_array, portion, margin=50, size=64):
    img_height, img_width, _ = img_array.shape
    if portion is None:
        portion = [0, 0, img_width, img_height]

    (x, y, w, h) = portion
    margin = int(min(w, h) * margin / 100)
    x_a = x - margin
    y_a = y - margin
    x_b = x + w + margin
    y_b = y + h + margin

    if x_a < 0:
        x_b = min(x_b - x_a, img_width - 1)
        x_a = 0
    if y_a < 0:
        y_b = min(y_b - y_a, img_height - 1)
        y_a = 0
    if x_b > img_width:
        x_a = max(x_a - (x_b - img_width), 0)
        x_b = img_width
    if y_b > img_height:
        y_a = max(y_a - (y_b - img_height), 0)
        y_b = img_height

    cropped_image = img_array[y_a: y_b, x_a: x_b]
    resized_image = cv2.resize(cropped_image, (size, size), interpolation=cv2.INTER_AREA)
    resized_image = np.array(resized_image)
    return resized_image, (x_a, y_a, x_b - x_a, y_b - y_a)


def preprocess(frame):
    face_cascade = cv2.CascadeClassifier(HAARCASSCADE_PATH)

    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray_scale, scaleFactor=1.2, minNeighbors=10, minSize=(64, 64))

    return detected_faces


class AgeGender(object):
    def __init__(self, depth=16, width=8, face_size=64):

        self.model = WideResNet(face_size, depth=depth, k=width)()
        model_path = os.path.join(os.getcwd(), "pretrained_models").replace("//", "\\")
        weight_file_path = get_file('weights.18-4.06.hdf5', WEIGHTS_PATH, cache_subdir=model_path)
        self.model.load_weights(weight_file_path)

    def detect_face(self, video, outpath_path):

        video_capture = cv2.VideoCapture(os.path.join("video_files", video))
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print("\nFRAME COUNT: ", str(frame_count)+"\n")

        writer = None
        print("EXTRACTING FRAMES AND PERFORMING PREDICTION.\n")
        while True:
            grab, frame = video_capture.read()

            if grab is True:
                detected_faces = preprocess(frame)
            else:
                break

            if detected_faces is not ():

                face_images = np.empty((len(detected_faces), 64, 64, 3))
                for i, face in enumerate(detected_faces):
                    face_image, cropped_img = cropping(frame, face, margin=50, size=64)
                    (x, y, w, h) = cropped_img
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
                    face_images[i, :, :, :] = face_image

                    results = self.model.predict(face_images)
                    genders = results[0]
                    ages = np.arange(0, 101).reshape(101, 1)
                    ages = results[1].dot(ages).flatten()

                    label = "{}, {}".format(int(ages[i]), "F" if genders[i][0] > 0.5 else "M")
                    labeling(frame, (face[0], face[1]), label)

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(outpath_path, fourcc, 30, ((frame.shape[1], frame.shape[0])), True)

            writer.write(frame)
            cv2.waitKey(1)

        video_capture.release()
        writer.release()
        cv2.destroyAllWindows()


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--depth", type=int, default=16, help="network depth")
    parser.add_argument("--width", type=int, default=8, help="network width")
    parser.add_argument("--video", help="name of the video file", required=True)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    depth = args.depth
    width = args.width
    video = args.video

    output_video_file = video[:-4] + "_output.mp4"
    outpath_path = os.path.join("video_files", "outputs", output_video_file)

    if os.path.exists(outpath_path):
        print("\n" + video + " has already been processed by the model. Playing its output file.\n")
        os.system(outpath_path)
    else:
        ag = AgeGender(depth=depth, width=width)
        ag.detect_face(video, outpath_path)
        os.system(outpath_path)


if __name__ == "__main__":
    main()
