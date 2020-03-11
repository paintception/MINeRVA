
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from os.path import join as jp
import os
import pandas as pd


def square_image(im, size=(224, 224)):
    width, height = im.size

    delta = width - height
    if delta > 0:
        transform = transforms.Compose([transforms.Pad((0, delta // 2, 0, delta - delta // 2)),
                                        transforms.Resize(size)])
        im = transform(im)
    elif delta < 0:
        transform = transforms.Compose([transforms.Pad((-delta // 2, 0, -delta + delta // 2, 0)),
                                        transforms.Resize(size)])
        im = transform(im)
    else:
        transform = transforms.Compose([transforms.Resize(size)])
        im = transform(im)

    return im


def map_images(data_path, base, part, save_path, class_name='instrument_name', crop=True):
    df = pd.read_csv(os.path.join(base, part + '.txt'))

    for row in df.iterrows():
        file_name = row[1]['image_filename']
        instrument = row[1][class_name]
        bb = row[1]['bounding_box_coordinates'].split(",")

        bb = list(map(int, bb))
        x_min, y_min, x_max, y_max = bb

        try:
            img = Image.open(os.path.join(data_path, file_name.replace('%22', '"')))
        except:
            print("The annotation is corrupted:", file_name)
            continue

        if min(x_min, y_min) < 0 and crop:
            # annotation is corrupted
            continue

        if crop:
            cropped = img.crop((x_min, y_min, x_max, y_max))
        else:
            cropped = img
        cropped = square_image(cropped)

        class_path = jp(jp(save_path, part), instrument)

        if not os.path.isdir(class_path):
            os.mkdir(class_path)

        cropped.save(jp(class_path, file_name))


if __name__ == '__main__':

    top = "source"

    for p in ("train", "dev", "test"):
        map_images(data_path="/home/nbanar/pycharmProjects/datasets/RawInstrumentsFullCollection",
                   base=f"/home/nbanar/pycharmProjects/art_detector/classification_of_crops/splits/{top}/",
                   part=p,
                   save_path=f"/home/nbanar/pycharmProjects/art_detector/classification_of_crops/data/{top}/",
                   class_name='instrument_name',
                   crop=False)

