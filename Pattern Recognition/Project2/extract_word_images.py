import glob
import numpy
from utils import basename_without_ext
from svgpathtools import parse_path
from PIL import Image, ImageDraw
from xml.dom import minidom


image_path = "images/*.jpg"
image_paths = glob.glob(image_path)

# Iterate through all images in images and cutout the word images
for fpath in image_paths:
    base_name = basename_without_ext(fpath)

    doc = minidom.parse(f"ground-truth/locations/{base_name}.svg")
    path_nodes = doc.getElementsByTagName("path")

    im = Image.open(fpath).convert("RGBA")
    imArray = numpy.asarray(im)

    for p in path_nodes:
        id = p.attributes['id'].value
        path = p.attributes['d'].value
        path_alt = parse_path(path)

        polygon = [(path_alt[0].start.real, path_alt[0].start.imag)]

        for path in path_alt:
            x = path.end.real
            y = path.end.imag
            polygon.append((x, y))

        mask = Image.new('L', (imArray.shape[1], imArray.shape[0]), color=0)
        ImageDraw.Draw(mask).polygon(polygon, outline=255, fill=255)

        newIm = im
        newIm.putalpha(mask)

        # Cut out minimal box
        max_x = int(max(polygon, key=lambda x: x[0])[0])
        min_x = int(min(polygon, key=lambda x: x[0])[0])
        max_y = int(max(polygon, key=lambda x: x[1])[1])
        min_y = int(min(polygon, key=lambda x: x[1])[1])

        box = (int(min_x), int(min_y), int(max_x), int(max_y))
        cropped = newIm.crop(box)

        height = cropped.size[0]
        width = cropped.size[1]
        px = cropped.load()
        for h in range(height):
            for w in range(width):
                if px[h, w][3] == 0:
                    px[h, w] = (255, 255, 255, 0)

        word_image_path = f"ground-truth/word-images/{id}.png"
        cropped.save(word_image_path)
        print(f"Generated {word_image_path}")