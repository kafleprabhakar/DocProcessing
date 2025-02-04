import os
import json
from pdf2image import convert_from_path
from typing import List, Tuple, Any
from base.classes import Box, Cell
import numpy as np
import cv2
import math
from base.customEncoder import CustomEncoder
import random
import pytesseract

def edit_json(jsonFile, newData):
    try:
        with open(jsonFile, "r+") as file:
            data = json.load(file)
            data.update(newData)
            file.seek(0)
            json.dump(data, file)

    except:
        with open(jsonFile, "w") as file:
            json.dump(newData, file)


def pdf_to_image(pdf_path: str) -> List[str]:
    print('the pdf path', pdf_path)
    images = convert_from_path(pdf_path)
    filename = os.path.basename(pdf_path).split('.')[0]
    paths = []
    for i in range(len(images)):
        img_name = 'img/filename_' + str(i) + '.jpg'
        images[i].save(img_name, 'JPEG')
        paths.append(img_name)
    return paths


def get_document_segmentation(image, dilate_kernel_size: Tuple[int, int] = (10, 3)) -> List[Box]:
    """
    Given an image of a document, divides the document into different segments
    by grouping close elements together and returns a list of boxes for those segments
    """
    padding = (-dilate_kernel_size[0] + 1, -dilate_kernel_size[1] + 1, dilate_kernel_size[0] - 1, dilate_kernel_size[1] - 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Create rectangular structuring element and dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilate_kernel_size)
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    # Find contours and draw rectangle
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    boxes = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        vertices = (x, y, x + w, y + h)
        # Remove the padding so we have a tighter bound the enclosing area and clear enough space between the patches
        vertices = tuple(vertices[i] - padding[i] for i in range(4))
        boxes.append(Box(vertices))
        
    # show_image(dilate, name="Dilated", delay=0)
    return boxes


def remove_long_lines(im: np.ndarray, threshold: int = 100) -> np.ndarray:
    """
    Given an image and a threshold amount, removes all the horizontal and vertical lines of length greater
    than the threshold from the image
    -----
    returns: a new image with all long horizontal and vertical lines removed
    """
    vertical_lines = get_vertical_lines(im, threshold=threshold)

def random_rgb():
    """
    Generates a random RGB color
    """
    return (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))

def draw_contours(im: np.ndarray, contours: List[Box], random_color: bool = False) -> None:
    """
    Given a list of boxes, draws them on the image
    """
    contours = [np.reshape(box.get_vertices(), (-1, 1, 2)) for box in contours]
    for contour in contours:
        color = (255, 0, 0) if not random_color else random_rgb()
        cv2.drawContours(im, [contour], 0, color, 2)


def show_image(im: np.ndarray, name: str = "Image", delay: int = 2000) -> None:
    """
    Displays the given image.
    --------
    name: name of the window frame
    delay: number of milliseconds to display the image
    """
    cv2.imshow(name, im)

    cv2.waitKey(delay)
    cv2.destroyAllWindows()


# removes all duplicate boxes
def remove_duplicate_lines(lines):
    final_lines = []
    for line in lines:
        duplicate = False
        for final_line in final_lines:
            if abs(line[0]-final_line[0]) < 10 and abs(line[1]-final_line[1])<10 and abs(line[2]-final_line[2]) < 10 and abs(line[3]-final_line[3])<10:
                duplicate = True
        if not duplicate:
            final_lines.append(line)
    return final_lines


# returns all major horizontal lines, used for creating label boundaries
def get_vertical_lines(path, show=False, show_duration=1000):
    img = cv2.imread(path, 0)

    im_color_line = cv2.imread(path)

    # thresholding the image to a binary image
    thresh, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # inverting the image
    img_bin = 255 - img_bin

    # Length(width) of kernel as 100th of total width
    kernel_len = np.array(img).shape[1] // 50

    # Defining a vertical kernel to detect all vertical lines of image
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))

    # Changing to iterations 1 impacts detection of blank lines
    image_1 = cv2.erode(img_bin, ver_kernel, iterations=1)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=1)

    lines = cv2.HoughLinesP(vertical_lines, 30, math.pi / 2, 100, None, 20, 1)

    lines = lines.squeeze()
    lines = remove_duplicate_lines(lines)

    for line in lines:
        pt1 = (line[0], line[1])
        pt2 = (line[2], line[3])

        cv2.line(im_color_line, pt1, pt2, (0, 0, 255), 3)

    dims = im_color_line.shape
    img_resize = cv2.resize(im_color_line, (int(dims[1] / 3), int(dims[0] / 3)))

    if show:
        cv2.imshow('vertical lines', img_resize)
        cv2.waitKey(show_duration)

    return lines

def save_data_to_json(data, filename, key):
    try:
        with open(filename, "r") as file:
            prev_data = json.load(file)
    except:
        prev_data = {}

    prev_data[key] = data

    with open(filename, "w") as file:
        json.dump(prev_data, file, cls= CustomEncoder)


def read_text_in_patch(image: np.ndarray, patch: Box) -> str:
    """
    Given an image and a box representing a patch on the image, reads and returns the text in the patch
    """
    EXCLUDE_SYMBOLS = ['!', '®', '™', '?', "|", "~"]
    patch_X = patch.get_X_range()
    patch_Y = patch.get_Y_range()
    patch_img = image[patch_Y[0]:patch_Y[1], patch_X[0]:patch_X[1]]
    # print("Patch Shape", patch_img.shape)
    # show_image(patch_img, name="Patch", delay=0)

    raw_text = pytesseract.image_to_string(patch_img)
    final_text = ''
    for char in list(raw_text):
        if char not in EXCLUDE_SYMBOLS:
            final_text += char

    return final_text.replace('\n', ' ')


def remove_all_except_boxes(image: np.ndarray, boxes: List[Box]) -> np.ndarray:
    """
    Given an image and a list of boxes, removes everything outside of the boxes
    """
    final_img = np.zeros(image.shape, np.uint8)
    for box in boxes:
        box_X, box_Y = box.get_X_range(), box.get_Y_range()
        final_img[box_Y[0]:box_Y[1], box_X[0]:box_X[1]] = image[box_Y[0]:box_Y[1], box_X[0]:box_X[1]].copy()

    return final_img

def flatten_table(table: List[List[Cell]]) -> List[Box]:
    """
    Flattens a table to give list of individual boxes
    """
    return [box for row in table for cell in row for box in cell.get_boxes()]