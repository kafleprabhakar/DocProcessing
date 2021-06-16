import os
import json
from pdf2image import convert_from_path
from typing import List
from base.box import Box
import numpy as np
import cv2

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


def get_document_segmentation(path: str) -> List[Box]:
    """
    Given a path to an image of a document, divides the document into different segments
    by grouping close elements together and returns a list of boxes for those segments
    """
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Create rectangular structuring element and dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    # Find contours and draw rectangle
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    boxes = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        boxes.append(Box((x, y, x + w, y + h)))
        
    
    return boxes


def draw_contours(im: np.ndarray, contours: List[Box]) -> None:
    """
    Given a list of boxes, draws them on the image
    """
    contours = [np.reshape(box.get_vertices(), (-1, 1, 2)) for box in contours]
    for contour in contours:
        cv2.drawContours(im, [contour], 0, (0, 255, 0))


def show_image(im: np.ndarray, name: str = "Image", delay: int = 2000) -> None:
    """
    Displays the given image.
    --------
    name: name of the window frame
    delay: number of milliseconds to display the image
    """
    cv2.imshow('im', im)

    cv2.waitKey(2000)
    cv2.destroyAllWindows()