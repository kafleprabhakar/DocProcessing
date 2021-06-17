import os, sys
sys.path.append('services')
# sys.path.append('checkbox')
import cv2

from services import checkbox_detect, checkbox_util, table_analysis, util
# from checkbox import checkbox_detect, checkbox_util
import json
# from tables import table_analysis, util
import template_extract
from base.classes import Box
import pprint
import pytesseract
from pytesseract import Output
import numpy as np


cwd = os.getcwd()
blank_fpath = 'demo_files/'
template_fpath = 'template/'
output_fpath = 'output/'
filled_fpath = 'filled_files/'
img_fpath = cwd + '/img/'






if __name__ == "__main__":
    # file_in = 'multi-jurisdiction.pdf'
    file_in = 'alaska-table.pdf'

    pdf_path = blank_fpath + file_in
    im_paths = util.pdf_to_image(pdf_path)
    path = im_paths[0]


    # Load image, grayscale, Gaussian blur, Otsu's threshold
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
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)

    cv2.imshow('blur', blur)
    cv2.imshow('thresh', thresh)
    cv2.imshow('dilate', dilate)
    cv2.imshow('image', image)
    cv2.waitKey()

    # im = cv2.imread(path)
    # imgray = cv2.imread(path, 0)

    # img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # custom_oem_psm_config = r'--psm 3'
    # d = pytesseract.image_to_string(img, config=custom_oem_psm_config)
    # pprint.pprint(d)