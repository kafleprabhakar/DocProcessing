import os, sys
sys.path.append('tables')
sys.path.append('checkbox')
import cv2

from checkbox import checkbox_detect, checkbox_util
import json
from tables import table_analysis, util
import template_extract
from base.box import Box
import pprint
import pytesseract
from pytesseract import Output


cwd = os.getcwd()
blank_fpath = 'demo_files/'
template_fpath = 'template/'
output_fpath = 'output/'
filled_fpath = 'filled_files/'
img_fpath = cwd + '/img/'






if __name__ == "__main__":
    file3 = 'alaska-table.pdf'

    pdf_path = blank_fpath + file3
    im_paths = util.pdf_to_image(pdf_path)
    path = im_paths[0]

    im = cv2.imread(path)
    imgray = cv2.imread(path, 0)

    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    pprint.pprint(d)