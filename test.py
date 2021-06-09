import os, sys
sys.path.append('tables')
sys.path.append('checkbox')
import cv2

from checkbox import checkbox_detect, checkbox_util
import json
from tables import table_analysis, util
import template_extract
from base.box import Box


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

    _, threshold = cv2.threshold(imgray, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # print('contours', contours)

    cnt = contours[0]
    ratio=0.015
    approx = cv2.approxPolyDP(cnt, ratio * cv2.arcLength(cnt, True), True)
    approx_box = Box(approx)
    print('approx', approx)
    print('approx ravel', approx.ravel())
    print('box: ', approx_box)