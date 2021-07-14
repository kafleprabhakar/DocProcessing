import os, sys
sys.path.append('services')
# sys.path.append('checkbox')
import cv2

from services import checkbox_detect, util, table_analysis
# from checkbox import checkbox_detect, util
import json
# from tables import table_analysis, util
import template_extract
from base.classes import Box
import pprint
import pytesseract
from pytesseract import Output
import numpy as np
import math
from tabulate import tabulate


cwd = os.getcwd()
blank_fpath = 'demo_files/'
template_fpath = 'template/'
output_fpath = 'output/'
filled_fpath = 'filled_files/'
img_fpath = cwd + '/img/'

def show_wait_destroy(winname, img):
    cv2.imshow(winname, img)
    cv2.moveWindow(winname, 500, 0)
    cv2.waitKey(0)
    # cv2.destroyWindow(winname)





if __name__ == "__main__":
    # file_in = 'multi-jurisdiction.pdf'
    file_in = 'Exemption_filled.pdf'

    pdf_path = filled_fpath + file_in
    im_paths = util.pdf_to_image(pdf_path)
    path = im_paths[0]

    # for i, path in enumerate(im_paths):
    #     # Load image, grayscale, Gaussian blur, Otsu's threshold
    #     image = cv2.imread(path)
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #     boxes = util.get_document_segmentation(image, dilate_kernel_size=(20, 10))
    #     # util.draw_contours(image, boxes)
    #     for box in boxes:
    #         endpoints = box.get_box_endpoints()
    #         cv2.rectangle(image, endpoints[0], endpoints[1], (0, 0, 255), 2)
    #     util.show_image(image, delay=0)
    #     cv2.imwrite(f"img/bnm_c_{i}.jpg", image)

    result = table_analysis.extract_tables(path, debug=True)
    print(result)
    # result = table_analysis.return_table(path, debug=True)
    # print('the result: ', result)

    # if len(result) > 0:
    #     result = table_analysis.read_tables(path, result[0], result[1])
    #     print('the result', result)
    #     result = tabulate(result, tablefmt='grid')
    #     print(result)

    # # [bin]
    # # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    # gray = cv2.bitwise_not(gray)
    # bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
    #                             cv2.THRESH_BINARY, 15, -2)
    # # Show binary image
    # show_wait_destroy("binary", bw)
    # # [bin]
    # # [init]
    # # Create the images that will use to extract the horizontal and vertical lines
    # horizontal = np.copy(bw)
    # vertical = np.copy(bw)
    # # [init]
    # # [horiz]
    # # Specify size on horizontal axis
    # cols = horizontal.shape[1]
    # horizontal_size = cols // 30
    # # Create structure element for extracting horizontal lines through morphology operations
    # horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    # # Apply morphology operations
    # horizontal = cv2.erode(horizontal, horizontalStructure)
    # horizontal = cv2.dilate(horizontal, horizontalStructure)
    # # Show extracted horizontal lines
    # show_wait_destroy("horizontal", horizontal)
    # # [horiz]
    # # [vert]
    # # Specify size on vertical axis
    # rows = vertical.shape[0]
    # verticalsize = rows // 30
    # # Create structure element for extracting vertical lines through morphology operations
    # verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    # # Apply morphology operations
    # vertical = cv2.erode(vertical, verticalStructure)
    # vertical = cv2.dilate(vertical, verticalStructure)
    # # Show extracted vertical lines
    # show_wait_destroy("vertical", vertical)

    # # vertical = cv2.bitwise_not(vertical)
    # show_wait_destroy("vertical_bit", vertical)

    # edges = cv2.adaptiveThreshold(vertical, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
    #                             cv2.THRESH_BINARY, 3, -2)
    # show_wait_destroy("edges", edges)
    # # Step 2
    # kernel = np.ones((2, 2), np.uint8)
    # edges = cv2.dilate(edges, kernel)
    # show_wait_destroy("dilate", edges)
    # # Step 3
    # smooth = np.copy(vertical)
    # # Step 4
    # smooth = cv2.blur(smooth, (2, 2))
    # # Step 5
    # (rows, cols) = np.where(edges != 0)
    # vertical[rows, cols] = smooth[rows, cols]
    # # Show final result
    # show_wait_destroy("smooth - final", vertical)







    # # contours, _ = cv2.findContours(vertical, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # print(contours)


    # lines = cv2.HoughLinesP(vertical, 30, math.pi / 2, 100, None, 20)

    # lines = lines.squeeze()
    # print(lines)


    # for line in lines:
    #     pt1 = (line[0], line[1])
    #     pt2 = (line[2], line[3])

    #     cv2.line(image, pt1, pt2, (0, 0, 255), 3)

    # show_wait_destroy("Final", image)