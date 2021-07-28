import pytesseract
import cv2

import os
import numpy as np
import pandas as pd
import math
from typing import List, Tuple, Dict, Union, Optional
from pytesseract import Output

import util
from base.classes import Box, Cell


def remove_duplicate_boxes(boxes: List[Box]) -> List[Box]:
    """
    Removes any box in the list of boxes which overlaps too much with other box
    -----
    returns: list of unique boxes
    """
    final_boxes = []
    for box in boxes:
        duplicate = any([box.check_duplicate(other_box) for other_box in final_boxes])
        if not duplicate:
            final_boxes.append(box)

    return final_boxes


# remove overlapping lines
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


def sort_contours(boxes: List[Box], method: str = "left-to-right") -> List[Box]:
    """
    Sorts the given list of boxes according to the top left vertex of the bounding box
    under given method
    -----
    returns: the sorted list of boxes
    """
    # initialize the reverse flag
    reverse = method in ("right-to-left", "bottom-to-top")
    # handle if we are sorting against the y-coordinate or x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        get_x_or_y = lambda box: box.get_Y_range()[0]
    else:
        get_x_or_y = lambda box: box.get_X_range()[0]
    
    # construct the list of bounding boxes and sort them from top to bottom
    boxes = sorted(boxes, key=lambda box: get_x_or_y(box), reverse=reverse)
    return boxes


# read entire document
def full_pdf_detection(path):
    im = cv2.imread(path, 0)

    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    d = pytesseract.image_to_data(img, output_type=Output.DICT)

    return d


def is_vertically_adjacent(this: Box, that: Box) -> bool:
    """
    Given two boxes, returns true if this and that are vertically adjacent (i.e. vertically stacked)
    """
    this_top_left = this.get_box_endpoints()[0]
    that_top_left = that.get_box_endpoints()[0]
    if abs(this_top_left[0] - that_top_left[0]) < 10:
        if this_top_left[1] <= that_top_left[1]:
            height = this.get_height()
        else:
            height = that.get_height()

        if abs(this_top_left[1] - that_top_left[1]) < height + 5:
            return True
    
    return False


def is_horizontally_adjacent(this: Box, that: Box) -> bool:
    """
    Given two boxes, returns true if this and that are horizontally adjacent (i.e. horizontally stacked)
    """
    this_top_left = this.get_box_endpoints()[0]
    that_top_left = that.get_box_endpoints()[0]
    if abs(this_top_left[1] - that_top_left[1]) < 10:
        if this_top_left[0] <= that_top_left[0]:
            width = this.get_width()
        else:
            width = that.get_width()

        if abs(this_top_left[0] - that_top_left[0]) < width + 5:
            return True
    
    return False


def filter_boxes_with_siblings(boxes: List[Box]) -> List[Box]:
    """
    Given a list of boxes, removes all boxes with no horizontal and vertical siblings
    """
    boxes_with_siblings = []
    for i, box in enumerate(boxes):
        remaining_boxes = boxes[:i] + boxes[i + 1:]
        found_horizontal = any([is_horizontally_adjacent(box, other_box) for other_box in remaining_boxes])
        found_vertical = any([is_vertically_adjacent(box, other_box) for other_box in remaining_boxes])
        
        if found_horizontal and found_vertical:
            boxes_with_siblings.append(box)
        ## For debugging
        # else:
        #     print("Box {} was removed".format(box.get_box_endpoints()))
        #     print('Found horizontal: ', found_horizontal)
        #     print('Found vertical: ', found_vertical)
    
    return boxes_with_siblings


# only look at lines that are over 50% width of page
# sort lines, return clusters of lines
def cluster_horizontal_lines(lines):
    """
    Returns a list of list where each list is a cluster of lines within 100 pixels
    of one other and sorted in order of their y coordinate
    """
    sorted_lines = sorted(lines, key=lambda x: x[1])
    #print(sorted_lines)

    line_clusters = []
    cluster = []
    for i in range(len(sorted_lines)-1):
        #print(sorted_lines[i])
        #print(sorted_lines[i][1])
        if abs(sorted_lines[i][1]-sorted_lines[i+1][1]) < 100:
            if len(cluster) == 0:
                cluster.append(sorted_lines[i])
            cluster.append(sorted_lines[i+1])
        else:
            if len(cluster) > 0:
                line_clusters.append(cluster)
                cluster = []
    if len(cluster) > 0:
        line_clusters.append(cluster)

    return line_clusters


def sort_horizontal_contours(contours):
    """
    Returns the bounding rectangle of all the contours sorted by y, then by x.
        - Also for contours side by side  within a distance of 10 are merged
    """
    all_coord = []
    # creates list of x,y,w,h comp
    for i in range(len(contours)):
        all_coord.append(cv2.boundingRect(contours[i]))

    # Sorting
    print('*****')
    print('original {x}'.format(x=all_coord))
    all_coord.sort(key=lambda all_coord: all_coord[0])
    print('x sort {x}'.format(x=all_coord))
    all_coord.sort(key=lambda all_coord: all_coord[1])
    print('y sort {x}'.format(x=all_coord))

    print(all_coord)

    all_coord_final = []
    i = 0
    combined_contour = []

    print(len(all_coord))

    # Merge contours if needed and add to all_coord_final
    while i < len(all_coord):
        print(i)
        if len(combined_contour) == 0:
            combined_contour = list(all_coord[i])

        # check if x+w is within 10 of the next x --> combine into one contour
        if i < len(all_coord)-1 and all_coord[i][0]+all_coord[i][2] >= (all_coord[i+1][0]-10):
            # adjust width and height to include the next one over
            print(True)
            combined_contour[2] += all_coord[i+1][2] + 10

        else:
            if len(combined_contour) > 0:
                all_coord_final.append(tuple(combined_contour))
            combined_contour = []

        i += 1

    print(all_coord_final)

    print()
    return all_coord_final
    #return all_coord


def get_horizontal_lines(path, jsonFile=None):
    # goes after im_color

    img = cv2.imread(path, 0)

    im_color = cv2.imread(path)
    im_color_line = cv2.imread(path)

    # thresholding the image to a binary image
    thresh, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # inverting the image
    img_bin = 255 - img_bin

    # Length(width) of kernel as 100th of total width
    # kernel_len = np.array(img).shape[1] // 100
    kernel_len = np.array(img).shape[1] // 50
    #kernel_len_hor = np.array(img).shape[0] // 40
    kernel_len_hor = np.array(img).shape[0] // 60
    # Defining a vertical kernel to detect all vertical lines of image
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    # Defining a horizontal kernel to detect all horizontal lines of image
    # hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len_hor, 1))
    # A kernel of 2x2

    # Use horizontal kernel to detect and save the horizontal lines in a jpg
    image_2 = cv2.erode(img_bin, hor_kernel, iterations=1)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=2)

    image_ver = cv2.erode(img_bin, ver_kernel, iterations=1)
    vertical_lines = cv2.dilate(image_ver, ver_kernel, iterations=1)

    # cv2.imwrite("/Users/YOURPATH/horizontal.jpg", horizontal_lines)
    #print(PACKAGE_DIR)
    #cv2.imwrite(PACKAGE_DIR + '/horizontal_lines.jpg', horizontal_lines)


    lines = cv2.HoughLinesP(horizontal_lines, 30, math.pi/2, 0, None, 30, 1)
    lines = lines.squeeze() # Why squeeze?
    lines = remove_duplicate_lines(lines)
    #print(lines.shape)
    # filter for size
    lines2 = []
    #print(lines)
    #print(len(lines))
    for line in lines:
        #print(line)
        #print(line.shape)
        pt1 = (line[0], line[1])
        pt2 = (line[2], line[3])
        dist = math.sqrt((pt2[0]-pt1[0])**2+(pt2[1]-pt1[1])**2)
        #print(pt1)

        if dist > np.array(img).shape[0] // 2:
            lines2.append(line)


    #print(lines2)
    cluster_lines = cluster_horizontal_lines(lines2)
    #print(lines)
    print('Number of clusters in whole document: {x}'.format(x=len(cluster_horizontal_lines(lines))))
    print("")
    print('Number of clusters with length filtering: {x}'.format(x=len(cluster_lines)))

    # key is label: (area of interest on original file, label location)
    data = {}
    label = {}
    seen = {}

    for cluster in cluster_lines:
        for line in cluster:
            pt1 = (line[0], line[1])
            pt2 = (line[2], line[3])
            cv2.line(im_color_line, pt1, pt2, (0, 0, 255), 3)

    dims = im_color_line.shape
    img_resize = cv2.resize(im_color_line, (int(dims[1] / 3), int(dims[0] / 3)))



    cv2.imshow('horizontal lines', img_resize)
    cv2.waitKey(2000)


    for cluster in cluster_lines:

        #cluster_0 = cluster_lines[0]
        # Extract boxes within the areas between each line
        for i in range(len(cluster)-1):
            top_line = cluster[i]
            bottom_line = cluster[i+1]

            crop_image = img[top_line[1]:bottom_line[1], top_line[0]:top_line[2]]

            #data = pytesseract.image_to_boxes(crop_image)
            #print(data)
            rgb = cv2.cvtColor(crop_image, cv2.COLOR_GRAY2RGB)

            small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

            # threshold the image
            _, bw = cv2.threshold(small, 0.0, 255.0, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

            #cv2.imshow('bw', bw)

            # get horizontal mask of large size since text are horizontal components
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))

            connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)


            # find all the contours
            contours, hierarchy, = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            #contours, hierarchy, = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            #print(crop_image.shape)
            # Segment the text lines
            final_contours = []
            for idx in range(len(contours)):
                x, y, w, h = cv2.boundingRect(contours[idx])
                #print(x,y,w,h)
                #if w > 1000 or h < 5 or w < 5:
                if h < 5 or w < 5 or (h > (bottom_line[1]-top_line[1])/2) or (w > (top_line[2]-top_line[0])/2):
                    continue

                # check to see if the text box takes up
                if h > abs((bottom_line[1]-top_line[1]) - (y+h)):
                    continue

                final_contours.append(contours[idx])

            #print(len(final_countors))
            ## need to sort the contours in order going left to right
            final_contours = sort_horizontal_contours(final_contours)

            for idx in range(len(final_contours)):
                #num_labels_per_line = len(final_countors)

                x, y, w, h = final_contours[idx]
                #print(x, y, w, h)

                cv2.rectangle(rgb, (max(0,x-5), max(0,y-5)), (x + w+5, y + h+5), (0, 255, 0), 1)

                dims = rgb.shape
                img_resize = cv2.resize(rgb, (int(dims[1] / 1.2), int(dims[0] / 1.2)))

                cv2.imshow('rgb', img_resize)
                cv2.waitKey(1000)
                # area of interest goes from bottom of label to lower line
                # goes from same x value to just before the start of the next label
                #print(crop_image.shape)
                if idx != len(final_contours)-1:
                    x_next, y_next, _, _ = final_contours[idx+1]

                if idx == len(final_contours)-1 or y_next > y:
                    #print('last')
                    x_bound = [x-5, crop_image.shape[1]-20]
                else:
                    x_bound = [x-5, x_next-20]

                #print(x_bound)

                cv2.rectangle(rgb, (x_bound[0], y + h + 5), (x_bound[1], bottom_line[1]), (255, 0, 0), 2)

                text = pytesseract.image_to_string(crop_image[max(0, y-7):y+h+7, max(0, x-7):x+w+7])
                if len(text) != 0:
                    print(text)
                    if text in data:
                        if text not in seen:
                            seen[text] = 1
                        else:
                            seen[text] += 1

                        text = text+'_'+str(seen[text])

                    # need to scale up the coordinates to the total page
                    # crop_image = img[top_line[1]:bottom_line[1], top_line[0]:top_line[2]]
                    # current x value: added onto top_line[0]
                    # current y value: added onto top_line[1]
                    label_x = top_line[0] + max(0, x-5)
                    label_y = top_line[1] + max(0, y-5)
                    label[text] = [label_x, label_y, w+5, h+5]

                    data_x = top_line[0] + x_bound[0]
                    data_y = top_line[1] +  y + h + 5
                    data[text] = [data_x, data_y, x_bound[1]-x_bound[0], (bottom_line[1]-top_line[1])-(y + h + 5)]

                print()


            text = pytesseract.image_to_string(crop_image)
            print(text)
            #cv2.imshow('crop image', crop_image)

            #cv2.imshow('rgb', rgb)
            dims = rgb.shape
            img_resize = cv2.resize(rgb, (int(dims[1] / 1.2), int(dims[0] / 1.2)))

            cv2.imshow('rgb', img_resize)
            cv2.waitKey(1000)

    print(label)
    print(data)

    for i in (label):
        x,y,w,h = label[i]
        cv2.rectangle(im_color, (x, y), (x+w, y+h), (0, 255, 0), 2)

    for i in (data):
        x,y,w,h = data[i]
        cv2.rectangle(im_color, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # cv2.imwrite(fpath+outfile, im_color)

    dims = im_color.shape
    img_resize = cv2.resize(im_color, (int(dims[1] / 3), int(dims[0] / 3)))

    cv2.imshow('horizontal lines', img_resize)
    cv2.imwrite('testing.jpg', im_color)
    print('Saving color image')
    #cv2.imshow('horizontal lines 1', im_color)

    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    for k,v in data.items():
        data[k] = [int(val) for val in v]

    final_data = {'bounding box': data}

    if jsonFile:
        util.edit_json(jsonFile, final_data)

    return label, data

def get_table_segments(image: np.ndarray, debug: bool = False) -> Tuple[np.ndarray, List[Box]]:
    """
    Given a color image, returns a list of bounding boxes for each table in the image.
    -----
    Args:
        image: numpy array representing the image
        debug: whether or not to display the bounding boxes
    Returns:
        A tuple of two elements where the first element is the grayscale image of all the
        table-like structures and second element is the list of bounding boxes detected
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # util.show_image(gray, 'grey', 0)
    thresh, img_bin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # img_bin = 255 - img_bin
    img_bin = cv2.bitwise_not(img_bin)
    # util.show_image(img_bin, 'bin', 0)

    # Kernel lengths to extract horizontal and vertical lines
    ver_kernel_len = np.array(image).shape[1] // 40
    hor_kernel_len = np.array(image).shape[0] // 40
    # Defining kernels to detect all vertical and horizontal lines of image
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ver_kernel_len))
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hor_kernel_len, 1))

    # 1 iteration of erosion and 2 iterations of dilation to make the lines more continuous
    vertical_lines = cv2.erode(img_bin, ver_kernel, iterations=1)
    vertical_lines = cv2.dilate(vertical_lines, ver_kernel, iterations=2)

    horizontal_lines = cv2.erode(img_bin, hor_kernel, iterations=1)
    horizontal_lines = cv2.dilate(horizontal_lines, hor_kernel, iterations=2)

    # Combine horizontal and vertical lines in a new third image, with both having same weight.
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    # Eroding and thesholding the image --- Not eroding though ---
    _, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Segment the obtained image into individual tables
    im_vh_color = cv2.cvtColor(img_vh, cv2.COLOR_GRAY2BGR)
    im_vh_color = cv2.bitwise_not(im_vh_color)

    table_boxes = util.get_document_segmentation(im_vh_color, dilate_kernel_size=(1,1))
    
    util.draw_contours(im_vh_color, table_boxes)
    if debug:
        util.show_image(im_vh_color, delay=0)
    
    return img_vh, table_boxes


def group_boxes_in_rows(table_boxes: List[Box]) -> List[List[Box]]:
    """
    Arranges a list of boxes to list of rows of boxes (not necessarily in order within rows)
    -----
    Args:
        table_boxes: list of boxes sorted in top-to-bottom order
    """
    mean_height = np.mean([box.get_height() for box in table_boxes])
    delta = mean_height / 10 # Making this a small enough constant should also work

    table = []
    current_row = [table_boxes[0]]
    previous_Y = current_row[-1].get_Y_range()[0]

    for i, box in enumerate(table_boxes[1:]):
        this_Y = box.get_Y_range()[0]
        same_row_as_previous = this_Y <= previous_Y + delta

        if not same_row_as_previous: # Start a new row
            table.append(current_row)
            current_row = []

        current_row.append(box)

        # if i == len(table_boxes) - 1:
        #     table.append(current_row)
        
        previous_Y = this_Y
    table.append(current_row)
    return table


def sort_table_cells(table: List[List[Box]]) -> List[List[Cell]]:
    """
    Given a table of rows of unsorted cells, sorts the cells in each row by their x-coordinate.
    -----
    Returns:
        Table where each cell is a list of boxes in that cell
    """
    # calculating maximum number of cells in a row
    countcol = max([len(row) for row in table])

    # Retrieving the x-coordinate of the center of each column assuming each row has same number of cells
    center = np.array([int(np.mean(cell.get_X_range())) for cell in table[0]])
    center.sort()

    # Regarding the distance to the columns center, the boxes are arranged in respective order
    final_table = []

    for row in table:
        sorted_row = []

        for _ in range(countcol): # Each row will have countcol number of cells
            sorted_row.append(Cell())
        
        for box in row:
            diff = abs(center - (box.get_X_range()[0] + box.get_width() / 4))
            min_idx = np.argmin(diff)
            sorted_row[min_idx].add_box(box)

        final_table.append(sorted_row)
    
    return final_table


def find_table(img_vh: np.ndarray) -> Optional[List[List[Cell]]]:
    """
    Given a grayscale image with only a table (and no content), returns the table as a list of rows where each row
    is a list of cells and each cell is represented by list of boxes it spans.

    Algorithm inspired by:
    https://towardsdatascience.com/a-table-detection-cell-recognition-and-text-extraction-algorithm-to-convert-tables-to-excel-files-902edcf289ec
    """
    # Detect contours for following box detection
    contours, _ = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [Box(contour) for contour in contours]
    # Sort all the contours by top to bottom.
    boxes = sort_contours(boxes, method='top-to-bottom')
    # Filter only boxes of reasonable height
    # boxes = [box for box in boxes if 50 < box.get_width() < 1000 and 25 < box.get_height() < 800]

    # print(f'{len(boxes)} boxes before filtering')
    # Filter out boxes which are duplicate or don't have any siblings
    boxes = remove_duplicate_boxes(boxes)
    # print(f'{len(boxes)} boxes after removing duplicate boxes')
    table_boxes = filter_boxes_with_siblings(boxes)
    # print(f'{len(table_boxes)} boxes after removing filtering for siblings')

    if len(table_boxes) == 0:
        print('Not a Table')
        return None
    else:
        print('Table Found')
        list_of_rows = group_boxes_in_rows(table_boxes)
        table = sort_table_cells(list_of_rows)
        
        return table


def read_tables(image: np.ndarray, table: List[List[Cell]], fpath: str = "",\
                csv_name: str = "", template_name: str = "") -> List[List[str]]:
    """
    Reads the content of the table in the given image
    Also mutates the table to include the text in each cell
    -----
    Args:
        image: image to read table from
        table: table to read
        fpath: path of the folder to save the csv and template in
        csv_name: name of the csv file to save
        template_name: name of the template json file to save
    Returns:
        List of lists of strings representing the content of the table
    """
    print('reading table content')
    EXCLUDE_SYMBOLS = ['!', '®', '™', '?', "|", "~"]
    # from every single image-based cell/box the strings are extracted via pytesseract and stored in a list
    pd.set_option('display.max_columns', None)

    empty_boxes = {}
    table_contents = []
    for row in table:
        print('new row')
        row_content = []
        for cell in row:
            print('new cell')
            cell_content = ''
            for box in cell.get_boxes():
                if box.get_width() > 0 and box.get_height() > 0:
                    cell_content += util.read_text_in_patch(image, box)
                    print('finished reading a box')
            cell_content = cell_content.strip()
            row_content.append(cell_content)
            cell.set_content(cell_content)
        table_contents.append(row_content)
    
    print('Finished reading table content')
    # Creating a dataframe of the generated OCR list
    df = pd.DataFrame(table_contents)

    if len(fpath) > 0:
        df.to_csv(fpath + csv_name)

        final_data = {'uniform_table':empty_boxes, "df_file": fpath+csv_name}

        jsonFile = fpath + template_name

        util.edit_json(jsonFile, final_data)

    # return df.to_dict(orient='records')
    return df.to_numpy().tolist()


def extract_tables(path, outfile: str = None, debug: bool = False) -> List[List[List[str]]]:
    """
    Given a path to an image, extract the tables from the PDF and save them as a CSV file.
    -----
    Args:
        path: path to the image
        outfile: path to the output CSV file
        debug: whether or not to display the tables
    """
    im_color = cv2.imread(path)
    img_copy = im_color.copy()
    im_vh, table_boxes = get_table_segments(im_color, debug=debug)
    # util.show_image(im_vh, "VH", 0)
    table_boxes.reverse() # since the boxes are detected bottom to top

    table_contents = []
    tables = []
    for table in table_boxes:
        only_table = util.remove_all_except_boxes(im_vh, [table])
        final_table = find_table(only_table)

        if final_table:
            table_content = read_tables(im_color, final_table)
            table_contents.append(table_content)
            tables.append(final_table)
        
            all_cells = util.flatten_table(final_table)
            util.draw_contours(img_copy, all_cells)
    
    if debug:
        util.show_image(img_copy, delay=0)
    if outfile:
        cv2.imwrite(outfile, img_copy)
    
    return tables


    # data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    # confidences = data['conf']
    # texts = data['text']
    # heights = data['height']