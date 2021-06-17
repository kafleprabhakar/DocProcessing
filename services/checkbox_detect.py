import numpy as np
import cv2
import pytesseract
from pytesseract import Output
import json
from typing import List, Tuple, Dict, Union, Optional
from base.classes import Box, Checkbox
from base.customEncoder import CustomEncoder
from checkbox_util import save_data_to_json
from util import draw_contours, show_image, get_document_segmentation


def find_inner_checkbox(checkboxes: List[Checkbox], approx: Box, thold: int = 2) -> bool:
    """
    returns: True if the box `approx` can be safely added to the list `checkboxes`. A box is safe
            if no other smaller box overlaps too much with it (determined by threshold)
    - Also deletes any box which is larger than this box and overlaps too much
    """
    for i in range(len(checkboxes) - 1, -1, -1):
        box = checkboxes[i].get_box()
        box_center = box.get_center()
        box_area = box.get_area()

        approx_center = approx.get_center()
        approx_area = approx.get_area()

        # If the centers of the two boxes are within the threshold
        # and if the approx_checkbox has the smaller area, rm the larger box
        if abs(box_center[0] - approx_center[0]) <= thold and abs(box_center[1] - approx_center[1]) <= thold:
            if approx_area < box_area:
                del checkboxes[i]
                return True
            else:
                return False
    return True


# Find the minimum dimensions of the
def minimum_box_dimensions(checkboxes: List[Box]) -> Tuple[int]:
    """
    Returns a tuple of minimum height and width among the given boxes
    """
    min_height = min([box.get_height() for box in checkboxes])
    min_width = min([box.get_width() for box in checkboxes])

    return int(min_height), int(min_width)


def find_checkboxes(threshold: np.ndarray, ratio: float, delta: int,\
                    side_length_range: Tuple[int, int]) -> List[Checkbox]:
    """
    Given a threshold image, returns the checkboxes in the image
    """
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    checkboxes = []

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, ratio * cv2.arcLength(cnt, True), True)

        # only detect polygons with 4 sides
        if len(approx) == 4:
            approx_box = Box(approx)
            width = approx_box.get_width()
            height = approx_box.get_height()

            # If the box is a square within a given size range and delta
            if abs(height - width) < delta and width in range(*side_length_range) and height in range(*side_length_range):
                if not checkboxes or find_inner_checkbox(checkboxes, approx_box):
                    
                    checkboxes.append(Checkbox(approx_box))
    return checkboxes


def get_percent_filled(threshold: np.ndarray, center: np.ndarray, min_width: int,\
                       min_height: int) -> float:
    """
    Sees the area of size min_width x min_height centered around `center` and returns
    the percentage of the area that is filled
    """
    total_pixels = min_height * min_width
    half_width, half_height = int(min_width / 2), int(min_height / 2)
    count = 0
    for i in range(center[0] - half_width, center[0] + half_width):
        for j in range(center[1] - half_height, center[1] + half_height):
            # fill in area if white - used for debugging
            if threshold[j][i] > 200:
                threshold[j][i] = 100
            # if pixel is black, add to count
            if threshold[j][i] < 5:
                count += 1
    return round(count / total_pixels, 1)


def checkbox_detect(path: str, ratio: float = 0.015, delta: int = 12, side_length_range: Tuple[int, int] = (16,51),
                    plot: bool = True, fileout: Optional[str] = None, jsonFile: Optional[str] = None) -> List[List[Checkbox]]:
    """
    Detects checkboxes in the image in the given path
    """
    im = cv2.imread(path)
    imgray = cv2.imread(path, 0)
    _, threshold = cv2.threshold(imgray, 200, 255, cv2.THRESH_BINARY)
    
    checkboxes = find_checkboxes(threshold, ratio, delta, side_length_range)
    checkbox_boxes = [checkbox.get_box() for checkbox in checkboxes]
    draw_contours(im, checkbox_boxes)

    print('Number of checkboxes found: ', len(checkboxes))
    if len(checkboxes) == 0:
        return

    # Find the percent filled for each checkbox
    min_height, min_width = minimum_box_dimensions(checkbox_boxes)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    for checkbox in checkboxes:
        box = checkbox.get_box()
        center = box.get_center().astype(int)
        percent = get_percent_filled(threshold, center, min_width, min_height)

        # box_dict['percent_filled'] = percent
        checkbox.set_percent_filled(percent)
        cv2.putText(im, str(percent), (center[0] - 60, center[1] + 5), font, 1.2, (0, 0, 255), thickness=2)

    if fileout:
        save_img = str(fileout) + '_1.jpg'
        cv2.imwrite(save_img, im)

    if plot:
        show_image(im)

    checkboxes = get_unique_checkboxes(checkboxes)
    add_checkbox_label(path, checkboxes, saveImg=fileout, plot=True)
    clusters = cluster_checkboxes(path, checkboxes, saveImg=fileout)

    if jsonFile:
        save_data_to_json(clusters, jsonFile, 'checkbox')

    return clusters


def add_checkbox_label(path: str, checkboxes: List[Checkbox], plot: bool = True, saveImg: Optional[str] = None) -> None:
    """
    Given the image path and the list of checkboxes detected in the image, augments each checkbox with
    the label found in the image.
    """
    image = cv2.imread(path)
    segments = get_document_segmentation(image)
    print('----- New label algorithm -----')
    for checkbox in checkboxes:
        # Need a method to narrow it down if there are multiple segments of interests
        segments_of_interests = [segment for segment in segments if segment.contains(checkbox.get_box())]
        focus_segment = segments_of_interests[0]
        top_left, bottom_right = focus_segment.get_box_endpoints()

        if plot or saveImg:
            cv2.rectangle(image, top_left, bottom_right, (36,255,12), 2)
        
        focus_img = image[top_left[1]: bottom_right[1], top_left[0]: bottom_right[0]]
        label = pytesseract.image_to_string(focus_img).strip().replace('\n', ' ')
        
        checkbox.set_label(label)
        checkbox.set_patch(focus_segment)

    if plot:
        show_image(image, name="Checkboxes and their labels")
    if saveImg:
        cv2.imwrite(saveImg + "_with_labels.jpg", image)
        

def cluster_checkboxes(path: str, checkboxes: List[Checkbox], plot: bool = False, saveImg: Optional[str] = None) -> List[List[Checkbox]]:
    """
    Given the path of the image and all the checkboxes detected in the image, clusters
    the checkboxes together based on the proximity of their checkbox-label patch segment.
    -----
    Returns: list of clusters where each cluster is a list of checkboxes in that cluster
    """
    image = cv2.imread(path)
    print('Image shape', np.array(image).shape)
    patch_image = np.ones(np.array(image).shape, np.uint8) * 255
    
    # Draw boxes of the checkbox patches in the blank patch_image
    for checkbox in checkboxes:
        patch = checkbox.get_patch_box()
        top_left, bottom_right = patch.get_box_endpoints()
        cv2.rectangle(patch_image, top_left, bottom_right, (0,0,0), 2)

    # Dilate the above patch image so that patches close together are clustered together
    patch_grouping = get_document_segmentation(patch_image, dilate_kernel_size=(1, 8))
    clusters = []
    for patch in patch_grouping:
        top_left, bottom_right = patch.get_box_endpoints()
        cluster = [checkbox for checkbox in checkboxes if patch.contains(checkbox.get_patch_box())]
        clusters.append(cluster)
        cv2.rectangle(image, top_left, bottom_right, (36, 255, 12), 2)

    if plot:
        show_image(image, name="Black boxed patches")
    if saveImg:
        cv2.imwrite(saveImg + "_with_labels_clusters.jpg", image)

    return clusters

# Type of checkbox: List[Dict[str, Union[int, List[float], Box, float]]]
def get_unique_checkboxes(checkboxes: List[Checkbox]) -> List[Checkbox]:
    """
    Returns a dict of checkboxes after replacing all the overlapping checkboxes by single unique one
    """
    unique_checkboxes = []
    idxs_to_remove = []
    for i, checkbox in enumerate(checkboxes):
        if i not in idxs_to_remove:
            # checkbox["number"] = new_idx
            # new_idx += 1
            unique_checkboxes.append(checkbox)
        for j, other_checkbox in enumerate(checkboxes):
            if i != j and checkbox.get_box().check_overlap(other_checkbox.get_box()):
                idxs_to_remove.append(j)
    
    return unique_checkboxes


def add_checkbox_label_old(path, checkbox_dicts, clusters, fileout=None, show_labels_box=True):
    """
    Detects the label of the checkboxes in `checkbox_dicts` and modifies the dictionary by adding
    the property `label` to each checkbox mapping to the detected label.
    """
    im = cv2.imread(path)

    for k, cluster in clusters.items():
        y_upperbound = None
        for i, checkbox in enumerate(cluster["checkboxes"]):

            box = checkbox['box']

            x_range = box.get_X_range()
            y_range = box.get_Y_range()

            try:
                y_lowerbound = y_range[0] + cluster["y gaps"][i]
            except:
                y_lowerbound = y_range[1] + 15

            # if y_upperbound is None:
            y_upperbound = y_range[0] - 10

            cv2.rectangle(im, (x_range[1]+11, y_upperbound), (cluster["xlabel_boundary"], y_lowerbound), (0, 255, 0), 2)

            crop = im[y_upperbound: y_lowerbound, x_range[1]+11: cluster["xlabel_boundary"]]

            y_upperbound = y_lowerbound - 15

            if crop.shape[1] != 0:
                # if plot:
                #     cv2.imshow("crop", crop)
                #     cv2.waitKey(2000)
                #     cv2.destroyAllWindows()

                label = pytesseract.image_to_string(crop)
                checkbox_dicts[checkbox["number"]]["label"] = label

                d = pytesseract.image_to_data(crop, output_type=Output.DICT)

                label2 = (' ').join([d['text'][i] for i in range(len(d['text']))
                                 if (d['text'][i]!=' ' and d['text'][i] != '' and d['conf'][i] > 0)])

                # Better way to detect error?
                if len(label2) == 0:
                    label2 = "Error"

                print(label.strip())
                checkbox['label'] = label2

    if show_labels_box:
        show_image(im, name="Labels")

    if fileout:
        cv2.imwrite(fileout+"_labels.jpg", im)


# reads from output of above file
def checkbox_read(path, checkbox_dicts):

    imgray = cv2.imread(path, 0)
    _, threshold = cv2.threshold(imgray, 200, 255, cv2.THRESH_BINARY)

    data = {}
    checkboxes = [Box(checkbox['box']) for checkbox in checkbox_dicts]
    min_height, min_width = minimum_box_dimensions(checkboxes)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    for box_dict in checkbox_dicts:
        box = Box(box_dict['box'])
        count = 0
        center = box.get_center().astype(int)

        percent = get_percent_filled(threshold, center, min_width, min_height)
        data[box_dict['label']] = percent > 0.15

    return data

# cluster based on x coord
def cluster_checkbox_old(checkbox_dicts, im=None, showLabelBound=False, boundarylines=None):
    """
    Clusters the checkboxes in `checkbox_dicts` based on their left x coordinates and y-gap between
    other checkboxes.
    Parameters
    ----------
    checkbox_dicts: dictionary containing information about the checkboxes
    im: an opencv image. When set to some image, displays the detected clusters on the image
    showLabelBound: when set to true, shows the bounds of the labels too on the above image
    boundarylines: vertical boundary lines on the image; useful when the document has checkbox sections which
        don't span the whole width of the document
    """
    threshold = 3*40
    seen_x = {}

    # Group boxes by their leftmost x position
    for checkbox in checkbox_dicts:
        box = checkbox['box']
        min_x = box.get_X_range()[0]

        if min_x not in seen_x:
            seen_x[min_x] = []
        seen_x[min_x].append(checkbox)

    clusters = {}
    count = 0
    # Cluster the checkboxes depending on their left x coordinates, and y distance
    for x in seen_x:
        seen_x[x] = sorted(seen_x[x], key=lambda checkbox: checkbox['box'].get_Y_range()[0]) # Sort by min y value

        previous_box = None
        for checkbox in seen_x[x]:
            box = checkbox['box']
            box_top_y = box.get_Y_range()[0]
            box_left_x = box.get_X_range()[0]
            prev_bottom_y = previous_box.get_Y_range()[1] if previous_box else None
            prev_top_y = previous_box.get_Y_range()[0] if previous_box else None
            prev_right_x = previous_box.get_X_range()[1] if previous_box else None

            if prev_bottom_y is None or abs(box_top_y - prev_top_y) > threshold:
                if prev_bottom_y is not None: # finish off last cluster; key is always already defined
                    clusters[key]['dims'].append((prev_right_x + 5, prev_bottom_y + 5))
                
                # Start a new cluster
                key = "cluster_" + str(count)
                count += 1
                clusters[key] = {}
                clusters[key]["checkboxes"] = [checkbox]
                clusters[key]['dims'] = [(box_left_x - 5, box_top_y - 5)]
                clusters[key]['y gaps'] = []

            else: # Add to the running cluster
                clusters[key]["checkboxes"].append(checkbox)
                clusters[key]['y gaps'].append(abs(box_top_y - prev_top_y))
            
            previous_box = box

        # The last checkbox 
        last_bottom_y = previous_box.get_Y_range()[1]
        last_right_x = previous_box.get_X_range()[1]
        clusters[key]['dims'].append((last_right_x + 5, last_bottom_y + 5))

        count += 1

    # create label bondaries
    # for each cluster, seen if the y coords overlap, if they do mark the left side as a boundary for the label
    for k, cluster in clusters.items():
        for k2, cluster2 in clusters.items():
            if k == k2:
                continue
            else:
                #if the x coordinates > current x coord
                if cluster2["dims"][0][0] > cluster["dims"][0][0]:
                    # For two side by side clusters, end of cluster 1's label is the start of cluster 2's left side
                    if (cluster["dims"][0][1] <= cluster2["dims"][0][1] <= cluster["dims"][1][1]) or (cluster["dims"][0][1] <= cluster2["dims"][1][1] <= cluster["dims"][1][1]):

                        if "xlabel_boundary" not in cluster:
                            cluster["xlabel_boundary"] = cluster2["dims"][0][0]-20
                        elif cluster["xlabel_boundary"] > cluster2["dims"][0][0]-20:
                            cluster["xlabel_boundary"] = cluster2["dims"][0][0] - 20

                # if the y coordinates overlap, bottom is btw
        if "xlabel_boundary" not in cluster:
            cluster["xlabel_boundary"] = im.shape[1]-20

    # To take care of boundries of the section containing the checkbox
    if boundarylines is not None:
        for k, cluster in clusters.items():
            # compare to each line x position

            for line in boundarylines:
                p1 = (line[2], line[3])
                p2 = (line[0], line[1])

                # line must be right bound
                if p1[0] > cluster["dims"][0][0]:
                    #check if y values overlap

                    if (p1[1] <= cluster["dims"][0][1]  <= p2[1]) or \
                            (p1[1] <= cluster["dims"][1][1] <= p2[1]):
                        if p1[0] < cluster["xlabel_boundary"]:

                            cluster["xlabel_boundary"] = p1[0]

    #plot clusters
    if im is not None:
        for k, cluster in clusters.items():
            cv2.rectangle(im, cluster["dims"][0], cluster["dims"][1], (255, 0, 0), 3)
            if showLabelBound:
                cv2.rectangle(im, cluster["dims"][0], (cluster["xlabel_boundary"], cluster["dims"][1][1]), (0, 0, 255), 2)

        dims = im.shape
        im1 = cv2.resize(im, (int(dims[1] / 3), int(dims[0] / 3)))
        cv2.imshow('im', im1)

        cv2.waitKey(6000)
        cv2.destroyAllWindows()
    print(count)
    print(clusters)

    return clusters











