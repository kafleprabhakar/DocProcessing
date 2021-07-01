import numpy as np
import cv2
import pytesseract
from pytesseract import Output
import json
from typing import List, Tuple, Dict, Union, Optional
from base.classes import Box, Checkbox
from base.customEncoder import CustomEncoder
from util import draw_contours, show_image, get_document_segmentation, save_data_to_json


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
    checkboxes.reverse() # The contours returned by opencv if from bottom to up
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
                    plot: bool = True, fileout: Optional[str] = None, jsonFile: Optional[str] = None) -> Tuple[List[List[Checkbox]], str]:
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

        checkbox.set_percent_filled(percent)
        cv2.putText(im, str(percent), (center[0] - 60, center[1] + 5), font, 1.2, (0, 0, 255), thickness=2)

    if fileout:
        save_img = str(fileout) + '_1.jpg'
        cv2.imwrite(save_img, im)

    if plot:
        show_image(im)

    checkboxes = get_unique_checkboxes(checkboxes)
    patch_image = add_checkbox_label(path, checkboxes, saveImg=fileout, plot=True)
    clusters = cluster_checkboxes(path, checkboxes, saveImg=fileout)

    if jsonFile:
        save_data_to_json(clusters, jsonFile, 'checkbox')

    return clusters, patch_image


def add_checkbox_label(path: str, checkboxes: List[Checkbox], plot: bool = True, saveImg: Optional[str] = None) -> str:
    """
    Given the image path and the list of checkboxes detected in the image, augments each checkbox with
    the corresponding label and the patch it belongs to in the image.
    -----
    Returns: image path of checkbox-label patches plotted on it if saveImg is set to a string, else empty string
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
            cv2.rectangle(image, top_left, bottom_right, (36, 255, 12), 2)
        
        focus_img = image[top_left[1]: bottom_right[1], top_left[0]: bottom_right[0]]
        label = pytesseract.image_to_string(focus_img).strip().replace('\n', ' ')
        
        checkbox.set_label(label)
        checkbox.set_patch(focus_segment)

    imgFileName = ''
    if plot:
        show_image(image, name="Checkboxes and their labels")
    if saveImg:
        imgFileName = saveImg + "_with_labels.jpg"
        cv2.imwrite(imgFileName, image)
    return imgFileName
        

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
        imgFileName = saveImg + "_with_labels_clusters.jpg"
        cv2.imwrite(imgFileName, image)

    clusters.reverse() # The contours returned by opencv if from bottom to up
    return clusters


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


def checkbox_read(path, checkbox_clusters):

    imgray = cv2.imread(path, 0)
    _, threshold = cv2.threshold(imgray, 200, 255, cv2.THRESH_BINARY)

    data = {}
    checkbox_dicts = [checkbox for cluster in checkbox_clusters for checkbox in cluster]
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