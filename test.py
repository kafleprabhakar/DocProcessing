import os, sys
sys.path.append('tables')
sys.path.append('checkbox')

from checkbox import checkbox_detect, checkbox_util
import json
from tables import table_analysis, util
import template_extract


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

    vert_lines = checkbox_util.get_vertical_lines(im_paths[0], True, 10000)