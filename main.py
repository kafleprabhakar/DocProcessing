import os, sys
sys.path.append('tables')
sys.path.append('checkbox')

from checkbox import checkbox_detect, checkbox_util
import json
from tables import table_analysis, util
import template_extract

# OUTDATED PATHS - update for new github
cwd = os.getcwd()
blank_fpath = 'demo_files/'
template_fpath = 'template/'
output_fpath = 'output/'
filled_fpath = 'filled_files/'
img_fpath = cwd + '/img/'

def make_checkbox_template(filename):
    name = os.path.basename(filename).split('.')[0]

    template = template_fpath + name + '.json'
    output_path = img_fpath + name + '_box'
    pdf_path = blank_fpath + filename

    im_paths = util.pdf_to_image(pdf_path)

    checkbox = checkbox_detect.checkbox_detect(im_paths[0], jsonFile=template,
                                                             fileout=output_path)

def extract_checkbox_data(filename, template_fname):
    name = os.path.basename(filename).split('.')[0]
    output_file = output_fpath + name + '.json'
    template_file = template_fpath + template_fname

    template_extract.extract_template(filled_fpath + filename, name, template_file, fpath_out=output_fpath,
                     file_out = output_file)

if __name__ == "__main__":
    # make_checkbox_template("multi-jurisdiction.pdf")
    extract_checkbox_data("multi-jurisdiction_filled.pdf", "multi-jurisdiction.json")