import os, sys
sys.path.append('tables')
sys.path.append('checkbox')

from checkbox import checkbox_detect, checkbox_util
import json
from tables import table_analysis, util
import template_extract
import random

# OUTDATED PATHS - update for new github
cwd = os.getcwd()
# blank_fpath = 'Ryan_data/ExemptionCertificatesRyan/'
blank_fpath = 'demo_files/'
template_fpath = 'template/'
output_fpath = 'output/'
filled_fpath = 'filled_files/'
img_fpath = cwd + '/img/'

def make_checkbox_template(filename):
    name = os.path.basename(filename).split('.')[0]

    template = template_fpath + name + '.json'
    output_path = img_fpath + 'checkbox/' + name
    pdf_path = blank_fpath + filename

    im_paths = util.pdf_to_image(pdf_path)

    checkbox = checkbox_detect.checkbox_detect(im_paths[0], jsonFile=template,
                                                             fileout=output_path)
    # checkbox = checkbox_detect.checkbox_detect(im_paths[0], fileout=output_path, plot=True)

def extract_checkbox_data(filename, template_fname):
    name = os.path.basename(filename).split('.')[0]
    output_file = output_fpath + name + '.json'
    template_file = template_fpath + template_fname

    template_extract.extract_template(filled_fpath + filename, name, template_file, fpath_out=output_fpath,
                     file_out = output_file)

def make_table_template(filename, table_type="uniform"):
    pdf_path = blank_fpath + filename
    name = os.path.basename(filename).split('.')[0]
    img_path = img_fpath + 'table/' + name + '.jpg'
    csv_fname = name + "_" + table_type + '.csv'
    template_fname = name + "_" + table_type + '.json'

    im_paths = util.pdf_to_image(pdf_path)

    if table_type == "uniform":
        result = table_analysis.check_table(im_paths[0], outfile = img_path) #check for uniform table

        if len(result) > 0:
            table_analysis.read_tables(im_paths[0], result[0], result[1], result[2], fpath=output_fpath, #+ 'table/',
                                    csv_name=csv_fname, template_name=template_fname)
    else:
        result = table_analysis.get_horizontal_lines(im_paths[0], output_fpath + template_fname)
        print("The final results: ", result)


if __name__ == "__main__":
    # files = list(filter(lambda filename: filename.endswith(".pdf"), os.listdir(blank_fpath)))
    # make_checkbox_template(random.choice(files))
    # make_checkbox_template("multi-jurisdiction.pdf")
    extract_checkbox_data("multi-jurisdiction_filled.pdf", "multi-jurisdiction.json")
    # make_table_template("alabama_blank.pdf", table_type="non-uniform")
    
    # for filename in os.listdir(blank_fpath):
    #     if filename.endswith(".pdf"):
    #         print('Current file', filename)
    #         make_table_template(filename)
    #         # print(os.path.join(fpath, filename))
    #         # pdf_path = os.path.join(fpath, filename)
    #         # im_paths = util.pdf_to_image(pdf_path)

    #         # checkbox = checkbox_detect.checkbox_detect(im_paths[0], )
    #         # continue
    #     else:
    #         continue