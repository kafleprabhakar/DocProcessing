import os, sys
# sys.path.append('tables')
sys.path.append('services')

from services import checkbox_detect, checkbox_util, table_analysis, util, template_extract
# from checkbox import checkbox_detect, checkbox_util
import json
# from tables import table_analysis, util
# import template_extract
import random

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
    # pdf_path = filled_fpath + filename

    im_paths = util.pdf_to_image(filename)

    vert_lines = checkbox_util.get_vertical_lines(im_paths[0])

    checkbox = checkbox_detect.checkbox_detect(im_paths[0], jsonFile=template,
                                                             fileout=output_path, boundarylines=vert_lines)
    # checkbox = checkbox_detect.checkbox_detect(im_paths[0], fileout=output_path, plot=True)

def extract_data(filename, template_fname):
    name = os.path.basename(filename).split('.')[0]
    output_file = output_fpath + name + '.json'
    template_file = template_fpath + template_fname
    file_path = filled_fpath + filename
    im_paths = util.pdf_to_image(file_path)

    result = template_extract.extract_template(im_paths[0], file_path, template_file, fpath_out=output_fpath,
                     file_out = output_file)
    print(result)

def make_table_template(filename, table_type="uniform"):
    pdf_path = filled_fpath + filename
    name = os.path.basename(filename).split('.')[0]
    img_path = img_fpath + name + '.jpg'
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

# def extract_table_data(filename, template_fname):


if __name__ == "__main__":
    filename = blank_fpath + "alaska-table.pdf"
    # filename = filled_fpath + "alaska_FILLED.pdf"
    # files = list(filter(lambda filename: filename.endswith(".pdf"), os.listdir(blank_fpath)))
    # make_checkbox_template(random.choice(files))
    # make_checkbox_template(filename)
    # extract_data("alaska_FILLED.pdf", "alaska-table.json")
    make_table_template("exemption_filled.pdf", table_type="uniform")
    
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