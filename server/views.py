import os
from server import flask_app as app
from server.settings import UPLOAD_FOLDER, OUTPUT_FOLDER
from flask import request, Response, jsonify, render_template
import json

from services import checkbox_detect, table_analysis, util, template_extract
from base.customEncoder import CustomEncoder

output_fpath = 'output/'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    action = request.form['actionType']
    document = request.files['document']
    filename = UPLOAD_FOLDER + document.filename
    name = os.path.basename(filename).split('.')[0]
    document.save(filename)
    print("The filename: ", filename)
    im_paths = util.pdf_to_image(filename)

    if action == 'checkbox':
        clusters, img = checkbox_detect.checkbox_detect(im_paths[0], plot=False, fileout=OUTPUT_FOLDER + name)
        image_path = '/static/outputs/' + os.path.basename(img)
        response = {
            'clusters': clusters,
            'image': image_path
        }
    elif action == 'uniform_table':
        result = table_analysis.check_table(im_paths[0]) #check for uniform table
        csv_fname = name + "_uniform.csv"
        template_fname = name + "_uniform.json"
        if len(result) > 0:
            response = table_analysis.read_tables(im_paths[0], result[0], result[1], result[2], fpath=output_fpath, #+ 'table/',
                                                                                         csv_name=csv_fname, template_name=template_fname)
        else:
            response = []
    elif action == 'non_uniform_table':
        response = table_analysis.get_horizontal_lines(im_paths[0])
    else:
        template = request.files['template']
        template_filename = UPLOAD_FOLDER + template.filename
        template.save(template_filename)
        
        name = os.path.basename(filename).split('.')[0]
        output_file = output_fpath + name + '.json'
        response = template_extract.extract_template(im_paths[0], filename, template_filename, output_fpath, output_file)

    return json.dumps(response, cls=CustomEncoder)