import os
from server import flask_app as app
from server.settings import UPLOAD_FOLDER, OUTPUT_FOLDER
from flask import request, Response, jsonify, render_template
import json

from services import checkbox_detect, table_analysis, util, template_extract
from base.customEncoder import CustomEncoder

output_fpath = '/static/outputs/'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    action = request.form['actionType']
    page = int(request.form['page'])
    print('page number: ', page)
    document = request.files['document']
    filename = UPLOAD_FOLDER + document.filename
    name = os.path.basename(filename).split('.')[0]
    document.save(filename)
    print("The filename: ", filename)
    im_paths = util.pdf_to_image(filename)
    image_path = im_paths[page-1]

    if action == 'checkbox':
        clusters, img = checkbox_detect.checkbox_detect(image_path, plot=False, fileout=OUTPUT_FOLDER + name)
        clusters = [{'type': 'checkboxes', 'data': cluster} for cluster in clusters]
        image_path = output_fpath + os.path.basename(img)
        response = {
            'clusters': clusters,
            'image': image_path
        }
    elif action == 'uniform_table':
        img_fname = name + "_uniform.jpg"
        result = table_analysis.extract_tables(image_path, out_img=OUTPUT_FOLDER + img_fname) #check for uniform table
        # result = table_analysis.return_table(image_path, outfile=OUTPUT_FOLDER + img_fname) #check for uniform table
        # csv_fname = name + "_uniform.csv"
        # template_fname = name + "_uniform.json"
        # if len(result) > 0:
        #     data = table_analysis.read_tables(image_path, result[0], result[1], fpath=OUTPUT_FOLDER, #+ 'table/',
        #                                                                                  csv_name=csv_fname, template_name=template_fname)
        #     response = [{
        #         'type': 'uniform_table',
        #         'data': data
        #     }]
        # else:
        #     response = []

        response = []
        for table in result:
            if len(table) > 0:
                response.append({
                    'type': 'uniform_table',
                    'data': table
                })
        response = {
            'clusters': response,
            'image': output_fpath + img_fname
        }
    elif action == 'non_uniform_table':
        response = table_analysis.get_horizontal_lines(image_path)
    else:
        template = request.files['template']
        template_filename = UPLOAD_FOLDER + template.filename
        template.save(template_filename)
        
        name = os.path.basename(filename).split('.')[0]
        output_file = output_fpath + name + '.json'
        response = template_extract.extract_template(image_path, filename, template_filename, output_fpath, output_file)

    return json.dumps(response, cls=CustomEncoder)