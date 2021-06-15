import os
import json
from pdf2image import convert_from_path

def edit_json(jsonFile, newData):
    try:
        with open(jsonFile, "r+") as file:
            data = json.load(file)
            data.update(newData)
            file.seek(0)
            json.dump(data, file)

    except:
        with open(jsonFile, "w") as file:
            json.dump(newData, file)


def pdf_to_image(pdf_path):
    print('the pdf path', pdf_path)
    images = convert_from_path(pdf_path)
    filename = os.path.basename(pdf_path).split('.')[0]
    paths = []
    for i in range(len(images)):
        img_name = 'img/filename_' + str(i) + '.jpg'
        images[i].save(img_name, 'JPEG')
        paths.append(img_name)
    return paths