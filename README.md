# DocProcessing

Reads uniform and non-uniform tables as well as checkbox data from tax exemption forms.
See demo_run for example driver code (update file paths as needed).
For uniform tables use: read_tables
For non-uniform tables use: get_horizontal_lines

## Setup
Install all the python packages required by running `pip install -r requirements.txt`.

Install tesseract library. This blog might be helpful: https://www.pyimagesearch.com/2017/07/03/installing-tesseract-for-ocr/.

You might also need to install `poppler`. It helps render the PDFs. Instructions can be found in the description of pdf2image here: https://pypi.org/project/pdf2image/.