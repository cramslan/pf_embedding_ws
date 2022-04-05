
import logging
from flask import Flask, request, abort
from rfiEmbeddingGen import RFICalculator
from io import StringIO


app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

@app.route('/findembedding/', methods=['GET', 'POST'])
def flask_find_outliers():
    if request.method == 'POST':
        obj = request.form
    else:
        obj = request.args

    tsv = obj.get('tsv')
    if tsv is None:
        tsv = request.files.get('tsv').read().decode('UTF-8')

    numDesc = obj.get('NumDesc')
    if numDesc is None:
        numDesc = 12

    instance = RFICalculator()
    jsonobj = instance.findEmbedding(StringIO(tsv), numDesc)

    return jsonobj


if __name__ == '__main__':
    # app.run(host='localhost', port=5000, debug=True)
    app.run(host='0.0.0.0', port=9092, debug=True)
