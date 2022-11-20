import warnings
warnings.filterwarnings('ignore')

import sys
import os
import glob
import re
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

#try:
	#import shutil
	#shutil.rmtree('uploaded\image')
	#print()
#except:
	#pass

model = load_model('resnet_model.h5')

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploaded\image'

@app.route('/')
def upload_f():
	return render_template('upload.html')

def finds():
	image_size = (224, 224)
	batch_size = 64
	seed_number = 123

	gen_args = dict(target_size=image_size,
					batch_size=batch_size,
					class_mode="binary",
					seed=seed_number)
	test_generator = ImageDataGenerator(rescale=1. / 255)
	test_dataset = test_generator.flow_from_directory(directory="uploaded",
													  shuffle=False,
													  **gen_args)

	pred = model.predict(test_dataset)
	
	if pred[0][0] > 0.5:
		return 'Tyre is predicted to be Cracked'
	else:
		return 'Tyre is predicted to be Normal'

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		f = request.files['file']
		f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
		val = finds()
		return render_template('pred.html', ss = val)

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8080)
