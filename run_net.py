import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import warnings
import datetime
import sys
import numpy as np
import imageio
from PIL import Image
import onnx
import onnx_tf

modelPath = (sys.argv[1])
#rep = backend.prepare(model, 'CPU')
#img = imageio.imread('/var/www/html/Secondary/trashnet/data/dataset-resized/trash/trash1.jpg', pilmode='RGB')
onnx_model = onnx.load(modelPath)
prep = onnx_tf.backend.prepare(onnx_model)
switcher={
            0:'Cardboard: Recycle or Blue Paper Bin: ',
            1:'Glass: Recycle',
            2:'Metal: Recycle',
            3:'Paper: Recycle or Blue Paper Bin',
            4:'Plastic: Recycle',
            5:'Trash: Trash',
    }
os.system("clear")
f = open('trashData','a')
i = 0
while(1):
    time = str(datetime.datetime.now().time())
    os.system("fswebcam --device V4L2:/dev/video0 -r 1280x720 -i 0 --no-banner --quiet "+time)
    img = imageio.imread(time, pilmode='RGB')
    img = np.array(Image.fromarray(img).resize((224, 224))).astype(np.float32)
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    probs = (list(list(list(prep.run(img))[0])[0]))
    probdata = str(probs)
    max_prob = max(probs)
    binMode = ""
    if (max_prob<0.6)!=True:
        binMode = str(switcher.get(probs.index(max_prob)))
        print(binMode)
    f.write(probdata+": "+time+": "+str(max_prob)+": "+binMode+"\n")
    i+=1
    if(i>=10):
        os.system("gcloud compute scp trashData --zone us-west2-a testgreen:~/trashData&")
        i=0


