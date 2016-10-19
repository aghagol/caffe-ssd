import numpy as np
import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
import os, sys, time
os.environ['GLOG_minloglevel'] = '2'
sys.path.append('/home/mo/github/ssd/python')
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()
from google.protobuf import text_format
from caffe.proto import caffe_pb2

# load PASCAL VOC labels
labelmap_file = '/home/mo/github/ssd/data/VOC0712/labelmap_voc.prototxt'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

model_pre = '/home/mo/github/ssd/models/VGGNet/VOC0712/SSD_300x300/'
model_def = 'deploy.prototxt'
model_wts = 'VGG_VOC0712_SSD_300x300_iter_60000.caffemodel'

t0 = time.time()
print 'loading the net'
net = caffe.Net(model_pre+model_def,      # defines the structure of the model
                model_pre+model_wts,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
print 'done (took %f seconds)' % (time.time()-t0)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

image_resize = 300
net.blobs['data'].reshape(1,3,image_resize,image_resize)

# image_path = '/home/mo/Desktop/ROS_data/jaguar/2016-07-22/train/left/'
# image_path = '/home/mo/Desktop/ROS_data/jaguar/2016-09-16/train/left/'
image_path = '/home/mo/Desktop/security_cam/SLR/MVI_0886/'
image_list = [float(i[:-4]) for i in os.listdir(image_path) if i.endswith('jpg')]
image_list.sort()
# image_list = np.loadtxt(open('/home/mo/Desktop/ROS_data/jaguar/2016-07-22/dataset_train_nn.txt'))[:,0]

for image_idx, image_time in enumerate(image_list):
	# if image_idx%10: continue

	image = caffe.io.load_image(image_path+'%010d.jpg'%image_time)
	plt.imshow(image)

	transformed_image = transformer.preprocess('data', image)
	net.blobs['data'].data[...] = transformed_image

	# t0 = time.time()
	detections = net.forward()['detection_out']
	# print 'forward took %f seconds' % (time.time()-t0)

	det_label = detections[0,0,:,1]
	det_conf = detections[0,0,:,2]
	det_xmin = detections[0,0,:,3]
	det_ymin = detections[0,0,:,4]
	det_xmax = detections[0,0,:,5]
	det_ymax = detections[0,0,:,6]

	# Get detections with confidence higher than 0.6.
	top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.1]

	top_conf = det_conf[top_indices]
	top_label_indices = det_label[top_indices].tolist()
	top_labels = get_labelname(labelmap, top_label_indices)
	top_xmin = det_xmin[top_indices]
	top_ymin = det_ymin[top_indices]
	top_xmax = det_xmax[top_indices]
	top_ymax = det_ymax[top_indices]

	colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

	plt.imshow(image)
	currentAxis = plt.gca()

	for i in xrange(top_conf.shape[0]):
	    xmin = int(round(top_xmin[i] * image.shape[1]))
	    ymin = int(round(top_ymin[i] * image.shape[0]))
	    xmax = int(round(top_xmax[i] * image.shape[1]))
	    ymax = int(round(top_ymax[i] * image.shape[0]))
	    score = top_conf[i]
	    label = int(top_label_indices[i])
	    label_name = top_labels[i]
	    display_txt = '%s: %.2f'%(label_name, score)
	    coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
	    color = colors[label]
	    currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
	    currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})

	# plt.waitforbuttonpress(timeout=-1)
	plt.savefig('out/%010d.png'%image_idx)

	print 'done with %f (%.3f%%)' %(image_time,float(image_idx)/len(image_list)*100)
	plt.clf()