import os

filename = 'labelmap_voc.prototxt'

label_ids = []
label_txt = []
with open(filename) as f:
	for line in f:
		ss = [i.strip().strip('\"') for i in line.split(':')]
		if ss[0]=='label':
			label_ids.append(ss[1])
		if ss[0]=='display_name':
			label_txt.append(ss[1])

d = dict(zip(label_ids,label_txt))

with open('labels.txt','w') as f:
	for i in label_ids:
		f.write('%s,%s,%s\n'%(i,i,d[i]))

