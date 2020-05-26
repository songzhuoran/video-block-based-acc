from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import cv2
import numpy as np
from keras.models import *
from keras.layers import *
from keras.models import Model
from types import MethodType
import six
import json
import glob
from tqdm import tqdm
import random
import itertools

import re

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.18
set_session(tf.Session(config=config))

IPframe_SRC_DIR = '/home/wufeiyang/Proj/quan_video/test/full_version/Data/DAVIS2016/Annotations/480p/'
IPframe_IDX_DIR = '../data/idx/p/'
FAVOS_DIR = '/home/wufeiyang/Proj/quan_video/FAVOS/results/favos_add_Sseg_CRF_Tracker/'


IMAGE_ORDERING = 'channels_last'
# MERGE_AXIS = 1
if IMAGE_ORDERING == 'channels_first':
    MERGE_AXIS = 1
elif IMAGE_ORDERING == 'channels_last':
	MERGE_AXIS = -1

def get_segmentation_model( input , output ):
    
	img_input = input
	o = output

	o_shape = Model(img_input , o ).output_shape
	i_shape = Model(img_input , o ).input_shape

	if IMAGE_ORDERING == 'channels_first':
		output_height = o_shape[2]
		output_width = o_shape[3]
		input_height = i_shape[2]
		input_width = i_shape[3]
		n_classes = o_shape[1]
		o = (Reshape((  -1  , output_height*output_width   )))(o)
		o = (Permute((2, 1)))(o)
	elif IMAGE_ORDERING == 'channels_last':
		output_height = o_shape[1]
		output_width = o_shape[2]
		input_height = i_shape[1]
		input_width = i_shape[2]
		n_classes = o_shape[3]
		o = (Reshape((   output_height*output_width , -1    )))(o)

	o = (Activation('softmax'))(o)
	model = Model( img_input , o )
	model.output_width = output_width
	model.output_height = output_height
	model.n_classes = n_classes
	model.input_height = input_height
	model.input_width = input_width
	model.model_name = ""

	model.train = MethodType( train , model )
	model.predict_segmentation = MethodType( predict , model )
	model.predict_multiple = MethodType( predict_multiple , model )
	model.evaluate_segmentation = MethodType( evaluate , model )

	return model 

def unet_mini( n_classes , input_height=360, input_width=480   ):

	if IMAGE_ORDERING == 'channels_first':
		img_input = Input(shape=(3,input_height,input_width)) # Todo: change channels
	elif IMAGE_ORDERING == 'channels_last':
		img_input = Input(shape=(input_height,input_width , 3 )) # Todo: change channels

	conv1 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING, activation='relu', padding='same')(img_input)
	conv1 = Dropout(0.2)(conv1)
	#conv1 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING, activation='relu', padding='same')(conv1)
	pool1 = MaxPooling2D((2, 2), data_format=IMAGE_ORDERING)(conv1)
	
	conv2 = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING, activation='relu', padding='same')(pool1)
	conv2 = Dropout(0.2)(conv2)
	#conv2 = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING, activation='relu', padding='same')(conv2)

	up1 = concatenate([UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(conv2), conv1], axis=MERGE_AXIS)
	conv3 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING, activation='relu', padding='same')(up1)
	conv3 = Dropout(0.2)(conv3)
	#conv3 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING, activation='relu', padding='same')(conv3)
	
	#songzhuoran:need to see the output of label
	o = Conv2D( n_classes, (1, 1) , data_format=IMAGE_ORDERING ,padding='same')(conv3)

	model = get_segmentation_model(img_input , o )
	model.model_name = "unet_mini"
	return model

def get_pairs_from_paths( images_path , segs_path ):
	images = glob.glob( os.path.join(images_path,"*.jpg")  ) + glob.glob( os.path.join(images_path,"*.png")  ) +  glob.glob( os.path.join(images_path,"*.jpeg")  )
	segmentations  =  glob.glob( os.path.join(segs_path,"*.png")  ) 

	segmentations_d = dict( zip(segmentations,segmentations ))

	ret = []

	for im in images:
		seg_bnme = os.path.basename(im).replace(".jpg" , ".png").replace(".jpeg" , ".png")
		seg = os.path.join( segs_path , seg_bnme  )
		assert ( seg in segmentations_d ),  (im + " is present in "+images_path +" but "+seg_bnme+" is not found in "+segs_path + " . Make sure annotation image are in .png"  )
		ret.append((im , seg) )

	return ret
	
def get_image_arr( path, Iframe, Pframe , width , height , imgNorm="sub_mean" , odering='channels_last' ):
    

	if type( path ) is np.ndarray:
		img = path
	else:
		img = cv2.imread(path, 1)

	img = cv2.resize(img, ( width , height ))
	Iframe = cv2.resize(Iframe, ( width , height ))
	Pframe = cv2.resize(Pframe, ( width , height ))
	img[:,:,1] = Iframe[:,:,0]
	img[:,:,2] = Pframe[:,:,0]

	if odering == 'channels_first':
		img = np.rollaxis(img, 2, 0)
	return img

def get_segmentation_arr( path , nClasses ,  width , height , no_reshape=False ):
    
	seg_labels = np.zeros((  height , width  , nClasses ))
		
	if type( path ) is np.ndarray:
		img = path
	else:
		img = cv2.imread(path, 1)

	img = cv2.resize(img, ( width , height ) , interpolation=cv2.INTER_NEAREST )
	img = img[:, : , 0]

	# Modified by jzm
	lutClassLabel = {
		0 : 0,
		1 : 255
	}
	for c in range(nClasses):
		seg_labels[: , : , c ] = (img == lutClassLabel[c] ).astype(int)

	# print(seg_labels)
	
	if no_reshape:
		return seg_labels

	seg_labels = np.reshape(seg_labels, ( width*height , nClasses ))
	return seg_labels

def verify_segmentation_dataset( images_path , segs_path , n_classes ):
    	
	img_seg_pairs = get_pairs_from_paths( images_path , segs_path )

	assert len(img_seg_pairs)>0 , "Dataset looks empty or path is wrong "
	
	for im_fn , seg_fn in tqdm(img_seg_pairs) :
		img = cv2.imread( im_fn )
		seg = cv2.imread( seg_fn )

		assert ( img.shape[0]==seg.shape[0] and img.shape[1]==seg.shape[1] ) , "The size of image and the annotation does not match or they are corrupt "+ im_fn + " " + seg_fn
		#add by songzhuoran
		# assert ( np.max(seg[:,:,0]) < n_classes) , "The pixel values of seg image should be from 0 to "+str(n_classes-1) + " . Found pixel value "+str(np.max(seg[:,:,0]))

	print("Dataset verified! ")

def image_segmentation_generator( images_path , segs_path ,  batch_size,  n_classes , input_height , input_width , output_height , output_width  , do_augment=False ):
    	

	img_seg_pairs = get_pairs_from_paths( images_path , segs_path )
	random.shuffle( img_seg_pairs )
	zipped = itertools.cycle( img_seg_pairs  )

	while True:
		X = []
		Y = []
		for _ in range( batch_size) :
			im , seg = next(zipped) 
			# print("*************************************************************")
			# print(im)
			# Todo: Find I and P frame for B.

			ptn_name=re.compile(r'bframe/(.*?)_')
			ptn_idx=re.compile(r'_0(.*?).png')
			print(im)
			Bframe_filename=ptn_name.findall(im)[0]
			Bframe_fileidx=ptn_idx.findall(im)[0]
			Bframe_fileidx=int(Bframe_fileidx.lstrip('0'))

			idx_i = 0
			idx_p = 0
			# print(Bframe_filename)
			# print(Bframe_fileidx)
			# exit()
			p_idx_list = []
			with open(IPframe_IDX_DIR+Bframe_filename) as idx_file:
				for line in idx_file:
					p_idx_list.append(int(line)-1)
			
			for i in range(0, len(p_idx_list)):
				if p_idx_list[i] > Bframe_fileidx:
					idx_i = p_idx_list[i-1]
					idx_p = p_idx_list[i]
					break


			im_i = cv2.imread(IPframe_SRC_DIR + Bframe_filename + '/%05d.png' % idx_i , 1 )
				# cv2.imwrite('tmp/izhen.png', im_i)

			im_p = cv2.imread(IPframe_SRC_DIR + Bframe_filename + '/%05d.png' % idx_p , 1 )
				# cv2.imwrite('tmp/pframe.png', im_p)

			im = cv2.imread(im , 1 )
			seg = cv2.imread(seg , 1 )

			if do_augment:
				im , seg[:,:,0] = augment_seg( im , seg[:,:,0] )

			X.append( get_image_arr(im, im_i, im_p, input_width , input_height ,odering=IMAGE_ORDERING)  )
			Y.append( get_segmentation_arr( seg , n_classes , output_width , output_height )  )
			
			# wfy
			# exit()

		yield np.array(X) , np.array(Y)

def train( model  , 
		train_images  , 
		train_annotations , 
		input_height=None , 
		input_width=None , 
		n_classes=None,
		verify_dataset=True,
		checkpoints_path=None , 
		epochs = 5,
		batch_size = 2,
		validate=False , 
		val_images=None , 
		val_annotations=None ,
		val_batch_size=2 , 
		auto_resume_checkpoint=False ,
		load_weights=None ,
		steps_per_epoch=512,
		optimizer_name='adadelta' 
	):


	# if  isinstance(model, six.string_types) : # check if user gives model name insteead of the model object
	# 	# create the model from the name
	# 	assert ( not n_classes is None ) , "Please provide the n_classes"
	# 	if (not input_height is None ) and ( not input_width is None):
	# 		model = model_from_name[ model ](  n_classes , input_height=input_height , input_width=input_width )
	# 	else:
	# 		model = model_from_name[ model ](  n_classes )

	n_classes = model.n_classes
	input_height = model.input_height
	input_width = model.input_width
	output_height = model.output_height
	output_width = model.output_width


	if validate:
		assert not (  val_images is None ) 
		assert not (  val_annotations is None ) 

	if not optimizer_name is None:
		model.compile(loss='categorical_crossentropy',
			optimizer= optimizer_name ,
			metrics=['accuracy'])

	if not checkpoints_path is None:
		open( checkpoints_path+"_config.json" , "w" ).write( json.dumps( {
			"model_class" : model.model_name ,
			"n_classes" : n_classes ,
			"input_height" : input_height ,
			"input_width" : input_width ,
			"output_height" : output_height ,
			"output_width" : output_width 
		}))

	if ( not (load_weights is None )) and  len( load_weights ) > 0:
		print("Loading weights from " , load_weights )
		model.load_weights(load_weights)

	if auto_resume_checkpoint and ( not checkpoints_path is None ):
		latest_checkpoint = find_latest_checkpoint( checkpoints_path )
		if not latest_checkpoint is None:
			print("Loading the weights from latest checkpoint "  ,latest_checkpoint )
			model.load_weights( latest_checkpoint )


	if verify_dataset:
		print("Verifying train dataset")
		verify_segmentation_dataset( train_images , train_annotations , n_classes )
		if validate:
			print("Verifying val dataset")
			verify_segmentation_dataset( val_images , val_annotations , n_classes )


	train_gen = image_segmentation_generator( train_images , train_annotations ,  batch_size,  n_classes , input_height , input_width , output_height , output_width   )


	if validate:
		val_gen  = image_segmentation_generator( val_images , val_annotations ,  val_batch_size,  n_classes , input_height , input_width , output_height , output_width   )


	if not validate:
		for ep in range( epochs ):
			print("Starting Epoch " , ep )
			model.fit_generator( train_gen , steps_per_epoch  , epochs=1 )
			if not checkpoints_path is None:
				model.save_weights( checkpoints_path + "." + str( ep ) )
				print("saved " , checkpoints_path + ".model." + str( ep ) )
			print("Finished Epoch" , ep )
	else:
		for ep in range( epochs ):
			print("Starting Epoch " , ep )
			model.fit_generator( train_gen , steps_per_epoch  , validation_data=val_gen , validation_steps=200 ,  epochs=1 )
			if not checkpoints_path is None:
				model.save_weights( checkpoints_path + "." + str( ep )  )
				print("saved " , checkpoints_path + ".model." + str( ep ) )
			print("Finished Epoch" , ep )

# 学姐Update
def find_latest_checkpoint( checkpoints_path ):
	ep = 0
	r = None
	while True:
		if os.path.isfile( checkpoints_path + "." + str( ep )  ):
			r = checkpoints_path + "." + str( ep ) 
		else:
			return r 

		ep += 1

def model_from_checkpoint_path( checkpoints_path ):
	assert ( os.path.isfile(checkpoints_path+"_config.json" ) ) , "Checkpoint not found."
	model_config = json.loads(open(  checkpoints_path+"_config.json" , "r" ).read())
	latest_weights = find_latest_checkpoint( checkpoints_path )
	assert ( not latest_weights is None ) , "Checkpoint not found."
	# model = model_from_name[ model_config['model_class']  ]( model_config['n_classes'] , input_height=model_config['input_height'] , input_width=model_config['input_width'] )
	print("loaded weights " , latest_weights )
	model.load_weights(latest_weights)
	return model

# 学姐Update End

def predict( model=None , inp=None , out_fname=None , checkpoints_path=None  ):
    
	# # if model is None and ( not checkpoints_path is None ):
	# 	# model = model_from_checkpoint_path(checkpoints_path)
	# model = model_from_checkpoint_path(checkpoints_path)
	# assert ( not inp is None )
	# assert( (type(inp) is np.ndarray ) or  isinstance( inp , six.string_types)  ) , "Inupt should be the CV image or the input file name"
	# print(inp)
	# ptn_name=re.compile(r'bframe/(.*?)/')
	# ptn_idx=re.compile(r'/0(.*?).png')
	# Bframe_filename=ptn_name.findall(inp)[0]
	# Bframe_fileidx=ptn_idx.findall(inp)[0]
	# Bframe_fileidx=int(Bframe_fileidx.lstrip('0'))
	
	# idx_i = 0
	# idx_p = 0

	# p_idx_list = []
	# with open(IPframe_IDX_DIR+Bframe_filename) as idx_file:
	# 	for line in idx_file:
	# 		p_idx_list.append(int(line)-1)
	
	# for i in range(0, len(p_idx_list)):
	# 	if p_idx_list[i] > Bframe_fileidx:
	# 		idx_i = p_idx_list[i-1]
	# 		idx_p = p_idx_list[i]
	# 		break

	# print("Index of I frame is: "+str(idx_i))
	# print("Index of P frame is: "+str(idx_p))

	im_i = cv2.imread('test/i.png' , 1 )
	im_p = cv2.imread('test/p.png' , 1 )

	inp = cv2.imread('test/b.png' , 1 )
	

	output_width = model.output_width
	output_height  = model.output_height
	input_width = model.input_width
	input_height = model.input_height
	n_classes = model.n_classes

	x = get_image_arr( inp, im_i, im_p , input_width  , input_height , odering=IMAGE_ORDERING )
	pr = model.predict( np.array([x]) )[0]
	pr = pr.reshape(( output_height ,  output_width , n_classes ) ).argmax( axis=2 )

	seg_img = np.zeros( ( output_height , output_width , 3  ) )
	
	
	# Modified by jzm

	# class_colors = [  ( random.randint(0,255),random.randint(0,255),random.randint(0,255)   ) for _ in range(5000)  ]
	# colors = class_colors
	colors = [(0, 0, 0), (255, 255, 255)]

	for c in range(n_classes):
		seg_img[:,:,0] += ( (pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
		seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
		seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')
	seg_img = cv2.resize(seg_img  , (input_width , input_height ))

	if not out_fname is None:
		cv2.imwrite(  out_fname , seg_img )


	return pr


def predict_multiple( model=None , inps=None , inp_dir=None, out_dir=None , checkpoints_path=None  ):

	if model is None and ( not checkpoints_path is None ):
		model = model_from_checkpoint_path(checkpoints_path)


	if inps is None and ( not inp_dir is None ):
		inps = glob.glob( os.path.join(inp_dir,"*.jpg")  ) + glob.glob( os.path.join(inp_dir,"*.png")  ) +  glob.glob( os.path.join(inp_dir,"*.jpeg")  )

	assert type(inps) is list
	
	all_prs = []

	for i , inp in enumerate(tqdm(inps)):
		if out_dir is None:
			out_fname = None
		else:
			if isinstance( inp , six.string_types)  :
				out_fname = os.path.join( out_dir , os.path.basename(inp) )
			else :
				out_fname = os.path.join( out_dir , str(i)+ ".jpg" )

		pr = predict(model , inp ,out_fname  )
		all_prs.append( pr )

	return all_prs




def evaluate( model=None , inp_inmges=None , annotations=None , checkpoints_path=None ):
	
	assert False , "not implemented "

	ious = []
	for inp , ann   in tqdm( zip( inp_images , annotations )):
		pr = predict(model , inp )
		gt = get_segmentation_arr( ann , model.n_classes ,  model.output_width , model.output_height  )
		gt = gt.argmax(-1)
		iou = metrics.get_iou( gt , pr , model.n_classes )
		ious.append( iou )
	ious = np.array( ious )
	print("Class wise IoU "  ,  np.mean(ious , axis=0 ))
	print("Total  IoU "  ,  np.mean(ious ))


# 学姐让我注释掉
# model = unet_mini(n_classes=2 ,  input_height=416, input_width=608  )
# model = unet_mini(n_classes=2 ,  input_height=480, input_width=852  )
model = unet_mini(n_classes=2 ,  input_height=480, input_width=854  )

seq_name = "train"
train_images = "../data/train/bframe/"
train_annotations = "../data/train/annotation/"
# model.train( train_images,train_annotations, input_height=None , input_width=None , n_classes=None,verify_dataset=True,
# 		checkpoints_path = "model/vgg_unet_1", epochs = 2,batch_size = 2,validate=False , val_images=None , val_annotations=None ,val_batch_size=2 , 
# 		auto_resume_checkpoint=False ,load_weights=None ,steps_per_epoch=500,optimizer_name='adadelta' 
# 	)

import sys

INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]

out = model.predict_segmentation(
    inp=INPUT_DIR,
    out_fname="result.png",
    checkpoints_path="model/vgg_unet_1"
)


# import matplotlib.pyplot as plt
# plt.imshow(out)
