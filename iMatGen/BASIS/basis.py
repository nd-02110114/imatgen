#!/usr/bin/env python
import os
import sys
import pickle

import numpy as np
import tensorflow as tf

from tqdm import *
from utils import *

'''
Global Parameters
'''
n_epochs   = 125*400
n_ae_epochs= 20+1
batch_size = 24
g_lr       = 0.0025
d_lr       = 0.0001
ae_lr      = 0.0003
beta1      = 0.5
d_thresh   = 0.8 
z_size     = 200
leak_value = 0.2
cube_len   = 64
obj_ratio  = 0.5
reg_l2     = 0.0e-6
gan_inter  = 100
ae_inter   = 5
maxbatch   = 915
trbatch    = int(915*0.9)

batch_directory = './../vo-1st-batch-single/'
train_sample_directory = './ae_test_sample/'
model_directory = './ae_models/'
is_local = False

weights = {}

def generator(z, batch_size=batch_size, phase_train=True, reuse=False):

	strides = [1,2,2,2,1]
	with tf.variable_scope("gen",reuse=reuse):
		z = tf.reshape(z,(batch_size,1,1,1,z_size))
		g_1 = tf.nn.conv3d_transpose(z, weights['wg1'], (batch_size,4,4,4,64), strides=[1,1,1,1,1], padding="VALID")
		g_1 = lrelu(g_1)

		g_2 = tf.nn.conv3d_transpose(g_1, weights['wg2'], (batch_size,8,8,8,64), strides=strides, padding="SAME")
		g_2 = lrelu(g_2)

		g_3 = tf.nn.conv3d_transpose(g_2, weights['wg3'], (batch_size,16,16,16,64), strides=strides, padding="SAME")
		g_3 = lrelu(g_3)

		g_4 = tf.nn.conv3d_transpose(g_3, weights['wg4'], (batch_size,32,32,32,64), strides=[1,2,2,2,1], padding="SAME")
		g_4 = lrelu(g_4)

		g_5 = tf.nn.conv3d_transpose(g_4, weights['wg5'], (batch_size,64,64,64,1), strides=[1,2,2,2,1], padding="SAME")
		g_5 = tf.nn.sigmoid(g_5)

		return g_5

def encoder(inputs, phase_train=True, reuse=False):

	strides = [1,2,2,2,1]
	with tf.variable_scope("enc",reuse=reuse):
		d_1 = tf.nn.conv3d(inputs, weights['wae1'], strides=[1,2,2,2,1], padding="SAME")
		d_1 = lrelu(d_1, leak_value)

		d_2 = tf.nn.conv3d(d_1, weights['wae2'], strides=strides, padding="SAME") 
		d_2 = lrelu(d_2, leak_value)
        
		d_3 = tf.nn.conv3d(d_2, weights['wae3'], strides=strides, padding="SAME")  
		d_3 = lrelu(d_3, leak_value) 

		d_4 = tf.nn.conv3d(d_3, weights['wae4'], strides=[1,2,2,2,1], padding="SAME")     
		d_4 = lrelu(d_4)

		d_5 = tf.nn.conv3d(d_4, weights['wae5'], strides=[1,1,1,1,1], padding="VALID")
		d_5 = tf.nn.tanh(d_5)

		return d_5

def initialiseWeights():

	global weights
	xavier_init = tf.contrib.layers.xavier_initializer()

	weights['wg1'] = tf.get_variable("wg1", shape=[4, 4, 4, 64, z_size], initializer=xavier_init)
	weights['wg2'] = tf.get_variable("wg2", shape=[4, 4, 4, 64, 64], initializer=xavier_init)
	weights['wg3'] = tf.get_variable("wg3", shape=[4, 4, 4, 64, 64], initializer=xavier_init)
	weights['wg4'] = tf.get_variable("wg4", shape=[4, 4, 4, 64, 64], initializer=xavier_init)
	weights['wg5'] = tf.get_variable("wg5", shape=[4, 4, 4, 1, 64], initializer=xavier_init)

	weights['wae1'] = tf.get_variable("wae1", shape=[4, 4, 4, 1, 64], initializer=xavier_init)
	weights['wae2'] = tf.get_variable("wae2", shape=[4, 4, 4, 64, 64], initializer=xavier_init)
	weights['wae3'] = tf.get_variable("wae3", shape=[4, 4, 4, 64, 64], initializer=xavier_init)
	weights['wae4'] = tf.get_variable("wae4", shape=[4, 4, 4, 64, 64], initializer=xavier_init)
	weights['wae5'] = tf.get_variable("wae5", shape=[4, 4, 4, 64, z_size], initializer=xavier_init)    

	return weights

def trainGAN(loadmodel,model_checkpoint):

	weights = initialiseWeights()
	x_vector = tf.placeholder(shape=[batch_size,cube_len,cube_len,cube_len,1],dtype=tf.float32)
	z_vector = tf.placeholder(shape=[batch_size,1,1,1,z_size],dtype=tf.float32) 

  # Weights for autoencoder pretraining
	xavier_init = tf.contrib.layers.xavier_initializer()
	with tf.variable_scope('encoders') as scope1:
		encoded = encoder(x_vector, phase_train=True, reuse=False)
		scope1.reuse_variables()
		encoded2 = encoder(x_vector, phase_train=False, reuse=True)

	with tf.variable_scope('gen_from_dec') as scope2:
		decoded = generator(encoded, phase_train=True, reuse=False)
		scope2.reuse_variables()
		decoded_test = generator(encoded2,phase_train=False, reuse=True)

	# Round decoder output
	decoded = threshold(decoded)
	decoded_test = threshold(decoded_test)
	# Compute MSE Loss and L2 Loss
	mse_loss = tf.reduce_mean(tf.pow(x_vector - decoded, 2))
	mse_loss2 = tf.reduce_mean(tf.pow(x_vector - decoded_test, 2))
	para_ae = [var for var in tf.trainable_variables() if any(x in var.name for x in ['wae','wg'])]
	l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in para_ae])
	ae_loss = mse_loss + reg_l2 * l2_loss
	optimizer_ae = tf.train.AdamOptimizer(learning_rate=ae_lr, name="Adam_AE").minimize(ae_loss,var_list=para_ae)

	saver = tf.train.Saver(max_to_keep=50) 

	with tf.Session() as sess:  
		sess.run(tf.global_variables_initializer())        
		if loadmodel == True:
			saver.restore(sess, model_checkpoint)
		np.random.seed(333)
		fname = [str(n) for n in range(maxbatch)]
		tr_name = fname
		np.random.shuffle(fname)
		tr_name = fname[:trbatch]
		test_name = fname[trbatch:]

		np.random.shuffle(tr_name)
		ee = []
		for epoch in range(n_ae_epochs):
			mse_tr = 0; mse_test = 0;
			for idx_tr in tqdm(range(len(tr_name))):
				[p,q,r,s,t] = np.load(batch_directory+tr_name[idx_tr]+'_images.npy',mmap_mode = 'r')
				u = np.load(batch_directory+tr_name[idx_tr]+'_pvals.npy',mmap_mode = 'r')
				p = list(p); q = list(q); r = list(r); s = list(s); t = list(t);
				x = np.zeros((batch_size,64,64,64,1))
				x[p,q,r,s,t] = u
				mse_l, _ = sess.run([mse_loss, optimizer_ae],feed_dict={x_vector:x})
				mse_tr += mse_l

			for idx_t in tqdm(range(len(test_name))):
				[p,q,r,s,t] = np.load(batch_directory+test_name[idx_t]+'_images.npy',mmap_mode = 'r')
				u = np.load(batch_directory+test_name[idx_t]+'_pvals.npy',mmap_mode = 'r')
				p = list(p); q = list(q); r = list(r); s = list(s); t = list(t);
				x_test = np.zeros((batch_size,64,64,64,1))
				x_test[p,q,r,s,t] = u
				mse_t = sess.run(mse_loss2,feed_dict={x_vector:x_test})
				mse_test += mse_t

			print epoch,' ',mse_tr/len(tr_name),' ',mse_test/len(test_name)

			ee.append([epoch,mse_tr/len(tr_name),mse_test/len(test_name)])

			np.random.shuffle(tr_name)

			if epoch % ae_inter == 0 and epoch != 0:
				dec = sess.run(decoded_test,feed_dict={x_vector:x_test})
				if not os.path.exists(train_sample_directory):
					os.makedirs(train_sample_directory)
				dec.dump(train_sample_directory+'/test_'+str(epoch)+'_'+str(test_name[-1]))

				if not os.path.exists(model_directory):
					os.makedirs(model_directory)      
				saver.save(sess, save_path = model_directory + '/' + str(epoch) + '_ae.ckpt')

if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[1]
	loadmodel = bool(int(sys.argv[2]))
	model_checkpoint = sys.argv[3]
	trainGAN(loadmodel,model_checkpoint)
