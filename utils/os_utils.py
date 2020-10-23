import os
import shutil
import argparse
import logging
import time
import getpass
import numpy as np
import tensorflow as tf
from termcolor import colored
from beautifultable import BeautifulTable
import pickle
import csv
from copy import deepcopy, copy

def str2bool(value):
	if isinstance(value, bool):
	   return value
	if value.lower() in ['yes', 'true', 't', 'y', '1']:
		return True
	elif value.lower() in ['no', 'false', 'f', 'n', '0']:
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def make_dir(dir_name, clear=True):
	if os.path.exists(dir_name):
		if clear:
			try: shutil.rmtree(dir_name)
			except: pass
			try: os.makedirs(dir_name)
			except: pass
	else:
		try: os.makedirs(dir_name)
		except: pass

def dir_ls(dir_path):
	dir_list = os.listdir(dir_path)
	dir_list.sort()
	return dir_list

def system_pause():
	getpass.getpass("Press Enter to Continue")

def get_arg_parser():
	return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def remove_color(key):
	for i in range(len(key)):
		if key[i]=='@':
			return key[:i]
	return key

def load_npz_info(file_path):
	return np.load(file_path)['info'][()]

class Logger:
	def __init__(self, name):
		make_dir('log',clear=False)
		make_dir('log/text',clear=False)
		if name is None: self.name = time.strftime('%Y-%m-%d-%H:%M:%S')
		else: self.name = name + time.strftime('-(%Y-%m-%d-%H:%M:%S)')

		self.my_log_dir = "log/{}/".format(name)
		log_file = self.my_log_dir + "output.log"
		self.logger = logging.getLogger(log_file)
		self.logger.setLevel(logging.DEBUG)

		make_dir(self.my_log_dir, clear=True)
		make_dir("{}temp/".format(self.my_log_dir), clear=True)
		self.csv_file_path = "{}progress.csv".format(self.my_log_dir)


		FileHandler = logging.FileHandler(log_file)
		FileHandler.setLevel(logging.DEBUG)
		self.logger.addHandler(FileHandler)

		StreamHandler = logging.StreamHandler()
		StreamHandler.setLevel(logging.INFO)
		self.logger.addHandler(StreamHandler)

		self.tabular_reset()

	def debug(self, *args): self.logger.debug(*args)
	def info(self, *args): self.logger.info(*args)  # default level
	def warning(self, *args): self.logger.warning(*args)
	def error(self, *args): self.logger.error(*args)
	def critical(self, *args): self.logger.critical(*args)

	def log_time(self, log_tag=''):
		log_info = time.strftime('%Y-%m-%d %H:%M:%S')
		if log_tag!='': log_info += ' '+log_tag
		self.info(log_info)

	def tabular_reset(self):
		self.keys = []
		self.colors = []
		self.values = {}
		self.counts = {}
		self.summary = []

	def tabular_clear(self):
		for key in self.keys:
			self.counts[key] = 0

	def summary_init(self, graph, sess):
		make_dir('log/board',clear=False)
		self.summary_writer = SummaryWriter(graph, sess, 'log/board/'+self.name)

	def summary_setup(self):
		self.summary_writer.setup()

	def summary_clear(self):
		self.summary_writer.clear()

	def summary_show(self, steps):
		self.summary_writer.show(steps)

	def check_color(self, key):
		for i in range(len(key)):
			if key[i]=='@':
				return key[:i], key[i+1:]
		return key, None

	def add_item(self, key, summary_type='none'):
		assert not(key in self.keys)
		key, color = self.check_color(key)
		self.counts[key] = 0
		self.keys.append(key)
		self.colors.append(color)
		if summary_type!='none':
			assert not(self.summary_writer is None)
			self.summary.append(key)
			self.summary_writer.add_item(key, summary_type)

	def add_record(self, key, value, count=1):
		key, _ = self.check_color(key)
		if type(value)==np.ndarray:
			count *= np.prod(value.shape)
			value = np.mean(value) # convert to scalar
		if self.counts[key]>0:
			self.values[key] += value*count
			self.counts[key] += count
		else:
			self.values[key] = value*count
			self.counts[key] = count
		if key in self.summary:
			self.summary_writer.add_record(key, value, count)

	def add_dict(self, info, prefix='', count=1):
		for key, value in info.items():
			self.add_record(prefix+key, value, count)

	def tabular_show(self, log_tag=''):
		table = BeautifulTable()
		table_c = BeautifulTable()
		for key, color in zip(self.keys, self.colors):
			if self.counts[key]==0: value = ''
			elif self.counts[key]==1: value = self.values[key]
			else: value = self.values[key]/self.counts[key]
			key_c = key if color is None else colored(key, color, attrs=['bold'])
			table.append_row([key, value])
			table_c.append_row([key_c, value])

		def customize(table):
			table.set_style(BeautifulTable.STYLE_NONE)
			table.left_border_char = '|'
			table.right_border_char = '|'
			table.column_separator_char = '|'
			table.top_border_char = '-'
			table.bottom_border_char = '-'
			table.intersect_top_left = '+'
			table.intersect_top_mid = '+'
			table.intersect_top_right = '+'
			table.intersect_bottom_left = '+'
			table.intersect_bottom_mid = '+'
			table.intersect_bottom_right = '+'
			table.column_alignments[0] = BeautifulTable.ALIGN_LEFT
			table.column_alignments[1] = BeautifulTable.ALIGN_LEFT

		customize(table)
		customize(table_c)
		self.log_time(log_tag)
		self.debug(table)
		print(table_c)

	def save_npz(self, info, info_name, folder, subfolder=''):
		make_dir('log/'+folder,clear=False)
		make_dir('log/'+folder+'/'+self.name,clear=False)
		if subfolder!='':
			make_dir('log/'+folder+'/'+self.name+'/'+subfolder,clear=False)
			save_path = 'log/'+folder+'/'+self.name+'/'+subfolder
		else:
			save_path = 'log/'+folder+'/'+self.name
		np.savez(save_path+'/'+info_name+'.npz',info=info)

	def save_agent(self, data, mode): # TODO: new
		# pickles current agent for later inspection
		if mode == "periodical":
			path = "{}agent_{}.pkl".format(self.my_log_dir, self.values["Epoch"])
			print("Saved periodic agent at epoch {} in {}".format(self.values["Epoch"], path))
		elif mode == "best":
			path = "{}agent_best.pkl".format(self.my_log_dir)
			print("Saved new best agent in {}".format(path))
		elif mode == "latest":
			path = "{}agent_latest.pkl".format(self.my_log_dir)
			print("Saved latest agent in {}".format(path))
		else:
			print("save_agent: no mode specified!")
		with open(path, 'wb') as f:
			pickle.dump(data, f)

	def save_csv(self):
		if (not os.path.isfile(self.csv_file_path)) or os.stat(self.csv_file_path).st_size == 0:
			with open(self.csv_file_path, 'w') as csv_file:
				self.writer = csv.DictWriter(csv_file, fieldnames=self.keys)
				self.writer.writeheader()
				self.writer.writerow(self.values)
		else:
			with open(self.csv_file_path, 'a') as csv_file:
				self.writer = csv.DictWriter(csv_file, fieldnames=self.keys)
				self.writer.writerow(self.values)
		return self.values

class LoggerExtra:
	def __init__(self, logger_log_dir, csv_filename):

		self.logger_log_dir = logger_log_dir
		self.csv_file_path = "{}{}.csv".format(self.logger_log_dir, csv_filename)
		self.keys = []
		self.colors = []
		self.values = {}
		self.counts = {}
		self.summary = []

	def check_color(self, key):
		for i in range(len(key)):
			if key[i]=='@':
				return key[:i], key[i+1:]
		return key, None

	def add_item(self, key):
		assert not(key in self.keys)
		key, color = self.check_color(key)
		self.counts[key] = 0
		self.keys.append(key)
		self.colors.append(color)

	def add_record(self, key, value, count=1):
		key, _ = self.check_color(key)
		if key == 'Step':
			a = 1
		#if type(value)==np.ndarray:
		#	count *= np.prod(value.shape)
		#	value = np.mean(value) # convert to scalar
		self.values[key] = value
		self.counts[key] = count

	def add_dict(self, info, prefix='', count=1):
		for key, value in info.items():
			self.add_record(prefix+key, value, count)


	def save_csv(self):
		if (not os.path.isfile(self.csv_file_path)) or os.stat(self.csv_file_path).st_size == 0:
			with open(self.csv_file_path, 'w') as csv_file:
				self.writer = csv.DictWriter(csv_file, fieldnames=self.keys)
				self.writer.writeheader()
				self.writer.writerow(self.values)
		else:
			with open(self.csv_file_path, 'a') as csv_file:
				self.writer = csv.DictWriter(csv_file, fieldnames=self.keys)
				self.writer.writerow(self.values)
		return self.values

class CSV_Logger:
	def __init__(self, fieldnames, args, iteration_fieldnames=['epoch', 'episode', 'step'], recover_filename=None,
				 test_filename=None):
		if recover_filename is not None:
			self.csv_filename = recover_filename
		else:
			if test_filename is not None:
				self.csv_file_path = args.dirpath +'csv_logs/'+\
									 time.strftime('%Y_%m_%d_%H_%M_%S') + test_filename + '.csv'
			else:
				self.csv_file_path = args.dirpath +'csv_logs/'+\
									 time.strftime('%Y_%m_%d_%H_%M_%S') + args.csv_filename + '.csv'
		self.fieldnames = fieldnames
		self.iteration_fieldnames = iteration_fieldnames
		all_fieldnames = self.iteration_fieldnames + self.fieldnames
		if (not os.path.isfile(self.csv_file_path)) or os.stat(self.csv_file_path).st_size == 0:
			with open(self.csv_file_path, 'w') as csv_file:
				self.writer = csv.DictWriter(csv_file, fieldnames=all_fieldnames)
				self.writer.writeheader()

		self.entries = {}
		self.num_entries = {}
		for k in self.fieldnames + self.iteration_fieldnames:
			self.entries[k] = []
			self.num_entries[k] = 0

	def add_log(self, keyname, val):
		if keyname not in self.fieldnames:
			raise Exception("The given keyname does not belong to the valid keynames for this logger")
		else:
			self.entries[keyname].append(copy(val))
			self.num_entries[keyname] += 1

	def finish_step_log(self, step):
		self._finish_it_log('step', step)

	def finish_episode_log(self, episode):
		self._finish_it_log('episode', episode)

	def finish_epoch_log(self, epoch):
		self._finish_it_log('epoch', epoch)
		self.write_to_csv_file()
		for k in self.fieldnames + self.iteration_fieldnames:
			self.entries[k] = []
			self.num_entries[k] = 0
	
	def _finish_it_log(self, it_keyname, it):
		#we see ehich from the entries has the maximum amount
		max_amount = -1
		for k in self.fieldnames:
			if self.num_entries[k] > max_amount:
				max_amount = self.num_entries[k]
		#the amount of logged value until this call must belong to this iteration; therefore each must have this is
		#we calculate how many
		it_diff = max_amount - self.num_entries[it_keyname]
		assert it_diff >= 0
		it_index = self.iteration_fieldnames.index(it_keyname)
		if it_index == len(self.iteration_fieldnames) - 1:
			prev_diff = it_diff
		else:
			prev_diff = copy(max_amount - self.num_entries[self.iteration_fieldnames[it_index+1]])
		if it_diff == 0:
			#No new entries
			self.entries[it_keyname].append(None)
			self.num_entries[it_keyname] += 1
			for k in self.fieldnames + self.iteration_fieldnames[it_index+1:]:
				self.entries[k].append(None)
				self.num_entries[k] +=1
		else:
			for _ in range(it_diff):
				self.entries[it_keyname].append(it)
			prev_it_amount = deepcopy(self.num_entries[it_keyname])
			self.num_entries[it_keyname] = max_amount
			#Now we are going to pad for those values that were not logged as often; should not be too much. But for example
			#we could get a loss for each iteration inside an episode and one reward per episode

			for k in self.fieldnames + self.iteration_fieldnames[it_index+1:]:
				diff = max_amount - self.num_entries[k]
				assert diff >= 0 and diff <= it_diff
				if diff == 0:
					continue
				assert prev_diff <= diff
				pad_after = [None] * prev_diff
				pad = [None] * (diff-prev_diff)
				new_entries = self.num_entries[k] - prev_it_amount
				if new_entries > 0:
					# we move the information to the last part
					self.entries[k] = self.entries[k][:-new_entries] + pad + self.entries[k][-new_entries:] + pad_after
				else:
					# There is no new entry and we just fill with None
					self.entries[k] += pad + pad_after
				self.num_entries[k] = max_amount

	def write_to_csv_file(self):
		all_fieldnames = self.iteration_fieldnames + self.fieldnames
		with open(self.csv_file_path, 'a') as csv_file:
			self.writer = csv.DictWriter(csv_file, fieldnames=all_fieldnames)
			for i in range(len(self.entries['epoch'])):
				row = {}
				for k in all_fieldnames:
					row[k] = self.entries[k][i]
				self.writer.writerow(row)

class SummaryWriter:
	def __init__(self, graph, sess, summary_path):
		self.graph = graph
		self.sess = sess
		self.summary_path = summary_path
		make_dir(summary_path, clear=True)

		self.available_types = ['scalar']
		self.scalars = {}

	def clear(self):
		for key in self.scalars:
			self.scalars[key] = np.array([0, 0], dtype=np.float32)

	def add_item(self, key, type):
		assert type in self.available_types
		if type=='scalar':
			self.scalars[key] = np.array([0, 0], dtype=np.float32)

	def add_record(self, key, value, count=1):
		if key in self.scalars.keys():
			self.scalars[key] += np.array([value, count])

	def check_prefix(self, key):
		return key[:6]=='train/' or key[:5]=='test/'

	def get_prefix(self, key):
		if key[:6]=='train/': return 'train'
		if key[:5]=='test/': return 'test'
		assert(self.check_prefix(key))

	def remove_prefix(self,key):
		if key[:6]=='train/': return key[6:]
		if key[:5]=='test/': return key[5:]
		assert(self.check_prefix(key))

	def register_writer(self, summary_path, graph=None):
		make_dir(summary_path, clear=False)
		return tf.summary.FileWriter(summary_path, graph=graph)

	def setup(self):
		if self.graph is not None:
			with self.graph.as_default():
				self.summary_ph = {}
				self.summary = []
				self.summary_cmp = []
				with tf.variable_scope('summary_scope'):
					for key in self.scalars.keys():
						if self.check_prefix(key):
							# add to test summaries
							key_cmp = self.remove_prefix(key)
							if not(key_cmp in self.summary_ph.keys()):
								self.summary_ph[key_cmp] = tf.placeholder(tf.float32, name=key_cmp)
								self.summary_cmp.append(tf.summary.scalar(key_cmp, self.summary_ph[key_cmp], family='test'))
						else:
							# add to debug summaries
							assert not(key in self.summary_ph.keys())
							self.summary_ph[key] = tf.placeholder(tf.float32, name=key)
							self.summary.append(tf.summary.scalar(key, self.summary_ph[key], family='train'))

				self.summary_op = tf.summary.merge(self.summary)
				self.writer = self.register_writer(self.summary_path+'/debug', self.graph)
				if len(self.summary_cmp)>0:
					self.summary_cmp_op = tf.summary.merge(self.summary_cmp)
					self.train_writer = self.register_writer(self.summary_path+'/train')
					self.test_writer = self.register_writer(self.summary_path+'/test')

	def show(self, steps):
		if self.graph is not None:
			feed_dict = {'debug':{},'train':{},'test':{}}
			for key in self.scalars:
				value = self.scalars[key][0]/max(self.scalars[key][1],1e-3)
				if self.check_prefix(key):
					# add to train/test feed_dict
					key_cmp = self.remove_prefix(key)
					feed_dict[self.get_prefix(key)][self.summary_ph[key_cmp]] = value
				else: # add to debug feed_dict
					feed_dict['debug'][self.summary_ph[key]] = value

			summary = self.sess.run(self.summary_op, feed_dict['debug'])
			self.writer.add_summary(summary, steps)
			self.writer.flush()
			if len(self.summary_cmp)>0:
				summary_train = self.sess.run(self.summary_cmp_op, feed_dict['train'])
				summary_test = self.sess.run(self.summary_cmp_op, feed_dict['test'])
				self.train_writer.add_summary(summary_train, steps)
				self.test_writer.add_summary(summary_test, steps)
				self.train_writer.flush()
				self.test_writer.flush()

def get_logger(name=None):
	return Logger(name)
