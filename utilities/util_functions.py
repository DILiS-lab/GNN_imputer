import dgl
import numpy as np
import os
import sys
import json

def get_constant_set(all_molecule_type_labels, molecule_type, all_molecule_abundances_label, fraction_to_take = 1.0, nan_label = -10):
	# print('all_molecule_type_labels', len(all_molecule_type_labels), 'molecule_type', molecule_type)

	all_molecule_type_labels = np.argmax(all_molecule_type_labels, axis=1) 
	molecule_type = np.argmax(molecule_type)

	# print(np.where(all_molecule_type_labels == np.array(molecule_type)))
	molecules_w_given_label = set(np.where(all_molecule_type_labels == np.array(molecule_type))[0])
	# print('molecules_w_given_label', len(molecules_w_given_label))
	molecules_w_abundances_present = set(np.where(all_molecule_abundances_label != nan_label)[0])
	# print('molecules_w_abundances_present', len(molecules_w_abundances_present))

	molecules_w_given_label = list(molecules_w_given_label.intersection(molecules_w_abundances_present))

	import random
	random.shuffle(molecules_w_given_label)
	constant_molecules = molecules_w_given_label[0:int(len(molecules_w_given_label)*fraction_to_take)]

	return set(constant_molecules)

def create_original_and_prediction_mapping_for_sample(original, prediction, test_node_indices):
	'''
	create object where the key is the index of the test node, the values are a list [orginal val, predicted val]

	'''
	mapping = dict()

	for test_node_index in test_node_indices:
		mapping[str(test_node_index)] = [float(original[test_node_index][0]), float(prediction[test_node_index][0])]

	return mapping

def get_mapping_after_removing_vals(original_indices, indices_to_remove):
	new_vals = np.delete(original_indices, indices_to_remove)
	mapping = dict()

	for idx, val in enumerate(new_vals):
		mapping[idx] = val 

	return mapping

def map_given_to_original_index(base, mapping):
	new_vals = []

	for val in base:
		new_vals.append(mapping[val])

	return new_vals

def remove_abundance_for_samples(features, sample_ids, substitute_value = 0):

	for node_idx in sample_ids:
		# print(node_idx, features[node_idx])
		features[node_idx, 0] = substitute_value
		# print(node_idx, features[node_idx])

	return features