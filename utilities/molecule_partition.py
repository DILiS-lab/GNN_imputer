import dgl
import numpy as np
import os
import sys
import json
import utilities.util_functions as utilities
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

def get_molecule_partition(samples, current_fold, NUM_FOLDS = 5, UNKNOWN_SAMPLE_LABEL = -100, MOLECULE_DICT_FILE = None):
	
	# We have two types of molecules, some for which we compute the loss and others that we don't (stationary molecules). 
	# The stationary molecules are not used to compute losses and we provide the abundcance values in their molecule feature vector. 
	# The former are divided into test unknown, test known, validation and train. All of these molecules do not have their abundance scores provided for the train test in the feature vector. 

	if os.path.isfile(MOLECULE_DICT_FILE):
		molecule_dict = np.load(MOLECULE_DICT_FILE, allow_pickle='TRUE').item()
		return molecule_dict.keys(), molecule_dict

	sample_molecule_sets = dict()
	new_samples = []

	for fold in range(NUM_FOLDS):
		MOLECULE_INFO_FILE = './saved_models/sample_info_all_moles_fold_' + str(current_fold) + '.csv'
		with open(MOLECULE_INFO_FILE, 'w') as f:
			f.write('sample, number of stationary molecules, number of train molecules, number of known test molecules, number of unknown test molecules \n') 

	for sample in samples:
		
		print('sample', sample)
		''' Read data '''
		(g,), _ = dgl.load_graphs('../data/' + sample)
		features = g.ndata['x'].float()
		labels = g.ndata['labels'].squeeze().numpy()
		labels = labels.astype(int)
		molecule_type = g.ndata['type'].numpy()
		abundances = g.ndata['abundance']

		''' get constant and unknown molecules '''
		test_unknown_set = list(np.where(labels == UNKNOWN_SAMPLE_LABEL)[0])

		constant_molecules = utilities.get_constant_set(molecule_type, [0, 1, 0], labels, fraction_to_take = 1.0, nan_label = UNKNOWN_SAMPLE_LABEL) #mRNA_molecule_set : [0, 1, 0]
		constant_molecules = constant_molecules.union(utilities.get_constant_set(molecule_type, [0, 0, 1], labels, fraction_to_take = 0.1, nan_label = UNKNOWN_SAMPLE_LABEL)) #protein_molecule_set : [0, 0, 1]
		constant_molecules = constant_molecules.union(utilities.get_constant_set(molecule_type, [1, 0, 0], labels, fraction_to_take = 1.0, nan_label = UNKNOWN_SAMPLE_LABEL)) #phosphosite_molecule_set : [1, 0, 0]
		constant_molecules = list(constant_molecules)
		print('Sample: ', sample,'Number of constant molecules ', len(constant_molecules))

		skf = StratifiedKFold(n_splits=NUM_FOLDS)
		new_labels = np.delete(labels, np.array(constant_molecules + test_unknown_set))
		mapping_indices_after_removal = utilities.get_mapping_after_removing_vals(range(len(labels)), constant_molecules + test_unknown_set)
		
		''' get test and non test data '''
		print("Labels division:", np.unique(new_labels, return_counts = True))

		_, label_counts = np.unique(new_labels, return_counts = True)

		if np.min(label_counts) > 2:
			fold = 0
			for train_set, test_known_set in skf.split(np.zeros(new_labels.shape), new_labels):
				if fold not in sample_molecule_sets.keys():
					sample_molecule_sets[fold] = {}
				print("FOLD", fold)
				
				train_set = utilities.map_given_to_original_index(train_set, mapping_indices_after_removal)
				test_known_set = utilities.map_given_to_original_index(test_known_set, mapping_indices_after_removal)
				
				sample_molecule_sets[fold][sample] = {'train': train_set, 'test': test_known_set, 'unknown': test_unknown_set, 'constant': constant_molecules}

				MOLECULE_INFO_FILE = './saved_models/sample_info_all_moles_fold_' + str(fold) + '.csv'
				with open(MOLECULE_INFO_FILE, 'a') as f:
					np.savetxt(f, [np.array([sample, len(constant_molecules), len(train_set), len(test_known_set), len(test_unknown_set)])], fmt = '%s', delimiter = ',')
				
				fold +=1

			new_samples.append(sample)
	
	os.makedirs(os.path.dirname(MOLECULE_DICT_FILE), exist_ok = True)

	for fold in range(NUM_FOLDS):
		MOLECULE_DICT_FILE = MOLECULE_DICT_FILE[0:-5] + str(fold) + '.npy'
		np.save(MOLECULE_DICT_FILE, sample_molecule_sets[fold])
	
	return new_samples, sample_molecule_sets[current_fold]