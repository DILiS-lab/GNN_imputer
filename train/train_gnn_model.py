import dgl
import numpy as np
import os
import sys
import json
import torch
import torch.nn as nn
from dgl.nn import GATConv
from dgl.nn.pytorch import HeteroGraphConv
import torch.nn.functional as F
from dgl.nn import GraphConv


path = os.getcwd()
parent_dir = os.path.abspath(os.path.join(path, os.pardir))
sys.path.append(parent_dir)
import utilities.molecule_partition as molecule_partition
import utilities.util_functions as util_functions
from models import GAT_model
# from GAT_model import * 

def train(fold = 'all', load_saved_model = False, MODE = 'Train'):
	torch.manual_seed(1)
	NUM_EPOCHS = 1000
	PATIENCE = 20
	UNKNOWN_SAMPLE_LABEL = -100
	SUBSTITUTE_VALUE = -100
	LR = 0.001
	NUM_FOLDS = 5

	samples_list = os.listdir('../data/')
	samples = [sample for sample in samples_list if sample.endswith(".dgl")]

	if fold == 'all':
		folds = range(NUM_FOLDS)
	else:
		folds = [int(fold)]

	for fold in folds:
		best_test_MAE = 1e6
		best_test_loss = 1e6
		best_test_R_2 = 1e6
		best_train_R_2 = 1e6
		best_epoch = 0
		num_epochs_since_improvement = 0
		sample_molecule_sets = None

		MOLECULE_DICT_FILE = './saved_models/samples_molecule_info_fold_' + str(fold) + '.npy'
		MODEL_PATH = './saved_models/best_model_fold_' + str(fold) + '.pt'
	
		model = GAT_model.GAT(in_dim = 4, hidden_dim = 40, out_dim = 1, num_heads= 20)
	
		optimizer = torch.optim.Adam(model.parameters(), lr=LR)
		START_EPOCH = 0
		if load_saved_model:
			print(MODEL_PATH, os.path.isfile(MODEL_PATH))
			print(MOLECULE_DICT_FILE + '.npy', os.path.isfile(MOLECULE_DICT_FILE))
			if os.path.isfile(MODEL_PATH) and os.path.isfile(MOLECULE_DICT_FILE):
				print('Loading saved model from ', MODEL_PATH)
				model.load_state_dict(torch.load(MODEL_PATH)['state_dict'])
				optimizer.load_state_dict(torch.load(MODEL_PATH)['optimizer'])
				START_EPOCH = torch.load(MODEL_PATH)['epoch']
				samples, sample_molecule_sets = molecule_partition.get_molecule_partition(samples, fold, NUM_FOLDS = NUM_FOLDS, UNKNOWN_SAMPLE_LABEL = UNKNOWN_SAMPLE_LABEL, MOLECULE_DICT_FILE = MOLECULE_DICT_FILE)
			for name, param in model.named_parameters():
				print('name', name, 'param', param.size())
			else:
				print("The given path does not exist. Please check.")
				# return

		
		if sample_molecule_sets is None:
			samples, sample_molecule_sets = molecule_partition.get_molecule_partition(samples, fold, NUM_FOLDS = NUM_FOLDS, UNKNOWN_SAMPLE_LABEL = UNKNOWN_SAMPLE_LABEL, MOLECULE_DICT_FILE= MOLECULE_DICT_FILE)
		
		fold_object_pred_vs_orig = dict() # dict contains dict of samples with each sample dict containing test molecules indices as IDs and values as [orig abundance, pred abundance]  
		fold_object_pred_vs_missing_orig = dict()

		for e in range(START_EPOCH, NUM_EPOCHS):
			print('***************** FOLD', fold, 'Epoch ', e, '************************')
			
			train_R_2, test_R_2 = [], []

			train_loss_val, test_loss_val, train_MAE, test_MAE = 0, 0, 0, 0

			if MODE == 'Test':
				model.eval()
			elif MODE == 'Train':
				model.train()

			for sample_idx, sample in enumerate(list(samples)):
				
				''' Read data '''
				print("Loading sample", sample)
				(g,), _ = dgl.load_graphs('../data/' + sample)

				features = g.nodes['molecule'].data['x'].float()
				labels = g.nodes['molecule'].data['labels'].squeeze().numpy()
				labels = labels.astype(int)
				molecule_type = g.nodes['molecule'].data['type'].numpy()
				abundances = g.nodes['molecule'].data['abundance']

				train_set = sample_molecule_sets[sample]['train'] 
				test_known_set = sample_molecule_sets[sample]['test']
				test_unknown_set = sample_molecule_sets[sample]['unknown']
				# constant_molecules = sample_molecule_sets[sample]['constant']

				features = util_functions.remove_abundance_for_samples(features, train_set + test_known_set, substitute_value = SUBSTITUTE_VALUE)				
				features_dict = {'molecule': features}

				pred = model(g, feat = features_dict)
				
				train_loss = F.mse_loss(pred[train_set], abundances[train_set])
				train_loss_val += train_loss.detach().numpy()
				test_loss_val += F.mse_loss(pred[test_known_set], abundances[test_known_set]).detach().numpy()

				# Compute accuracy on training/validation/test
				train_MAE += F.l1_loss(pred[train_set], abundances[train_set]).detach().numpy()
				test_MAE += F.l1_loss(pred[test_known_set], abundances[test_known_set]).detach().numpy()

				train_R_2.append(torch.corrcoef(torch.t(torch.cat((pred[train_set], abundances[train_set]), 1)))[0, 1]**2)
				test_R_2.append(torch.corrcoef(torch.t(torch.cat((pred[test_known_set], abundances[test_known_set]), 1)))[0, 1]**2)
			
				sample_pred_vs_original = util_functions.create_original_and_prediction_mapping_for_sample(abundances.detach().numpy(), pred.detach().numpy(), test_known_set)
				sample_pred_vs_missing_original = util_functions.create_original_and_prediction_mapping_for_sample(abundances.detach().numpy(), pred.detach().numpy(), test_unknown_set)
				
				fold_object_pred_vs_orig[sample] = sample_pred_vs_original
				fold_object_pred_vs_missing_orig[sample] = sample_pred_vs_missing_original
				# Backward
				optimizer.zero_grad()
				train_loss.backward()
				optimizer.step()

			
			train_R_2 = torch.Tensor(train_R_2)
			test_R_2 = torch.Tensor(test_R_2)

		
			num_epochs_since_improvement +=1

			# Save the best validation accuracy and the corresponding test accuracy.
			if best_test_loss > test_loss_val:
				best_test_MAE = test_MAE
				best_test_loss = test_loss_val
				best_test_R_2 = torch.mean(test_R_2).item()
				best_train_R_2 = torch.mean(train_R_2).item()
				best_epoch = e
				num_epochs_since_improvement = 0
				if MODE == 'Train':
					print('New best model with loss', best_test_loss,' saved at ', MODEL_PATH)
					state = {'epoch': best_epoch,
    						'state_dict': model.state_dict(),
    						'optimizer': optimizer.state_dict()
    						}
					torch.save(state, MODEL_PATH)

				FILE_PATH = './saved_models/mapping_abundances_prediction_vs_original_fold_' + str(fold) + '.json'
				with open(FILE_PATH, 'w') as fp:
					json.dump(fold_object_pred_vs_orig, fp)

				FILE_PATH = './saved_models/mapping_abundances_prediction_vs_missing_original_fold_' + str(fold) + '.json'
				with open(FILE_PATH, 'w') as fp:
					json.dump(fold_object_pred_vs_missing_orig, fp)

			else:
				print('No improvement in the test loss since', num_epochs_since_improvement, 'epochs')

			print('After epoch: ', e, 'train R^2', torch.mean(train_R_2).item(), 'test R^2', torch.mean(test_R_2).item(),'and best test R^2', best_test_R_2)

			with open('performance_GNN_regression.csv', 'a') as f:
				np.savetxt(f, [np.array([fold, e, train_loss_val, test_loss_val, best_test_loss, train_MAE, test_MAE, best_test_MAE, torch.mean(train_R_2).item(), torch.std(train_R_2).item(), torch.mean(test_R_2).item(), torch.std(test_R_2).item(), best_test_R_2])], fmt = '%s', delimiter = ',')
				
			if num_epochs_since_improvement > PATIENCE:
				print('Patience', str(PATIENCE) ,' reached. Stopping training. Check saved model at', MODEL_PATH)
				break

if __name__ == "__main__":
	FOLD = sys.argv[1]
	LOAD_SAVED_MODEL = True
	MODE = 'Train' # 'Test'
	
	with open('performance_GNN_regression.csv', 'a') as file:
		file.write('fold,epoch,train_loss,test_loss,best_test_loss,train_MAE,test_MAE,best_test_MAE,train R^2,train R^2 Std,test R^2,test R^2 Std,best_test R^2 \n')

	# Create the model with given dimensions
	train(fold = FOLD, load_saved_model = LOAD_SAVED_MODEL, MODE  = MODE)