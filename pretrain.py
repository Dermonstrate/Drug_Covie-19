from DeepPurpose import DTI as models
from DeepPurpose import utils, dataset
from deepfrier.DeepFRI import DeepFRI

X_drugs, X_targets, y = dataset.read_file_training_dataset_drug_target_pairs('C:\\Users\\57560\\PycharmProjects\\New_drug\\data\\pretrained.txt')

print('Drug 1: ' + X_drugs[0])
print('Target 1: ' + X_targets[0])
print('Score 1: ' + str(y[0]))

drug_encoding, target_encoding = 'Morgan', 'CNN'

train, val, test = utils.data_process(X_drugs, X_targets, y,
                                drug_encoding, target_encoding,
                                split_method='random',frac=[0.7,0.1,0.2],
                                random_seed = 1)

model = models.model_pretrained(path_dir= 'C:\\Users\\57560\\PycharmProjects\\New_drug\\tutorial_model2')
model.train(train, val, test)
model.save_model('./tutorial_model3')

