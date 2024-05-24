import os
import csv
import glob
import json
import gzip
import secrets

import numpy as np
import tensorflow as tf

from .utils import load_catalogue, load_FASTA, load_predicted_PDB, seq2onehot
from .layers import MultiGraphConv, GraphConv, FuncPredictor, SumPooling


# +
class GradCAM(object):
    """
    GradCAM for protein sequences.
    [Adjusted for GCNs based on https://arxiv.org/abs/1610.02391]
    """
    def __init__(self, model, layer_name="GCNN_concatenate"):
        self.grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    def _get_gradients_and_filters(self, inputs, class_idx, use_guided_grads=False):
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(inputs)
            loss = predictions[:, class_idx, 0]
        grads = tape.gradient(loss, conv_outputs)

        if use_guided_grads:
            grads = tf.cast(conv_outputs > 0, "float32")*tf.cast(grads > 0, "float32")*grads

#         print(f"Conv Outputs: {conv_outputs.numpy()}")
#         print(f"Gradients: {grads.numpy()}")
        
        return conv_outputs, grads

    def _compute_cam(self, output, grad):
        weights = tf.reduce_mean(grad, axis=1)
        # perform weighted sum
        cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1).numpy()
        
        #print(f"CAM: {cam}")
        
        return cam

    def heatmap(self, inputs, class_idx, use_guided_grads=False):
        output, grad = self._get_gradients_and_filters(inputs, class_idx, use_guided_grads=use_guided_grads)
        cam = self._compute_cam(output, grad)
        cam_min, cam_max = cam.min(), cam.max()

#         return heatmap.reshape(-1)
    
        if cam_max - cam_min == 0:
            heatmap = np.zeros_like(cam)  # Avoid division by zero
        else:
            heatmap = (cam - cam_min) / (cam_max - cam_min)

        #print(f"Heatmap: {heatmap}")
        
        return heatmap.reshape(-1)


# -

class Predictor(object):
    """
    Class for loading trained models and computing GO/EC predictions and class activation maps (CAMs).
    """
    def __init__(self, model_prefix, gcn=True):
        self.model_prefix = model_prefix
        self.gcn = gcn
        self._load_model()

    def _load_model(self):
        self.model = tf.keras.models.load_model(self.model_prefix + '.hdf5',
                                                custom_objects={'MultiGraphConv': MultiGraphConv,
                                                                'GraphConv': GraphConv,
                                                                'FuncPredictor': FuncPredictor,
                                                                'SumPooling': SumPooling})
        # load parameters
        with open(self.model_prefix + "_model_params.json") as json_file:
            metadata = json.load(json_file)

        self.gonames = np.asarray(metadata['gonames'])
        self.goterms = np.asarray(metadata['goterms'])
        self.thresh = 0.1*np.ones(len(self.goterms))

    def _load_cmap(self, filename, cmap_thresh=10.0):
        if filename.endswith('.pdb'):
            D, seq = load_predicted_PDB(filename)
            A = np.double(D < cmap_thresh)
        elif filename.endswith('.npz'):
            cmap = np.load(filename)
            if 'C_alpha' not in cmap:
                raise ValueError("C_alpha not in *.npz dict.")
            D = cmap['C_alpha']
            A = np.double(D < cmap_thresh)
            seq = str(cmap['seqres'])
        elif filename.endswith('.pdb.gz'):
            rnd_fn = "".join([secrets.token_hex(10), '.pdb'])
            with gzip.open(filename, 'rb') as f, open(rnd_fn, 'w') as out:
                out.write(f.read().decode())
            D, seq = load_predicted_PDB(rnd_fn)
            A = np.double(D < cmap_thresh)
            os.remove(rnd_fn)
        else:
            raise ValueError("File must be given in *.npz or *.pdb format.")
        # ##
        S = seq2onehot(seq)
        S = S.reshape(1, *S.shape)
        A = A.reshape(1, *A.shape)

        return A, S, seq

    def predict(self, test_prot, cmap_thresh=10.0, chain='query_prot'):
        print ("### Computing predictions on a single protein...")
        self.Y_hat = np.zeros((1, len(self.goterms)), dtype=float)
        self.goidx2chains = {}
        self.prot2goterms = {}
        self.data = {}
        self.test_prot_list = [chain]
        if self.gcn:
            A, S, seqres = self._load_cmap(test_prot, cmap_thresh=cmap_thresh)

            y = self.model([A, S], training=False).numpy()[:, :, 0].reshape(-1)
            self.Y_hat[0] = y
            self.prot2goterms[chain] = []
            self.data[chain] = [[A, S], seqres]
            go_idx = np.where((y >= self.thresh) == True)[0]
            for idx in go_idx:
                if idx not in self.goidx2chains:
                    self.goidx2chains[idx] = set()
                self.goidx2chains[idx].add(chain)
                self.prot2goterms[chain].append((self.goterms[idx], self.gonames[idx], float(y[idx])))
        else:
            S = seq2onehot(str(test_prot))
            S = S.reshape(1, *S.shape)
            y = self.model(S, training=False).numpy()[:, :, 0].reshape(-1)
            self.Y_hat[0] = y
            self.prot2goterms[chain] = []
            self.data[chain] = [[S], test_prot]
            go_idx = np.where((y >= self.thresh) == True)[0]
            for idx in go_idx:
                if idx not in self.goidx2chains:
                    self.goidx2chains[idx] = set()
                self.goidx2chains[idx].add(chain)
                self.prot2goterms[chain].append((self.goterms[idx], self.gonames[idx], float(y[idx])))

    def predict_from_PDB_dir(self, dir_name, cmap_thresh=10.0):
        print ("### Computing predictions from directory with PDB files...")
        pdb_fn_list = glob.glob(dir_name + '/*.pdb*')
        self.chain2path = {pdb_fn.split('/')[-1].split('.')[0]: pdb_fn for pdb_fn in pdb_fn_list}
        self.test_prot_list = list(self.chain2path.keys())
        self.Y_hat = np.zeros((len(self.test_prot_list), len(self.goterms)), dtype=float)
        self.goidx2chains = {}
        self.prot2goterms = {}
        self.data = {}
        for i, chain in enumerate(self.test_prot_list):
            A, S, seqres = self._load_cmap(self.chain2path[chain], cmap_thresh=cmap_thresh)
            y = self.model([A, S], training=False).numpy()[:, :, 0].reshape(-1)
            self.Y_hat[i] = y
            self.prot2goterms[chain] = []
            self.data[chain] = [[A, S], seqres]
            go_idx = np.where((y >= self.thresh) == True)[0]
            for idx in go_idx:
                if idx not in self.goidx2chains:
                    self.goidx2chains[idx] = set()
                self.goidx2chains[idx].add(chain)
                self.prot2goterms[chain].append((self.goterms[idx], self.gonames[idx], float(y[idx])))

    def predict_from_catalogue(self, catalogue_fn, cmap_thresh=10.0):
        print ("### Computing predictions from catalogue...")
        self.chain2path = load_catalogue(catalogue_fn)
        self.test_prot_list = list(self.chain2path.keys())
        self.Y_hat = np.zeros((len(self.test_prot_list), len(self.goterms)), dtype=float)
        self.goidx2chains = {}
        self.prot2goterms = {}
        self.data = {}
        for i, chain in enumerate(self.test_prot_list):
            A, S, seqres = self._load_cmap(self.chain2path[chain], cmap_thresh=cmap_thresh)
            y = self.model([A, S], training=False).numpy()[:, :, 0].reshape(-1)
            self.Y_hat[i] = y
            self.prot2goterms[chain] = []
            self.data[chain] = [[A, S], seqres]
            go_idx = np.where((y >= self.thresh) == True)[0]
            for idx in go_idx:
                if idx not in self.goidx2chains:
                    self.goidx2chains[idx] = set()
                self.goidx2chains[idx].add(chain)
                self.prot2goterms[chain].append((self.goterms[idx], self.gonames[idx], float(y[idx])))

    def predict_from_fasta(self, fasta_fn):
        print ("### Computing predictions from fasta...")
        self.test_prot_list, sequences = load_FASTA(fasta_fn)
        self.Y_hat = np.zeros((len(self.test_prot_list), len(self.goterms)), dtype=float)
        self.goidx2chains = {}
        self.prot2goterms = {}
        self.data = {}

        for i, chain in enumerate(self.test_prot_list):
            S = seq2onehot(str(sequences[i]))
            S = S.reshape(1, *S.shape)
            y = self.model(S, training=False).numpy()[:, :, 0].reshape(-1)
            self.Y_hat[i] = y
            self.prot2goterms[chain] = []
            self.data[chain] = [[S], str(sequences[i])]
            go_idx = np.where((y >= self.thresh) == True)[0]
            for idx in go_idx:
                if idx not in self.goidx2chains:
                    self.goidx2chains[idx] = set()
                self.goidx2chains[idx].add(chain)
                self.prot2goterms[chain].append((self.goterms[idx], self.gonames[idx], float(y[idx])))

    def save_predictions(self, output_fn):
        print ("### Saving predictions to *.json file...")
        # pickle.dump({'pdb_chains': self.test_prot_list, 'Y_hat': self.Y_hat, 'goterms': self.goterms, 'gonames': self.gonames}, open(output_fn, 'wb'))
        with open(output_fn, 'w') as fw:
            out_data = {'pdb_chains': self.test_prot_list,
                        'Y_hat': self.Y_hat.tolist(),
                        'goterms': self.goterms.tolist(),
                        'gonames': self.gonames.tolist()}
            json.dump(out_data, fw, indent=1)

    def export_csv(self, output_fn, verbose):
        with open(output_fn, 'w') as csvFile:
            writer = csv.writer(csvFile, delimiter=',', quotechar='"')
            writer.writerow(['### Predictions made by DeepFRI.'])
            writer.writerow(['Protein', 'GO_term/EC_number', 'Score', 'GO_term/EC_number name'])
            if verbose:
                print ('Protein', 'GO-term/EC-number', 'Score', 'GO-term/EC-number name')
            for prot in self.prot2goterms:
                sorted_rows = sorted(self.prot2goterms[prot], key=lambda x: x[2], reverse=True)
                for row in sorted_rows:
                    if verbose:
                        print (prot, row[0], '{:.5f}'.format(row[2]), row[1])
                    writer.writerow([prot, row[0], '{:.5f}'.format(row[2]), row[1]])
        csvFile.close()

    def compute_GradCAM(self, layer_name='GCNN_concatenate', use_guided_grads=False):
        print ("### Computing GradCAM for each function of every predicted protein...")
        gradcam = GradCAM(self.model, layer_name=layer_name)
        grad_cam_scores = []

        self.pdb2cam = {}
        for go_indx in self.goidx2chains:
            pred_chains = list(self.goidx2chains[go_indx])
            print ("### Computing gradCAM for ", self.gonames[go_indx], '... [# proteins=', len(pred_chains), ']')
            for chain in pred_chains:
                if chain not in self.pdb2cam:
                    self.pdb2cam[chain] = {}
                    self.pdb2cam[chain]['GO_ids'] = []
                    self.pdb2cam[chain]['GO_names'] = []
                    self.pdb2cam[chain]['sequence'] = None
                    self.pdb2cam[chain]['saliency_maps'] = []
                self.pdb2cam[chain]['GO_ids'].append(self.goterms[go_indx])
                self.pdb2cam[chain]['GO_names'].append(self.gonames[go_indx])
                self.pdb2cam[chain]['sequence'] = self.data[chain][1]
                self.pdb2cam[chain]['saliency_maps'].append(gradcam.heatmap(self.data[chain][0], go_indx, use_guided_grads=use_guided_grads).tolist())
                print(self.pdb2cam[chain]['saliency_maps'])
                #print((self.data[chain][0], go_indx, use_guided_grads=use_guided_grads).tolist())
                heatmap = gradcam.heatmap(self.data[chain][0], go_indx, use_guided_grads=use_guided_grads)
                self.pdb2cam[chain]['saliency_maps'].append(heatmap.tolist())
                
                if self.gonames[go_indx] not in grad_cam_scores:
                    grad_cam_scores[self.gonames[go_indx]] = []
                grad_cam_scores[self.gonames[go_indx]].append(heatmap)
                
                grad_cam_scores.append(heatmap)  # Append the heatmap (activation scores) to grad_cam_scores
    
        return grad_cam_scores if grad_cam_scores else None

    def save_GradCAM(self, output_fn):
        print ("### Saving CAMs to *.json file...")
        # pickle.dump(self.pdb2cam, open(output_fn, 'wb'))
        with open(output_fn, 'w') as fw:
            json.dump(self.pdb2cam, fw, indent=1)


# +
from collections import Counter

class DeepFool:
    def __init__(self, predictor):
        self.predictor = predictor

    def _get_most_dissimilar_amino_acids(self, aa):
        dissimilar_aa = {
            'A': ['F', 'W', 'Y'],  # Alanine
            'C': ['L', 'I', 'M'],  # Cysteine
            'D': ['K', 'R', 'H'],  # Aspartic acid
            'E': ['K', 'R', 'H'],  # Glutamic acid
            'F': ['S', 'T', 'D'],  # Phenylalanine
            'G': ['F', 'W', 'Y'],  # Glycine
            'H': ['D', 'E', 'C'],  # Histidine
            'I': ['D', 'E', 'K'],  # Isoleucine
            'K': ['D', 'E', 'C'],  # Lysine
            'L': ['D', 'E', 'K'],  # Leucine
            'M': ['D', 'E', 'K'],  # Methionine
            'N': ['F', 'I', 'L'],  # Asparagine
            'P': ['E', 'F', 'W'],  # Proline
            'Q': ['F', 'I', 'L'],  # Glutamine
            'R': ['D', 'E', 'C'],  # Arginine
            'S': ['F', 'W', 'Y'],  # Serine
            'T': ['F', 'W', 'Y'],  # Threonine
            'V': ['D', 'E', 'K'],  # Valine
            'W': ['S', 'T', 'D'],  # Tryptophan
            'Y': ['S', 'T', 'D']   # Tyrosine
        }
        return dissimilar_aa.get(aa, ['X', 'X', 'X'])  # Default to 'X' if not found
    
    def _assign_probabilities(self, grad_cam_scores):
        total_score = sum(grad_cam_scores)
        return [score / total_score for score in grad_cam_scores]

    def run_deepfool(self, fasta_sequence):
        initial_prediction = self.predictor.predict(fasta_sequence)
        print(f"Fasta sequence: {fasta_sequence}")
        print(f"Initial Prediction: {initial_prediction}")
        mutation_thresholds = []

        for _ in range(50):
            mutated_sequence = list(fasta_sequence)
            grad_cam_scores = self.predictor.compute_GradCAM(layer_name='CNN_concatenate', use_guided_grads=False)
            print(f"Grad CAM Scores: {grad_cam_scores}")
            aa_probabilities = self._assign_probabilities(grad_cam_scores)
            print(f"AA Probabilities: {aa_probabilities}")
            mutations = 0
            misclassified = False
            mutated_positions = set()

            while not misclassified and mutations < len(fasta_sequence):
                sorted_indices = np.argsort(aa_probabilities)[::-1]
                print(f"Sorted Indices: {sorted_indices}")

                for idx in sorted_indices:
                    if idx in mutated_positions:
                        continue

                    original_aa = mutated_sequence[idx]
                    dissimilar_aas = self._get_most_dissimilar_amino_acids(original_aa)

                    for new_aa in dissimilar_aas:
                        mutated_sequence[idx] = new_aa
                        new_prediction = self.predictor.predict_function(''.join(mutated_sequence))

                        if new_prediction != initial_prediction:
                            mutation_thresholds.append(mutations + 1)
                            misclassified = True
                            break
                    
                    if misclassified:
                        break

                    mutated_positions.add(idx)
                    mutations += 1
                    
        # Save mutation thresholds as JSON
        json_output_path = "output/mutation_thresholds.json"
        os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
        with open(json_output_path, 'w') as json_file:
            json.dump(mutation_thresholds, json_file)
        print(f"Saved JSON output to {json_output_path}")
        
        print(mutation_thresholds)
        return mutation_thresholds

    def plot_mutation_thresholds(self, mutation_thresholds):
        plt.figure(figsize=(10, 6))
        plt.hist(mutation_thresholds, bins=range(1, max(mutation_thresholds) + 1), edgecolor='black')
        plt.title('Distribution of Mutation Thresholds for Misclassification')
        plt.xlabel('Number of Mutations')
        plt.ylabel('Frequency')
        plt.savefig(output_path)
        plt.show()
# -


