from matplotlib import pyplot as plt
import torch
import numpy as np
import h5py
from torchvision.utils import save_image
import cv2
import random

def show_images_grid(imgs_, num_images=25):
    ncols = int(np.ceil(num_images**0.5))
    nrows = int(np.ceil(num_images / ncols))
    _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
    axes = axes.flatten()

    for ax_i, ax in enumerate(axes):
        if ax_i < num_images:
            ax.imshow(imgs_[ax_i], cmap='Greys_r', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')
    plt.savefig("/home/mila/l/lea.cote-turcotte/LUSR/data/sample_with_plt.png")

class Shape3dDataset():
    def __init__(self):
        self.factors = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                        'orientation']
        self.num_values_per_factors = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 
                            'scale': 8, 'shape': 4, 'orientation': 15}
        self.dataset = None
        self.images = None
        self.labels = None
        self.image_shape = None
        self.label_shape = None
        self.n_samples = None

    def process_obs(self, obs, device): 
        obs = np.ascontiguousarray(obs, dtype=np.float32)
        #obs = cv2.resize(obs[:84, :, :], dsize=(64,64), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(np.transpose(obs, (0,3,1,2))).to(device)

    def load_dataset(self, file_dir):
        self.dataset = h5py.File(file_dir, 'r')
        self.images = self.dataset['images'] # array shape [480000,64,64,3], uint8 in range(256)
        self.labels = self.dataset['labels']  # array shape [480000,6], float64
        self.image_shape = self.images.shape[1:]  # [64,64,3]
        self.label_shape = self.labels.shape[1:]  # [6]
        self.n_samples = self.labels.shape[0] # 480000

    def get_index(self, factors):
        """ Converts factors to indices in range(num_data)
        Args:
            factors: np array shape [6,batch_size].
                    factors[i]=factors[i,:] takes integer values in 
                    range(self.num_values_per_factors[self.factors[i]]).

        Returns:
            indices: np array shape [batch_size].
        """
        indices = 0
        base = 1
        for factor, name in reversed(list(enumerate(self.factors))):
            indices += factors[factor] * base
            base *= self.num_values_per_factors[name]
        return indices
    
    def sample_with_factors(self, batch_size):
        """ Samples a batch of images and the ground truth factors
        Args:
            batch_size: number of images to sample.
        Returns:
            batch: images shape [batch_size,64,64,3]
            factors: factors shape [6, batch_size]
        """
        factors = np.zeros([len(self.factors), batch_size], dtype=np.int32)
        for factor, name in enumerate(self.factors):
            num_choices = self.num_values_per_factors[name]
            factors[factor] = np.random.choice(num_choices, batch_size)
        indices = self.get_index(factors)
        ims = []
        for ind in indices:
            im = self.images[ind]
            im = np.asarray(im)
            ims.append(im)
        ims = np.stack(ims, axis=0)
        ims = ims / 255. # normalise values to range [0,1]
        ims = ims.astype(np.float32)
        return ims.reshape([batch_size, 64, 64, 3]), factors # factors size (6, batch_size)

    def sample_random_batch(self, batch_size):
        """ Samples a random batch of images.
        Args:
            batch_size: number of images to sample.

        Returns:
            batch: images shape [batch_size,64,64,3].
        """
        indices = np.random.choice(self.n_samples, batch_size)
        ims = []
        for ind in indices:
            im = self.images[ind]
            im = np.asarray(im)
            ims.append(im)
        ims = np.stack(ims, axis=0)
        ims = ims / 255. # normalise values to range [0,1]
        ims = ims.astype(np.float32)
        return ims.reshape([batch_size, 64, 64, 3])

    #TO DO : CHANGE MORE THAN ONE FACTOR, K FACTORS
    def sample_changed_factor(self, batch_size, changed_factor_str=None, other_changed_factor_str=None, k=None):
        """ Samples a batch of images with changed_factor varying randomly, but with
            the other factors not varying.
        Args:
            batch_size: number of images to sample.
            changed_factor: index of factor that is changed in range(6).
            k: number of factor to change
        Returns:
            batch: images shape [batch_size,64,64,3]
        """
        factors = np.zeros([len(self.factors), batch_size], dtype=np.int32)
        for factor, name in enumerate(self.factors):
            num_choices = self.num_values_per_factors[name]
            value = np.random.choice(num_choices, 1)
            factors[factor] = np.full(batch_size, value, dtype=np.int32)

        if k == None:
            if changed_factor_str == None:
                changed_factor_str = random.choice(self.factors)
            changed_factor = self.factors.index(changed_factor_str)
            factors[changed_factor] = np.random.choice(self.num_values_per_factors[changed_factor_str], batch_size)

            if other_changed_factor_str != None:
                other_changed_factor = self.factors.index(other_changed_factor_str)
                factors[other_changed_factor] = np.random.choice(self.num_values_per_factors[other_changed_factor_str], batch_size)
        elif k != None:
            for k in range(k):
                changed_factor_str = random.choice(self.factors)
                changed_factor = self.factors.index(changed_factor_str)
                factors[changed_factor] = np.random.choice(self.num_values_per_factors[changed_factor_str], batch_size)

        indices = self.get_index(factors)
        ims = []
        for ind in indices:
            im = self.images[ind]
            im = np.asarray(im)
            ims.append(im)
        ims = np.stack(ims, axis=0)
        ims = ims / 255. # normalise values to range [0,1]
        ims = ims.astype(np.float32)
        return ims.reshape([batch_size, 64, 64, 3])
    
    def sample_fixed_factor(self, batch_size, fixed_factor_str, other_fixed_factor_str=None):
        """ Samples a batch of images with fixed_factor=fixed_factor_value, but with
            the other factors varying randomly.
        Args:
            batch_size: number of images to sample.
            fixed_factor: index of factor that is fixed in range(6).
            fixed_factor_value: integer value of factor that is fixed 
            in range(self.num_values_per_factors[self.factors[fixed_factor]]).

        Returns:
            batch: images shape [batch_size,64,64,3]
        """

        factors = np.zeros([len(self.factors), batch_size],dtype=np.int32)
        for factor, name in enumerate(self.factors):
            num_choices = self.num_values_per_factors[name]
            factors[factor] = np.random.choice(num_choices, batch_size)

        fixed_factor = self.factors.index(fixed_factor_str)
        fixed_factor_value = np.random.choice(self.num_values_per_factors[fixed_factor_str], 1)
        factors[fixed_factor] = np.full(batch_size, fixed_factor_value, dtype=np.int32)

        if other_fixed_factor_str != None:
            other_fixed_factor = self.factors.index(other_fixed_factor_str)
            other_fixed_factor_value = np.random.choice(self.num_values_per_factors[other_fixed_factor_str], 1)
            factors[other_fixed_factor] = np.full(batch_size, other_fixed_factor_value, dtype=np.int32)


        indices = self.get_index(factors)
        ims = []
        for ind in indices:
            im = self.images[ind]
            im = np.asarray(im)
            ims.append(im)
        ims = np.stack(ims, axis=0)
        ims = ims / 255. # normalise values to range [0,1]
        ims = ims.astype(np.float32)
        return ims.reshape([batch_size, 64, 64, 3])
    
    def create_weak_vae_batch(self, batch_size, device, k=None):
        batch = []
        for i in range(batch_size):
            img_batch = self.sample_changed_factor(2, k=k)
            img_batch = self.process_obs(img_batch, device)
            sample_pair = torch.cat([img_batch[0], img_batch[1]], dim=1).unsqueeze(0) # torch.Size([1, 3, 128, 64])
            if batch == []:
                batch = sample_pair
            else:
                batch = torch.cat([batch, sample_pair], dim=0) # torch.Size([batch_size, 3, 128, 64])
        return batch

    def create_cycle_vae_batch(self, nb_groups, group_size, device):
        batch = []
        for group in range(nb_groups):
            img_batch = self.sample_changed_factor(group_size, 'shape', 'scale')
            img_batch = self.process_obs(img_batch, device)
            batch.append(img_batch.unsqueeze(0)) # list(torch.Size([1, 10, 3, 128, 64]))
        return torch.cat(batch, dim=0) # torch.Size([5, 10, 3, 128, 64])
    
    def create_gvae_batch(self, nb_groups, group_size, device):
        batch = []
        for group in range(nb_groups):
            #img_batch = self.sample_fixed_factor(group_size, 'shape', 'scale')
            img_batch = self.sample_fixed_factor(group_size, 'shape')
            img_batch = self.process_obs(img_batch, device) # torch.Size([10, 3, 128, 64])
            batch.append(img_batch.unsqueeze(0)) # list(torch.Size([1, 10, 3, 128, 64]))
        return torch.cat(batch, dim=0) # torch.Size([5, 10, 3, 128, 64])

    def generate_batch_factor_code(self, model, num_points, batch_size, device='cpu'):
        """ Samples a batch of images and the ground truth factors
        Args:
            model: Function that takes observations as input and
            outputs a dim_representation sized representation for each observation.
            num_points: Number of points to sample.
            batch_size: Batch size for sampling.
        Returns:
            representations: Codes (num_codes, num_points)-np array.
            factors: Factors generating the codes (num_factors, num_points)-np array.
        """
        representations = None
        factors = None
        i = 0
        while i < num_points:
            num_points_iter = min(num_points - i, batch_size) # num_points or batch_size if batch_size < num_points
            current_observations, current_factors = self.sample_with_factors(num_points_iter) #shape [num_points_iter, 64, 64, 3] and [num_factors, batch_size]
            if i == 0:
                factors = np.transpose(current_factors)
                representations, _ = model(self.process_obs(current_observations, device)) # torch.Size([num_points_iter, latent_size])
                representations = representations.numpy()
            else:
                factors = np.vstack((factors, np.transpose(current_factors)))
                new_representations, _ = model(self.process_obs(current_observations, device))
                representations = np.vstack((representations, new_representations.numpy()))
            i += num_points_iter
        return np.transpose(representations), np.transpose(factors) # shape [num_codes, num_train] and shape [num_factors, num_train]
     