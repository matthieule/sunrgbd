"""CLass to parse the SUNRGBD database"""
import h5py
import os
import scipy.io
import matplotlib

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

from scipy.ndimage.interpolation import zoom 
from tqdm import tqdm

class SunRGBDDataset:
    """SunRGBD curated dataset"""
    
    n_label = 38
    
    def __init__(self, data_dir, toolbox_dir, output_dir, *,
                resize=True, nx=128, ny=128, grayscale=True):
        """Init
        
        :param data_dir: directory where the image data is stored
         It can be downloaded from http://rgbd.cs.princeton.edu/data/SUNRGBD.zip
        :param toolbox_dir: directory to the toolbox which contain 
         files to link segmentations and labels. It can be downloaded
         at http://rgbd.cs.princeton.edu/data/SUNRGBDtoolbox.zip
        :param output_dir: output directory where the curated database will
         be saved
        :param resize: boolean, whether or not to resize the images
        :param nx: if resize is True, the shape of the resized image
         will be nx on the first axis
        :param ny: if resize is True, the shape of the resized image
         will be ny on the second axis
        :param grayscale: boolean, whether or not to turn the image grayscale
        """
          
        self.data_dir = data_dir
        path_2_label_code = os.path.join(
            toolbox_dir, 'Metadata', 'SUNRGBD2Dseg.mat'
        )
        path_2_metadata = os.path.join(
            toolbox_dir, 'Metadata', 'SUNRGBDMeta.mat'
        )
        
        self.meta_information = scipy.io.loadmat(path_2_metadata)
        self.label_code = h5py.File(path_2_label_code)
        
        path_2_label_name = os.path.join(
            toolbox_dir, 'Metadata', 'seg37list.mat'
        )
        label_name = scipy.io.loadmat(path_2_label_name)
        self.label_name = ['background'] + [
            label_name['seg37list'][0][idx][0] for idx in range(self.n_label-1)
        ]
        
        self.image_dir = os.path.join(output_dir, 'image')
        self.label_dir = os.path.join(output_dir, 'label')
        for dir_ in [self.image_dir, self.label_dir]:
            if not os.path.exists(dir_):
                os.mkdir(dir_)
        
        self.preprocess = {
            'resize': resize,
            'nx': nx,
            'ny': ny,
            'grayscale': grayscale
        }
    
    def _preprocess(self, image, label):
        """Preprocess the images and labels
        
        :param image: 3D ndarray with shape (x, y, 3)
        :param label: 2D ndarray with shape (x, y)
        :return: tuple wit preprocessed image and label
        """
        
        if self.preprocess['grayscale']:
            image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
            image = image[..., None]
        
        if self.preprocess['resize']:
            ratio = self._resize_ratio(image)
            image = zoom(image, ratio, order=1)
            label = self._resize_label(label, ratio[:-1])
        
        return image, label
    
    def _resize_label(self, label, ratio):
        """Resize label
        
        :param label: 2D ndarray with size (x, y)
        :param ratio: tuple (rx, ry) with the resizing ratio on
         the x and y axis
        :return: resized image
        """
        
        label_resized = np.zeros(
            (self.preprocess['nx'], self.preprocess['ny'], self.n_label)
        )
        for class_idx in range(self.n_label):
            if class_idx in label:
                current_class = np.zeros(label.shape)
                current_class[label==class_idx] = 1
                current_class = zoom(current_class, ratio, order=1)
                label_resized[..., class_idx] = current_class
        label = np.argmax(label_resized, axis=-1)
        
        return label
    
    def _resize_ratio(self, image):
        """Get resizing ratio
        
        :param image: 3D ndarray with shape (x, y, n_channel)
        :return: ratio, tuple (rx, ry, n_channel) with the resizing
         factor on the spatial dimensions (the number of channels is
         left unchanged)
        """
        
        ratio = np.array(
                [self.preprocess['nx'], self.preprocess['ny'], image.shape[-1]]
        ).astype(float)
        ratio /= np.array(image.shape).astype(float)
        
        return ratio
    
    def _save_img(self, image, label, idx):
        """Save image to disk
        
        :param image: 3D ndarray with shape (x, y, nbr_channel)
        :param label: 2D ndarray with shape (x, y)
        :param idx: identifying index of the image
        """
        
        png_fname = os.path.join(self.image_dir, '%05d.png' % idx)
        if self.preprocess['grayscale']:
            image = image[..., 0] 
        image = image.astype(np.uint8)
        image = Image.fromarray(image)
        image.save(png_fname)
            
        png_fname = os.path.join(self.label_dir, '%05d.png' % idx)
        label = label.astype(np.uint8)
        label = Image.fromarray(label)
        label.save(png_fname)
    
    def curate_database(self):
        """Save a curated version of the SUNRGBD database"""

        for idx in tqdm(range(self.size)):
            image, label = self.parse_one_img(idx)
            image, label = self._preprocess(image, label)
            self._save_img(image, label, idx)
    
    def image_label(self, idx):
        """Return one pair of image and label
        
        :param idx: identifying index of the image
        :return: tuple of ndarray (image, label) from the
         curated database
        """
        
        png_fname = os.path.join(self.image_dir, '%05d.png' % idx)
        image = Image.open(png_fname)
        encoding = 'L' if self.preprocess['grayscale'] else 'RGB'
        image = image.convert(encoding)
        image = np.asarray(image)
        
        png_fname = os.path.join(self.label_dir, '%05d.png' % idx)
        label = Image.open(png_fname)
        label = np.asarray(label)
        
        return image, label
        
    
    def parse_one_img(self, idx):
        """Parse on original image of the SUNRGBD database
        
        :param idx: identifying index of the image to be parsed
        :return: tuple of ndarray (image, label) where the shape
         of image is (x, y, 3) and the shape of label is (x, y)
        """
        
        seg_label = self.label_code['SUNRGBD2Dseg']['seglabel'][idx][0]
        label = self.label_code[seg_label].value
        label = label.T

        image_folder = self.meta_information['SUNRGBDMeta'][0][idx][0][0][8:]
        image_folder = os.path.join(self.data_dir, image_folder, 'image')
        _, _, image_filename = next(os.walk(image_folder))
        image = Image.open(os.path.join(image_folder, image_filename[0]))
        image = image.convert('RGB')
        image = np.asarray(image)

        assert np.all(label.shape == image.shape[:-1])
        assert image.shape[-1] == 3
        
        return image, label
    
    def plot_one_img(self, idx):
        """Plot one pair of image and label from the curated database
        
        :param idx: indentifying index of the image
        """
        
        image, label = self.image_label(idx)
        title = ''.join(
            test.label_name[idx] + ' / ' for idx in np.unique(label.astype(int))
        )
        
        plt.imshow(image, cmap='gray')
        plt.title(title, fontsize=10)
        plt.colorbar()
        plt.show()
        
        np.random.seed(0)
        cmap = matplotlib.colors.ListedColormap(np.random.rand(self.n_label, 3))
        plt.imshow(label, cmap=cmap)
        plt.title(title, fontsize=10)
        plt.colorbar()
        plt.show()
        
    @property
    def size(self):
        """Size of the database
        
        :return: number of images in the database
        """
        
        n_image = len(self.meta_information['SUNRGBDMeta'][0])
        
        return n_image  

