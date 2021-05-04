class RGBJitter:
    def __init__(self, distortion):
        self.distortion = distortion
        
    def __call__(self, image):       
        image_array = np.asarray(image)
        image_dtype = image_array.dtype
        image_shifted = self.pca_augmentation(image_array)
        image = image_shifted.astype(image_dtype)
        return image
    
    def pca_augmentation(self, image):
        #Normalization
        img_array = image/255.0
        mean = np.mean(img_array, axis=(0,1))
        img_norm = (img_array - mean)
        #Covariance matrix 
        img_rs = img_norm.reshape(img_norm.shape[0]*img_norm.shape[1], img_norm.shape[2])
        cov_matrix = np.cov(img_rs, rowvar=False)
        #Principal Components
        eig_values, eig_vectors = np.linalg.eig(cov_matrix)
        #Sorting the eigen_vectors in the order of their eigen_values (highest to lowest)
        indices = np.flipud(eig_values.argsort()) #indices of the sorted eig_values in decreasing order
        eig_values = sorted(eig_values, reverse=True) #eig_values sorted in descending order
        eig_vectors = eig_vectors[:, indices]
        alphas = np.random.normal(0, self.distortion, 3)
        delta = np.dot(eig_vectors, (alphas*eig_values))
        
        image_distorted = img_norm + delta
        image_distorted = (image_distorted + mean)*255.0
        image_distorted = np.rint(np.clip(image_distorted, 0, 255)).astype('uint8')
        #image_distorted = (image_distorted - 193.09203) / (56.450138 + 1e-7)
        return image_distorted
        
