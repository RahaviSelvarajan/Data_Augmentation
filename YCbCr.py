""" Source: Hosseini et.al., On Transferability of Histological Tissue Labels in Computational Pathology, ECCV, 2020."""


class YCbCr:   
    def __init__(self, distortion):
        self.distortion = distortion
    
    def __call__(self, image):     
        image_array = np.asarray(image)
        image_dtype = image_array.dtype
        
        A = np.array([[65.481, 128.553, 24.966], [-37.797, -74.203, 112.0], [112.0, -93.786, -18.214]])
        A = A / 255.
        b = np.array([16, 128, 128])

        x = image_array.reshape((image_array.shape[0] * image_array.shape[1], image_array.shape[2]))
        image_ycbcr = x @ A.T + b
        image_ycbcr = image_ycbcr.reshape((image_array.shape[0], image_array.shape[1], image_array.shape[2]))

        Y = image_ycbcr[:, :, 0]
        Cb = image_ycbcr[:, :, 1]
        Cr = image_ycbcr[:, :, 2]

        mean_Cb = np.mean(Cb, axis=(0, 1))
        mean_Cr = np.mean(Cr, axis=(0, 1))
        std_Cb = np.std(Cb, axis=(0, 1))
        std_Cr = np.std(Cr, axis=(0, 1))

        Cb_centered = Cb - mean_Cb
        Cr_centered = Cr - mean_Cr

        Cb_centered_augmented = Cb_centered + np.random.normal(loc=0, scale=(self.distortion * std_Cb))
        Cr_centered_augmented = Cr_centered + np.random.normal(loc=0, scale=(self.distortion * std_Cr))

        Cb_augmented = Cb_centered_augmented + mean_Cb
        Cr_augmented = Cr_centered_augmented + mean_Cr

        image_perturbed_ycbcr = np.empty(image_array.shape)
        image_perturbed_ycbcr[:, :, 0] = Y
        image_perturbed_ycbcr[:, :, 1] = Cb_augmented
        image_perturbed_ycbcr[:, :, 2] = Cr_augmented

        inv_A = np.linalg.inv(A)
        image_perturbed = (image_perturbed_ycbcr - b) @ inv_A.T
        image_perturbed = np.rint(np.clip(image_perturbed, 0, 255)).astype('uint8')
        #image_perturbed = (image_perturbed - 193.09203) / (56.450138 + 1e-7)
        image = image_perturbed.astype(image_dtype)        
        return image
