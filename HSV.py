"""Source: David Tellez et.al., Quantifying the effects of data augmentation and stain color normalization
in convolutional neural networks for computational pathology, arXiv:1902.06543v2 [cs.CV], 2020"""

class HSV:
    def __init__(self, distortion):
        self.distortion = distortion
    
    def __call__(self, image):     
        image_array = np.asarray(image)
        image_dtype = image_array.dtype
        
        HSV_image = rgb2hsv(image_array)
        H = HSV_image[:, :, 0]
        H_rad = H * [2 * math.pi] - math.pi
        S = HSV_image[:, :, 1]
        V = HSV_image[:, :, 2]
        mean_H_rad = circmean(H_rad)
        std_H_rad = circstd(H_rad)
        mean_S = np.mean(S, axis=(0, 1))
        std_S = np.std(S, axis=(0, 1))
        
        H_rad_centered = np.angle(np.exp(1j * (H_rad - mean_H_rad)))
        H_rad_centered_augmented = H_rad_centered + np.random.normal(loc=0, scale=(self.distortion * std_H_rad))
        H_rad_augmented = np.angle(np.exp(1j * (H_rad_centered_augmented + mean_H_rad)))
        H_augmented = np.divide(H_rad_augmented + math.pi, 2 * math.pi)
        
        S_centered = S - mean_S
        S_centered_augmented = S_centered + np.random.normal(loc=0, scale=(self.distortion * std_S))
        S_augmented = S_centered_augmented + mean_S
        
        image_perturbed_HSV = np.empty(image_array.shape)
        image_perturbed_HSV[:, :, 0] = H_augmented
        image_perturbed_HSV[:, :, 1] = S_augmented
        image_perturbed_HSV[:, :, 2] = V
        
        image_rgb = hsv2rgb(image_perturbed_HSV)
        image_rgb = image_rgb*255.0
        image_perturbed = np.rint(np.clip(image_rgb, 0, 255)).astype('uint8')
        #image_perturbed = (image_perturbed - 193.09203) / (56.450138 + 1e-7) 
        image = image_perturbed.astype(image_dtype)
        return image
