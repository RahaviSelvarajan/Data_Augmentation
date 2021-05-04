"""Source: Ting Chen et.al., A Simple Framework for Contrastive Learning of Visual Representations, arXiv:2002.05709v3 [cs.LG], 2020."""

class ColorDistortion:    
    def __init__(self, distortion):
        self.distortion = distortion
        
    def __call__(self, image):
        color_jitter = transforms.ColorJitter(0.8*self.distortion, 0.8*self.distortion, 
                                              0.8*self.distortion, 0.2*self.distortion)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=1.0)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray
            ])
        transformed_image = color_distort(image)
        image_array = np.asarray(transformed_image)
        return image_array
