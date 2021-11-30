import cv2
import os
import albumentations as A
from PIL import Image
from tqdm import tqdm
import argparse

# to run :
# ! python3 augment_v2.py --root-folder="birds_dataset/train_images"

class ImageAugmentationClass():
    """
    """
    def __init__(self, img_folder):
        """
        """
        self.img_folder = img_folder
        self.transformed_images = list()
    
    
    def augment(self, rotations_angles, brightness_limits, contrast_limits, blur_limits=None):
        """
        """
        images_list = os.listdir(self.img_folder)
        for i in tqdm(range(len(images_list))):
            #read image
            img_name = images_list[i]
            img_path = os.path.join(self.img_folder, img_name)
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            #perform rotations
            #each call to these functions should add images to the 
            #self.transormed_images list
            if not(rotations_angles == None):
                self.rotate_image(image=img, rotations_angles=rotations_angles)
            if (not(brightness_limits == None) and not(contrast_limits == None)):
                self.change_brightness_contrast_nflip(image=img, brightness_limits=brightness_limits, 
                                                contrast_limits=contrast_limits)
                
            if not(blur_limits == None):
                self.perform_bluring(image=img, blur_limits=blur_limits)
            
            #save images
            #self.save_images(generic_name=img_path)
            self.save_images(generic_name=self.img_folder)
            #clear memeory after each image treatement
            self.clear_memroy()
        return
    
    def clear_memroy(self):
        """
        """
        self.transormed_images = list()
        return
    
    def save_images(self, generic_name):
        """
        """
        for i, image in enumerate(self.transformed_images):
            comp = generic_name.split(".")
            #name = comp[0]+"."+comp[1] + "_" +str(i+1) + "_." + comp[-1]
            name = comp[0] + "." +comp[1]+"/"+ comp[1]+ "_copy_" + str(i+1) + "_.jpg"
            #name = os.getcwd() + "/" + comp[0] + "_" + str(i+1)+"_."+comp[-1]
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(name, image)
        return
    
    def perform_bluring(self, image, blur_limits):
        """
        for only one image
        """
        transform = A.GaussianBlur(sigma_limit=blur_limits, p=1.0)
        transormed_img = transform(image=image)['image']
        self.transformed_images.append(transormed_img)
        return
    
    def change_brightness_contrast_nflip(self, image, brightness_limits, contrast_limits):
        """
        for only one image
        """
        transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=brightness_limits,
                                                   contrast_limit=contrast_limits,
                                                   p=1.0),
            A.HorizontalFlip(p=1.0)
            ])
        
        transformed_img = transform(image=image)['image']
        self.transformed_images.append(transformed_img)
        
        return
    
    def rotate_image(self, image, rotations_angles):
        """
        perform for rotations
        """
        transform1 = A.Rotate(limit=(rotations_angles[0], rotations_angles[1]),p=1.0)
        transform2 = A.Rotate(limit=(rotations_angles[1], rotations_angles[2]),p=1.0)
        transform3 = A.Rotate(limit=(rotations_angles[2], rotations_angles[3]),p=1.0)
        transform4 = A.Rotate(limit=(rotations_angles[3], rotations_angles[-1]),p=1.0)
        
        transformed_img = transform1(image=image)['image']
        self.transformed_images.append(transformed_img)
        
        transformed_img = transform2(image=image)['image']
        self.transformed_images.append(transformed_img)
        
        transformed_img = transform3(image=image)['image']
        self.transformed_images.append(transformed_img)
        
        transformed_img = transform4(image=image)['image']
        self.transformed_images.append(transformed_img)
        
        return
    
    

if __name__ == '__main__':
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", dest="img_path", type=str, 
                        help="path to the folder containing images")
    parser.add_argument("--bluring", dest="bluring", type=bool, 
                        help="Indicate wether to include bluring in the transormations of not",
                        default=False)
    parser.add_argument("--rotation-angles", dest="rotations_angles", type=list, 
                        help="rotation angles", default=[40, 90, 130, 320])
    parser.add_argument("--bright-limits", dest="brightness_limits", type=list,
                        help="brightness limits in the transormations", 
                        default=[-0.3, 0.1])
    parser.add_argument("--contrast-limits", dest="contrast_limits", type=list,
                        help="contrast limits for the transormations", 
                        default=[-0.3, 0.0])
    
    args = parser.parse_args()
    
    augmentor = ImageAugmentationClass(img_folder=args.img_path)
    print("Starting .... ")
    augmentor.augment(rotations_angles=args.rotations_angles,
                      brightness_limits=args.brightness_limits,
                      contrast_limits=args.contrast_limits)
    
    print("finished ! check images in the corresponding folder" + 
          " : " + args.img_path)
    """



    parser = argparse.ArgumentParser()
    parser.add_argument("--root-folder", dest="root", type=str, 
                        help="root where data folders are stored")
    parser.add_argument("--bluring", dest="bluring", type=bool, 
                        help="Indicate wether to include bluring in the transormations of not",
                        default=True)
    parser.add_argument("--blur-limits", dest="blur_limits", type=float,
                        help="bluring limits", default=15.0)
    parser.add_argument("--rotation-angles", dest="rotations_angles", type=list, 
                        help="rotation angles", default=[40, 90, 130, 320])
    parser.add_argument("--bright-limits", dest="brightness_limits", type=list,
                        help="brightness limits in the transormations", 
                        default=[-0.3, 0.1])
    parser.add_argument("--contrast-limits", dest="contrast_limits", type=list,
                        help="contrast limits for the transormations", 
                        default=[-0.3, 0.0])
    args = parser.parse_args()
    
    folders = os.listdir(args.root)
    for i, folder in enumerate(folders):
        folder = os.path.join(args.root, folder)
        augmentor = ImageAugmentationClass(img_folder=folder)
        #print("Working on foder : ", str(i+1))
        #print(folder)
        augmentor.augment(rotations_angles=args.rotations_angles,
                          brightness_limits=args.brightness_limits,
                          contrast_limits=args.contrast_limits,
                          blur_limits=args.blur_limits)
        
    print("Done ! check images in the folders in : ", args.root)
    

    
    
    
    
    
    
        
    
    
            
    
    
    
    
    
    
    
