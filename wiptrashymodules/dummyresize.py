'''                       
Resize script adapted from Thung and Yang's trashnet (https://github.com/garythung/trashnet)              
Accepts images from an input folder, resizes them to dimensions specified in trashnet constants.py,   
and outputs them to a destination folder in subfolders by class.
'''

import os
import dummyresize_constants
import numpy as np
from scipy import misc, ndimage
import imageio
import imageio.v2 as iio
from skimage.transform import resize
from skimage import img_as_ubyte
from pathlib import Path
from tqdm import tqdm

def resize_(image, dim1, dim2):
    return resize(image, (dim1, dim2))


def fileWalk(directory, destPath):
    try:
        os.makedirs(destPath)
    except OSError:
        if not os.path.isdir(destPath):
            raise

    for subdir, dirs, files in os.walk(directory):
        for file in tqdm(files):
            if len(file) <= 4 or file[-4:] == '.jpeg':
                continue

            pic = iio.imread(os.path.join(subdir, file))
            dim1 = len(pic)
            dim2 = len(pic[0])
            if dim1 > dim2:
                pic = np.rot90(pic)
                
            picResized = resize_(pic, dummyresize_constants.DIM1, dummyresize_constants.DIM2)
            write_image_imageio(os.path.join(destPath, file), img_as_ubyte(picResized))
            
def write_image_imageio(img_file, img):
    #img = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    #kwargs = {}
    #if os.path.splitext(img_file)[1].lower() in [".jpg", ".jpeg"]:
    if img.ndim >= 3 and img.shape[2] > 3:
        img = img[:,:,:3]
    #kwargs["quality"] = quality
    #kwargs["subsampling"] = 0
    iio.imwrite(img_file, img)
            
# def read_image_imageio(img_file):
#     img = imageio.v2.imread(img_file)
#     img = np.asarray(img).astype(np.float32)
#     if len(img.shape) == 2:
#         img = img[:,:,np.newaxis]
#     return img / 255.0

            
def main():

    parentPath = os.path.dirname(os.getcwd())

    prepath = Path("/Users/rohitmaheshwari/Documents/GitHub/TrashBox-VGG19_model/dummydata")
    batteryDir = os.path.join(prepath, 'battery')
    beveragecansDir = os.path.join(prepath, 'beverage cans')
    breadDir = os.path.join(prepath, 'bread')
    cardboardDir = os.path.join(prepath, 'cardboard')
    cartonDir = os.path.join(prepath, 'cartons')
    cigaretteButtDir = os.path.join(prepath, 'cigarette butt')
    constructionDir = os.path.join(prepath, 'construction scrap')
    crockeryDir = os.path.join(prepath, 'crockery')
    diapersDir = os.path.join(prepath, 'diapers')
    electronicDir = os.path.join(prepath, 'electronic device')
    ewasteDir = os.path.join(prepath, 'ewaste')
    glassDir = os.path.join(prepath, 'glass')
    glassbottlesDir = os.path.join(prepath, 'glass bottles')
    glovesDir = os.path.join(prepath, 'gloves')
    leafletsDir = os.path.join(prepath, 'leaflets')
    leftoversDir = os.path.join(prepath, 'leftovers')
    lightbublDir = os.path.join(prepath, 'lightbulb')
    masksDir = os.path.join(prepath, 'masks')
    medicinesDir = os.path.join(prepath, 'medicines')
    metalContainersDir = os.path.join(prepath, 'metal containers')
    newspaperDir = os.path.join(prepath, 'news paper')
    paperDir = os.path.join(prepath, 'paper')
    paperCupsDir = os.path.join(prepath, 'paper cups')
    pensDir = os.path.join(prepath, 'pens')
    plasticBagsDir = os.path.join(prepath, 'plastic bags')
    plasticBottlesDir = os.path.join(prepath, 'plastic bottles')
    plasticContainersDir = os.path.join(prepath, 'plastic containers')
    plasticCupsDir = os.path.join(prepath, 'plastic cups')
    smallAppliancesDir = os.path.join(prepath, 'small appliances')
    spraycansDir = os.path.join(prepath, 'spray cans')
    syringeDir = os.path.join(prepath, 'syringe')
    tetraPakDir = os.path.join(prepath, 'tetra pak')
    tissueDir = os.path.join(prepath, 'tissuenapkin')
    
            

    destPath = os.path.join(parentPath, '../../GitHub/TrashBox-VGG19_model/dummydataresized')

    try:
        os.makedirs(destPath)

    except OSError:
        if not os.path.isdir(destPath):
            raise


    #GLASSBOTTLES-GlassBins
    fileWalk(glassbottlesDir, os.path.join(destPath, 'glass bottles'))

    #CARDBOARDPAPER-BlueorGreenBin
    fileWalk(cardboardDir, os.path.join(destPath, 'cardboard'))
    fileWalk(cartonDir, os.path.join(destPath, 'cartons'))
    fileWalk(newspaperDir, os.path.join(destPath, 'news paper'))
    fileWalk(paperDir, os.path.join(destPath, 'paper'))
    fileWalk(leafletsDir, os.path.join(destPath, 'leaflets'))
    
    #EWASTE-HardwareStoreSupermarket
    fileWalk(batteryDir, os.path.join(destPath, 'battery'))
    fileWalk(electronicDir, os.path.join(destPath, 'electronic device'))
    fileWalk(ewasteDir, os.path.join(destPath, 'ewaste'))
    fileWalk(smallAppliancesDir, os.path.join(destPath, 'small appliances'))
    
    #RESIDUAL-GreyorBlackBin
    fileWalk(cigaretteButtDir, os.path.join(destPath, 'cigarette butt'))
    fileWalk(diapersDir, os.path.join(destPath, 'diapers'))
    fileWalk(lightbublDir, os.path.join(destPath, 'lightbulb'))
    fileWalk(glassDir, os.path.join(destPath, 'glass'))
    fileWalk(tissueDir, os.path.join(destPath, 'tissuenapkin'))
    
     
    #MEDICAL-GreyorBlackBin
    fileWalk(glovesDir, os.path.join(destPath, 'gloves'))
    fileWalk(masksDir, os.path.join(destPath, 'masks'))
    fileWalk(medicinesDir, os.path.join(destPath, 'medicines'))
    fileWalk(syringeDir, os.path.join(destPath, 'syringe'))
    
    #TRASH-BrownBin
    fileWalk(breadsDir, os.path.join(destPath, 'bread'))
    fileWalk(leftoversDir, os.path.join(destPath, 'leftovers'))
    
    #METAL-YellowBag
    fileWalk(beverageCansDir, os.path.join(destPath, 'beverage cans'))
    fileWalk(constructionDir, os.path.join(destPath, 'construction scrap'))
    fileWalk(metalContainersDir, os.path.join(destPath, 'metal containers'))
    fileWalk(spraycansDir, os.path.join(destPath, 'spray cans'))
    
    #PLASTIC-YellowBag
    fileWalk(crockeryDir, os.path.join(destPath, 'crockery'))
    fileWalk(paperCupsDir, os.path.join(destPath, 'paper cups'))
    fileWalk(pensDir, os.path.join(destPath, 'pens'))
    fileWalk(plasticBagsDir, os.path.join(destPath, 'plastic bags'))
    fileWalk(plasticBottlesDir, os.path.join(destPath, 'plastic bottles'))
    fileWalk(plasticContainersDir, os.path.join(destPath, 'plastic containers'))
    fileWalk(plasticCupsDir, os.path.join(destPath, 'plastic cups'))
    fileWalk(tetraPakDir, os.path.join(destPath, 'tetra pak'))



if __name__ == '__main__':
    main()
