import PIL, torch
import numpy as np

def load_images(IMG_PATH):
    img_batch = torch.empty((0,3,224,224))
    for img in sorted(IMG_PATH.iterdir()):
        img = PIL.Image.open(img)
        img = process_imgs(img)
        img_batch = torch.cat((img_batch,torch.Tensor(img).unsqueeze(0)))
    img_batch = img_batch
    return img_batch

def load_labels(IMG_PATH):
    labels = []
    for i,f in enumerate(list(sorted(IMG_PATH.iterdir()))):
        labels.append(int(f.name[-6]))
    labels = torch.Tensor(labels)
    return labels
    
def load_et_maps(ETM_PATH):
    etm_batch = torch.empty((0,1,224,224))
    for mask in sorted(ETM_PATH.iterdir()):
        etm = process_maps(PIL.Image.open(mask))
        etm_batch = torch.cat((etm_batch,torch.Tensor(etm).unsqueeze(0)))
    return etm_batch


classes = {
    0:'fish',
    1:'dog',
    2:'cassette player',
    3:'chainsaw',
    4:'church',
    5:'music instrument',
    6:'garbage truck',
    7:'gas',
    8:'golfball',
    9:'parachute',
}

def process_imgs(img):
    img = img.resize((224,224))
    x = np.asarray(img)
    if len(x.shape) != 3:
        x = np.expand_dims(x,2)
        x = np.concatenate((x,x,x),2)
    x = np.transpose(x, (2,0,1))/255
    return x

def process_maps(etm):
    etm = etm.resize((224,224))
    x = np.asarray(etm)
    x = np.transpose(x, (2,0,1))[3]/255
    return np.expand_dims(x, axis=0)