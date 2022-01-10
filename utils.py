import torch
def batch_accuracy(model,x,y): return round(((model(x).argmax(1)==y).sum()/x.shape[0]).item(), 4)

def get_targets(model, img_batch, gt_labels):
    targets = list()
    logits = model(img_batch)
    for i,pred in enumerate(logits):
        if pred.argmax()==gt_labels[i]:
            targets.append(pred.argsort(descending=True)[1])
        else: targets.append(pred.argmax())
    return(torch.Tensor(targets))