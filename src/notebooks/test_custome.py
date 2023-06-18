import torch
from torch.utils.data import Dataset, DataLoader
from vgg_16 import Network
from tqdm import tqdm
from torch.nn.functional import one_hot
from svhn_custome_dataset import SVHN_Dataset
import sys

ckpt_name = "./models/VGG16_0.0001_SVHN_True.pt"
n = len(sys.argv)

if n == 2 :  
  name_input = sys.argv[1]
  ckpt_name = name_input



def get_num_correct(p, l):
  compared = p + l
  corrects = torch.sum(compared==2, axis=1)
  corrects_sum = torch.sum(corrects).item()
  return corrects_sum

def get_preds_from_outputs(outputs, num_images):
  preds = []
  for i in range(len(num_images)):
    n = num_images[i].item()
    tk = torch.topk(outputs[i], n)[1]
    tk_oh = one_hot(tk,  num_classes=10)
    tk_oh_sum = torch.sum(tk_oh, dim=0).tolist()
    preds.append(tk_oh_sum)

  preds = torch.tensor(preds, device=device)
  return preds

x = SVHN_Dataset('org_data/test/digitStruct.mat','org_data/test')
batch_size = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loader_test = DataLoader(x, batch_size=batch_size, shuffle=True, num_workers=0)
PATH = ckpt_name
checkpoint = torch.load(PATH, map_location=torch.device(device))
network = Network().to(device)
network.load_state_dict(checkpoint['model_state_dict'])
num_correct = 0
batch_count = 0
n_all_images = 0
accuracy = 0
network.eval()
    
with torch.no_grad():
  for batch in tqdm(loader_test, 'Test on Real SVHN'):
    batch_count += 1
    images = batch['image']
    labels = batch['labels']
    # labels = torch.permute(batch['labels'], (1, 0))
    images = images.to(device).float()
    labels = labels.to(device)
    num_images =torch.sum(labels, dim=1)
    n_all_images += torch.sum(num_images).item()
    [kasami_preds, outputs] = network(images)
    
    preds = get_preds_from_outputs(outputs, num_images)
    num_correct_now = get_num_correct(preds,labels)
    num_correct += num_correct_now
accuracy = num_correct / (n_all_images)
print("Test Accuarcy of Multi-Label", PATH, "is : ", accuracy * 100, "% no Overlap")

