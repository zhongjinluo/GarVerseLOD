import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torch
from lib.options import BaseOptions
from lib.mesh_util import *
from lib.sample_util import *
from lib.train_util import *
from lib.data import *
from lib.model import *
from lib.geometry import index

# get options
opt = BaseOptions().parse()

def test(opt):
    # set cuda
    cuda = torch.device('cuda:%d' % opt.gpu_id)

    test_dataset = TestDataset_Refine(opt, phase='test')
    test_dataset.is_train = False

    projection_mode = "orthogonal"

    # create net
    netG = HGPIFuNetwNML(opt, projection_mode).to(device=cuda)

    def set_eval():
        netG.eval()


    model_path = '../support_data/checkpoints/fine_garment.pth'
    netG.load_state_dict(torch.load(model_path))

    os.makedirs(opt.results_path, exist_ok=True)
        
    with torch.no_grad():
        set_eval()
        for i in range(0, len(test_dataset)):
            test_data = test_dataset[i]
            save_path = '%s/%s.obj' % (
                opt.results_path, test_data['name'].replace('/','_'))
            print(test_data['name'], "garment")
            gen_mesh(opt, netG, cuda, test_data, save_path)
            order_list = ["garment", "bottom", "right", "left", "top"]
            for oi in range(len(order_list)):
                if oi == 0:
                    continue
                print(test_data['name'], "garment-boundary", order_list[oi])
                gen_mesh_boundary(opt, netG, cuda, test_data, save_path, (order_list[oi], oi))
                
if __name__ == '__main__':
    test(opt)
