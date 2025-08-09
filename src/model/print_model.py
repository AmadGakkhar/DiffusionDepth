import sys
import os
# Add the src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.diffusion_dcbase_model import Diffusion_DCbase_Model

def create_dummy_args():
    """Create dummy arguments for model initialization"""
    class Args:
        def __init__(self):
            self.backbone_name = "mmbev_res18"
            self.backbone_module = "mmbev_resnet"
            self.head_specify = "DDIMDepthEstimate_Res"
            self.inference_steps = 20
            self.num_train_timesteps = 1000
            self.model_name = "Diffusion_DCbase_Model"
            self.network = "resnet18"
            self.prop_kernel = 3
            self.affinity = "TGASS"
            self.conf_prop = True
    return Args()

if __name__ == "__main__":
    args = create_dummy_args()
    model = Diffusion_DCbase_Model(args)
    print(model)
