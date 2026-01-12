Recommended to use Pythin veriosn >3.9

1. Dataset is present in dataset/data
The images are in RGB and code converts then to grayscale, and resizes them to 256*256. 

Python dependencies to install( likely):
pip install torch torchvision torchaudio        
pip install torchmetrics                          
pip install scikit-learn                       
pip install matplotlib                     
pip install tqdm                           
pip install pillow     
                        
2.Key default settings:

Image size: 256×256
Batch size: 10
Optimizer: Adam (β1=0.5, β2=0.999)
Learning rate (Generator): 2e-4
Learning rate (Discriminator): 1e-4
Epochs: 250
L1 loss weight (λ): 100
Loss type: LSGAN (MSE) + L1

3.Hyperparameter Tuning Suggestions
Batch Size
Current: 10
Increase if GPU memory allows → may stabilize gradients
Decrease if OOM errors occur
Learning Rates
Current: 2e-4 (G), 1e-4 (D)
Lower LR for D helps avoid overpowering G
Try 1e-4 for both if instability occurs
PatchGAN Receptive Field
Current: 94×94
Smaller (70×70) reduces overhead; larger (>94) can improve texture but increases compute time
Loss Weights
L1 weight (λ): 100 (balances realism vs structure)
Increase to prioritize structure, decrease to focus on fine details
Augmentation Strength
Adjust brightness/contrast augmentation to increase robustness
Perceptual Loss (Optional)
Adding VGG-based perceptual loss can improve realism but doubles training time