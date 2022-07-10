# Pytorch implementation of the "WorldModels" using Attention mechanism

Forked from https://github.com/ctallec/world-models

Paper: Ha and Schmidhuber, "World Models", 2018. https://doi.org/10.48550/arXiv.1803.10122. 


## Prerequisites

The implementation is based on Python3 and PyTorch, check their website [here](https://pytorch.org) for installation instructions. The rest of the requirements is included in the [requirements file](requirements.txt), to install them:
```bash
pip3 install -r requirements.txt
```

## Running the worldmodels

The model is composed of three parts:

  1. A Variational Auto-Encoder (VAE), whose task is to compress the input images into a compact latent representation.
  2. A Mixture-Density Recurrent Network (MDN-RNN) with or without attention, trained to predict the latent encoding of the next frame given past latent encodings and actions.
  3. A linear Controller (C), which takes both the latent encoding of the current frame, and the hidden state of the MDN-RNN given past latents and actions as input and outputs an action. It is trained to maximize the cumulated reward using the Covariance-Matrix Adaptation Evolution-Strategy ([CMA-ES](http://www.cmap.polytechnique.fr/~nikolaus.hansen/cmaartic.pdf)) from the `cma` python package.

In the given code, all three sections are trained separately, using the scripts `trainvae.py`, `trainmdrnn.py` and `traincontroller.py` for model without attention and the scripts 'trainmdrnn_attention.py' and 'traincontroller_attention.py' for the model with attention.

Training scripts take as argument:
* **--logdir** : The directory in which the models will be stored. If the logdir specified already exists, it loads the old model and continues the training.
* **--noreload** : If you want to override a model in *logdir* instead of reloading it, add this option.

### 1. Data generation
Before launching the VAE and MDN-RNN training scripts, you need to generate a dataset of random rollouts and place it in the `datasets/carracing` folder.

Data generation is handled through the `data/generation_script.py` script, e.g.
```bash
python data/generation_script.py --rollouts 1000 --rootdir datasets/carracing --threads 8
```

Rollouts are generated using a *brownian* random policy, instead of the *white noise* random `action_space.sample()` policy from gym, providing more consistent rollouts.

### 2. Training the VAE
The VAE is trained using the `trainvae.py` file, e.g.
```bash
python trainvae.py --logdir exp_dir
```

### 3. Training the MDN-RNN
The MDN-RNN is trained using the `trainmdrnn.py` file, e.g.
```bash
python trainmdrnn.py --logdir exp_dir
```
To train MDN-RNN with attention, `trainmdrnn_attention.py` is used
```bash
python trainmdrnn_attention.py --logdir exp_dir
```
A VAE must have been trained in the same `exp_dir` for this script to work.
### 4. Training and testing the Controller
Finally, the controller (if MDN-RNN is trained without attention) is trained using CMA-ES, e.g.
```bash
python traincontroller.py --logdir exp_dir --n-samples 4 --pop-size 4 --target-return 950 --display --max-workers 11
```
<B>Note:</B> Before you run the above code make sure to make the following changes:

You can test the obtained policy with `test_controller.py` e.g.
```bash
python test_controller.py --logdir exp_dir
```
If MDN-RNN is trained using attention, 'traincontroller_attention.py' should be used.
```bash
python traincontroller_attention.py --logdir exp_dir --n-samples 4 --pop-size 4 --target-return 950 --display --max-workers 11
```
You can test the obtained poicy with 'test_controller.py` but you will need to change 'ctrl' to 'ctrl_attn' in line 11 from 'test_controller.py' 

### Notes
When running on a headless server, you will need to use `xvfb-run` to launch the controller training script. For instance,
```bash
xvfb-run -s "-screen 0 1400x900x24" python traincontroller.py --logdir exp_dir --n-samples 4 --pop-size 4 --target-return 950 --display
```
If you do not have a display available and you launch `traincontroller` without
`xvfb-run`, the script will fail silently (but logs are available in
`logdir/tmp (or tmp_attn)`).

Be aware that `traincontroller` requires heavy gpu memory usage when launched
on gpus. To reduce the memory load, you can directly modify the maximum number
of workers by specifying the `--max-workers` argument.

If you have several GPUs available, `traincontroller` will take advantage of
all gpus specified by `CUDA_VISIBLE_DEVICES`.
