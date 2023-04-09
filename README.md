# Stochastic Optimization of 3D Non-Cartesian Sampling Trajectory (SNOPY)

### Introduction

Optimizing 3D k-space sampling trajectories is important for efficient MRI yet presents a challenging computational problem. This work proposes a generalized framework for optimizing 3D non-Cartesian sampling patterns via data-driven optimization (the code also supports 2D imaging.)

The methods can optimize various properties of a sampling trajectory, like the gradient waveform (experiment 3.2.1 in the paper) and the rotation angles (experiment 3.2.2). Users can also optimize properties of their own trajectories, such as the "density" of spiral trajectories. Feel free to contact me (guanhuaw@umich.edu) for any questions.

### Dependence

An anaconda environment file called `snopy.yml`. You can install it using Conda. Some packages, including MIRTorch, require manual installation from Github.

MIRTorch provides convenient abstraction of system matrices and common equations. Its demos are also helpful for beginners.

The code structure inherits [CycleGAN and pix2pix in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), which is highly modular and use script to pass arguments. You can find more detailed guidance and Q&A in that repo.

### How to tailor/optimize my own sampling trajectory together with reconstruction algorithm?

#### Data requirements and Data-loader

In the exemplary `calgary.py`, we assume that the data pair is stored in `.h5` archives. An test case is [here](https://www.dropbox.com/s/7ycnabypgr2epg4/e14089s3_P53248.7.h5?dl=0). It is also straightforward to create your own dataloader.

The dataset should be categorized into three subfolders, `train`, `test`, and `val`. Pass `--dataroot` to specify the directory.

#### Sampling trajectory initialization and Parameterization

As suggested, SNOPY supports various parameterization approaches, and users can define their own parameterization methods as long as the trajectory is differentiable w.r.t. the parameters of interest.

In `SamplingLayerBspline3D` class of `network.py`, we optimize the gradient waveforms of arbitrary 3D non-Cartesian sampling trajectories. The initialization trajectory can be formulated as a `.npy` file and passed via `--init_traj` command.

In `SOS` class of `network.py`, we optimize the rotation angles of a stack-of-stars trajectory.

#### Reconstructor

Here we provide two reconstruction algorithms. The first has analytical regularizers/priors and use iterative algorithms, such as CG-SENSE and QPLS. The second is unrolled neural network, such as MoDL, which has learnable parameters can be jointly optimized with sampling trajectories.

#### Jacobian approximation

/models/mirtorch_pkg.py implements the Jacobian operators required for optimizing sampling trajectories. The theory in described in Ref [2]. The forward mode, adjoint model, frame operator, and the inverse $(A'A+\lambda I)^{-1}$ are provided. The implementation use the MRI system matrix provided by [MIRTorch](https://github.com/guanhuaw/MIRTorch).

### Visualizer

It is cool see in how the trajectory evolves on a browser, hence the package uses Visdom as the visualizer. You may define a free port with command --display_port (default is 8097, as in basic_options.py). Then you could run `visdom --port xxxx` to start the visdom service (on a local or remote server). For detailed usage please refer to [CycleGAN and pix2pix in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

Note that the current visualizer is written for 3D. For optimization, you may want to disable the visualizer by setting `--displays-id` to -1.

### Passing variables

A simple optimization project, the experiment 3.2.1 on the SNOPY paper, is provided in the script. The code use parser to pass arguments. Definition are in the options folder. Some crucial arguments are listed below:

- --loss_xxxx: control the weighting of loss functions. 
- --which_model_netG: which parameterization strategy is used for sampling trajectory. For example,  `SamplingLayerBspline3D` for spline-parameterized optimization. It is also straightforward to define new strategies, such as optimizing the rotation angles of radial trajectories.
- ReconVSTraj: Ratio of updates for recon neural network and sampling trajectory.

### Parallel training

The current codes are for a single GPU (otherwise students of my lab may get mad at me), and it is straightforward to package the code under a `torch.nn.DataParallel` module to support multi-GPU training.

### Commercial usage

We welcome researchers to use SNOPY to optimize protocols in research settings. For commercial usage, the intellectual property and related patents belong to the Regents of University of Michigan. Please consult [Guanhua Wang](guanhuaw@umich.edu) for details.

### References:

If you use  codes here, please cite:

```
@article{wang:22:bjork,
  author={Wang, Guanhua and Luo, Tianrui and Nielsen, Jon-Fredrik and Noll, Douglas C. and Fessler, Jeffrey A.},
  journal={IEEE Transactions on Medical Imaging}, 
  title={B-spline Parameterized Joint Optimization of Reconstruction and K-space Trajectories (BJORK) for Accelerated 2D MRI}, 
  year={2022},
  pages={1-1},
  doi={10.1109/TMI.2022.3161875}}
```

```
@article{wang:23:eao,
  author = {Wang, Guanhua and Fessler, Jeffrey A.},
  journal = {IEEE Transactions on Computational Imaging},
  title = {Efficient Approximation of Jacobian Matrices Involving a Non-Uniform Fast Fourier Transform (NUFFT)},
  year = {2023},
  pages = {1--12},
  doi = {10.1109/TCI.2023.3240081}
}
```

```
@article{wang:22:SNOPY,
  title={Stochastic optimization of 3D non-cartesian sampling trajectory (SNOPY)},
  author={Wang, Guanhua and Nielsen, Jon-Fredrik and Fessler, Jeffrey A. and Noll, Douglas C.},
  journal={arXiv preprint arXiv:2209.11030},
  year={2022}
}
```