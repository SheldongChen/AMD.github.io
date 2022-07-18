# Interpreter Project of FastReID

FastReID is a research platform that implements state-of-the-art re-identification algorithms. It is a groud-up rewrite of the previous version, [reid strong baseline](https://github.com/michuanhaohao/reid-strong-baseline).

Our Interpreter Project is based on FastReID

## What's New
- [Oct 2021] Quick Start of Interpreter Project.
- [Sep 2020] Automatic Mixed Precision training is supported with pytorch1.6 built-in `torch.cuda.amp`. Set `cfg.SOLVER.AMP_ENABLED=True` to switch it on.
- [Aug 2020] [Model Distillation](https://github.com/JDAI-CV/fast-reid/tree/master/projects/DistillReID) is supported, thanks for [guan'an wang](https://github.com/wangguanan)'s contribution.
- [Aug 2020] ONNX/TensorRT converter is supported.
- [Jul 2020] Distributed training with multiple GPUs, it trains much faster.
- [Jul 2020] `MAX_ITER` in config means `epoch`, it will auto scale to maximum iterations.
- Includes more features such as circle loss, abundant visualization methods and evaluation metrics, SoTA results on conventional, cross-domain, partial and vehicle re-id, testing on multi-datasets simultaneously, etc.
- Can be used as a library to support [different projects](https://github.com/JDAI-CV/fast-reid/tree/master/projects) on top of it. We'll open source more research projects in this way.
- Remove [ignite](https://github.com/pytorch/ignite)(a high-level library) dependency and powered by [PyTorch](https://pytorch.org/).

We write a [chinese blog](https://l1aoxingyu.github.io/blogpages/reid/2020/05/29/fastreid.html) about this toolbox.

## Installation

See [INSTALL.md](https://github.com/JDAI-CV/fast-reid/blob/master/docs/INSTALL.md) and [INSTALL_for_interpreter.md](./docs/INSTALL_for_interpreter.md).


## Quick Start 

The designed architecture follows this guide [PyTorch-Project-Template](https://github.com/L1aoXingyu/PyTorch-Project-Template), you can check each folder's purpose by yourself.

See [GETTING_STARTED.md](https://github.com/JDAI-CV/fast-reid/blob/master/docs/GETTING_STARTED.md).

Learn more at out [documentation](). And see [projects/](https://github.com/JDAI-CV/fast-reid/tree/master/projects) for some projects that are build on top of fastreid.

## Requirements of Computer Hardware 

GPU>=16GB memory

RAM>=128GB memory

## Quick Start of Interpreter Project

Step 0, modify the PATH in the shell script to suit your computer.


Step 1, train ReID model:
```bash
bash shell/Market1501/0_train_market_bagtricks_euclidean.sh
```

Ps: [release model on Market1501](https://github.com/SheldongChen/AMD.github.io/releases/download/model/market_circle_r50_ip.pth)

Step 2, train Interpreter model:
```bash
bash shell/Market1501/1_train_market_interpretation.sh
```
## Attention of Interpreter Project
In order to solve the requirements of demo, after several update iterations, the code of this version adds some functions, backbone for demo and Loss for specific datasets on the basis of the framework of our paper.

For the slightly inconsistent parts between this code and paper description, please feel free to use, it does not affect the final performance.

## Samples of Visualization

<div align="center">
  <img src="https://github.com/SheldongChen/AMD.github.io/blob/main/projects/InterpretationReID/mask_samples/positive_att_mask.jpg" width="800px"/><br>
    <p style="font-size:1.5vw;">Positive Mask</p>
</div>


<div align="center">
  <img src="https://github.com/SheldongChen/AMD.github.io/blob/main/projects/InterpretationReID/mask_samples/negative_att_mask.jpg" width="800px"/><br>
    <p style="font-size:1.5vw;">Negative Mask</p>
</div>

<div align="center">
  <img src="https://github.com/SheldongChen/AMD.github.io/blob/main/projects/InterpretationReID/masked_img_samples/0060_c3s1_007726_00_jpgnotonly_hot_map.jpg" width="800px"/><br>
    <p style="font-size:1.5vw;">Person Sample</p>
</div>

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [Fastreid Model Zoo](https://github.com/JDAI-CV/fast-reid/blob/master/docs/MODEL_ZOO.md).

## Deployment

We provide some examples and scripts to convert fastreid model to Caffe, ONNX and TensorRT format in [Fastreid deploy](https://github.com/JDAI-CV/fast-reid/blob/master/tools/deploy).

## License

Fastreid is released under the [Apache 2.0 license](https://github.com/JDAI-CV/fast-reid/blob/master/LICENSE).

## Citing Fastreid

If you use Fastreid in your research or wish to refer to the baseline results published in the Model Zoo, please use the following BibTeX entry.

```BibTeX
@inproceedings{chen2021AMD,
  title={Explainable Person Re-Identification with Attribute-guided Metric Distillation},
  author={Chen, Liu and Liu, Zhang and Zhang, Mei},
  booktitle={IEEE International Conference on Computer Vision (ICCV)},
  year={2021}
}

@article{he2020fastreid,
  title={FastReID: A Pytorch Toolbox for General Instance Re-identification},
  author={He, Lingxiao and Liao, Xingyu and Liu, Wu and Liu, Xinchen and Cheng, Peng and Mei, Tao},
  journal={arXiv preprint arXiv:2006.02631},
  year={2020}
}
```
