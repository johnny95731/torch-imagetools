# torch-imagetools

An image processing library based on PyTorch that provides

- Basic: edge filters, wavelet decomposition, color space
  transformations, etc.
- Advanced: white balance, color transfer.

This library aim to provide some non deep learning-based tools for pre-/post-processing, data augmentation, or an alternative option when no data for training network.

Sphinx API page: https://johnny95731.github.io/torch-imagetools/

### Future plan

More implementations of the researches. For examples
- low-light image enhance
- highlight removal and shadow removal
- style transfer
- denoise
- chromatic adaptation
- frequency domain method
- image processing based on differential equations.

## Installation

1. Clone this repository
```shell
pip clone https://github.com/johnny95731/torch-imagetools.git
cd torch-imagetools
pip install -e .
```

2. Install PyTorch >= 2.7.0. Since the dependency will keep covering the
torch+cuda, torch is not in the dependencies of `pyproject.toml`. 

### Optional

Install PyWavelets if you want to use the submodule `imgtools.wavelets`
```shell
pip install pywavelets
```

## Usage

- The library assume that the RGB image is in the range of [0, 1].
- The functions will try to convert dtype and device to fit the first argument.
- The tests only test float32 and float64 dtypes. Some features in `torch.linalg` does not support float16.
- The image format should be `CHW`.
- The docstring indicate the tensor shape and the asterisk `*` means an optional
place. For example, the shape `(*, C, H, W)` allows `(C, H, W)` or `(B, C, H, W)`.

```python
import torch

import imgtools

img = torch.rand(3, 512, 512)
# Hist equalization
enhanced = imgtools.enhance.hist_equalize(img)
# Color transfer
enhanced = imgtools.enhance.transfer_reinhard(img)
# Color space transformation
xyz = imgtools.color.rgb_to_xyz(img)
```

