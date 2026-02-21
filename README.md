# torch-imagetools

An image processing library based on PyTorch that aim to provide some non deep learning-based tools for pre-/post-processing, data augmentation, or an alternative option when no data for training the network.

- `imgtools.color`: transformations between color spaces. The spaces including RGB, YUV, HSV, CIEXYZ, LMS, CIELAB, etc.
- `imgtools.balance`: chromatic adaptations, illuminant estimations, correlate color temperature estimations.
- `imgtools.enhance`: basic intensity trnasform (linear, log, gamma, ...), sharpening/unsharp masking, high dynamic range, low-light enhance.
- `imgtools.filters`: spatial domain and frequency domain filters.
- `imgtools.wavelets`: discrete wavelet transform and its inverse transform.
- `imgtools.utils`: conversion `np.ndarray` <-> `torch.Tensor` (`arrayize`, `tensorize`), math tools (PCA, filter2d, matrix transform).

This library 

- Docs: https://johnny95731.github.io/torch-imagetools/

### Future plan

More implementations of the researches. For examples
- low-light image enhance
- highlight removal and shadow removal
- style transfer
- denoise
- chromatic adaptation
- frequency domain methods
- gradient domain methods
- statistics tools.

## Installation

1. Clone this repository
```shell
pip clone https://github.com/johnny95731/torch-imagetools.git
cd torch-imagetools
pip install -e .
```

2. Install PyTorch >= 2.7.0. Since the dependency will keep covering the
torch+cuda, torch is not in the dependencies of `pyproject.toml`. 

### Optional Dependencies

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
import imgtools

hwc_img = cv2.imread(path)
chw_img = imgtools.utils.tensorize(hwc_img)

# Hist equalization
enhanced = imgtools.enhance.hist_equalize(chw_img)
# Color transfer
enhanced = imgtools.enhance.transfer_reinhard(img)
# Color space transformation
xyz = imgtools.color.rgb_to_xyz(img)
```

## License

See [LICENSE.md](LICENSE.md)