import torch


def rgb_to_hsi(rgb: torch.Tensor) -> torch.Tensor:
    """Converts an image from RGB space to HSI space.

    The input is assumed to be in the range of [0, 1].

    Parameters
    ----------
    rgb : torch.Tensor
        An RGB image in the range of [0, 1] with shape (*, 3, H, W).

    Returns
    -------
    torch.Tensor
        An image in HSI space with shape (*, 3, H, W). The H channel values
        are in the range [0, 360), S and L are in the range of [0, 1].
    """
    amax, argmax_rgb = torch.max(rgb, dim=-3)
    amin = torch.min(rgb, dim=-3).values
    delta = amax - amin

    r, g, b = torch.unbind(rgb, dim=-3)

    h1 = (g - b).divide_(delta)
    h2 = (b - r).divide_(delta).add_(2.0)
    h3 = (r - g).divide_(delta).add_(4.0)

    hue = torch.stack((h1, h2, h3), dim=-3)
    hue = torch.gather(hue, dim=-3, index=argmax_rgb.unsqueeze(-3)).squeeze(-3)
    hue = hue.remainder_(6.0).mul_(60.0).nan_to_num_(0.0, 0.0, 0.0)

    intensity = r.add(g).add_(b).mul_(1 / 3.0)
    sat = intensity.sub(amin).divide_(intensity)
    torch.nan_to_num(sat, 0.0, 0.0, 1.0, out=sat)

    hsi = torch.stack((hue, sat, intensity), dim=-3)
    return hsi


def hsi_to_rgb(hsi: torch.Tensor) -> torch.Tensor:
    """Converts an image from HSI space to RGB space.

    Parameters
    ----------
    hsi : torch.Tensor
        An image in HSI space with shape (*, 3, H, W).

    Returns
    -------
    torch.Tensor
        An RGB image in the range of [0, 1] with the shape (*, 3, H, W).
    """
    h, s, i = hsi.unbind(-3)

    h_prime = (h / 60).remainder_(6.0)
    z = 1 - h_prime.remainder(2.0).sub_(1.0).abs_()
    m = (1 - s).mul_(i)
    c = i.mul(3.0).mul_(s).divide_(1.0 + z)
    x = c.mul(z).add_(m)
    c.add_(m)

    h_idx = h_prime.long().unsqueeze(-3)
    r = torch.gather(torch.stack((c, x, m, m, x, c), dim=-3), -3, h_idx)
    g = torch.gather(torch.stack((x, c, c, x, m, m), dim=-3), -3, h_idx)
    b = torch.gather(torch.stack((m, m, x, c, c, x), dim=-3), -3, h_idx)
    rgb = torch.cat((r, g, b), dim=-3)
    return rgb


if __name__ == '__main__':
    from timeit import timeit

    img = torch.randint(0, 256, (16, 3, 512, 512)).type(torch.float32) / 255
    num = 10

    hsl = rgb_to_hsi(img)
    ret = hsi_to_rgb(hsl)

    d = torch.abs(ret - img)
    print(torch.max(d))

    print(timeit('rgb_to_hsi(img)', number=num, globals=locals()))
    print(timeit('hsi_to_rgb(hsl)', number=num, globals=locals()))
