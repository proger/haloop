"""
Parallel prefix scan

Up-down algorithm illustration: https://www.cs.princeton.edu/courses/archive/fall13/cos326/lec/23-parallel-scan.pdf
Up-down algorithm and recurrence reductions by Guy Blelloch: https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf

See also:
Generic Functional Parallel Algorithms: Scan and FFT by Conal Elliott
http://conal.net/papers/generic-parallel-functional/generic-parallel-functional.pdf
"""
import torch
import math
import torch.nn.functional as F


def pad_to_power_of_2(xs: torch.Tensor):
    width = xs.shape[-1]
    rounded_width = 2 ** round(math.log2(width))

    xs = F.pad(xs, (0, rounded_width - width), value=0)
    return xs


def scan(xs: torch.Tensor):
    """
    Solve the following recurrence relation in parallel:
    ys[0] = xs[0]
    ys[i] = xs[i] + ys[i-1]
    """
    width = int(math.log2(xs.shape[-1]))

    # up
    levels = [xs]
    for _level in range(width):
        xs = xs.view(-1, 2)
        left, right = xs[..., 0], xs[..., 1]
        xs = left + right
        #print(_level, xs)
        levels.append(xs)

    # down
    level = levels.pop()
    from_left = torch.zeros_like(level) # root starts with zero
    while levels:
        children = levels.pop()
        left_children = children.view(-1, 2)[:, 0]
        from_left = torch.stack([from_left, from_left + left_children], dim=-1).view(-1)

    return from_left + children


def scanrec(w: torch.Tensor, b: torch.Tensor):
    """
    Solve the following first order recurrence relation in parallel:
    y[0] = b[0]
    y[i] = b[i] + w[i] * y[i-1]

    It works by redefining (+) in scan as:
    (wl, yl) + (wr, yr) = (wl * wr, yl * wr + yr)
    """
    width = int(math.log2(w.shape[-1]))

    # up
    up_w, up_y = w, b
    levels = [(up_w, up_y)]
    for _level in range(width):
        up_w, up_y = up_w.view(-1, 2), up_y.view(-1, 2)

        up_w_left, up_w_right = up_w[..., 0], up_w[..., 1]
        up_w = up_w_left * up_w_right

        up_y_left, up_y_right = up_y[..., 0], up_y[..., 1]
        up_y = up_y_left * up_w_right + up_y_right

        levels.append((up_w, up_y))

    # down
    up_w, up_y = levels.pop()
    from_left_y = torch.zeros_like(up_y) # root starts with zero
    while levels:
        up_w, up_y = levels.pop()
        up_w_left, up_y_left = up_w.view(-1, 2)[:, 0], up_y.view(-1, 2)[:, 0]
        from_left_y = torch.stack([from_left_y, from_left_y * up_w_left + up_y_left], dim=-1).view(-1)

    return from_left_y * up_w + up_y


def scanrec_log(w: torch.Tensor, b: torch.Tensor):
    """
    Solve the following first order recurrence relation in parallel:
    y[0] = b[0]
    y[i] = b[i] `logaddexp` (w[i] + y[i-1])

    It works by redefining (+) in scan as:
    (wl, yl) `logaddexp` (wr, yr) = (wl + wr, (yl + wr) `logaddexp` yr)
    """
    N, T = w.shape
    width = int(math.log2(T))

    # up
    up_w, up_y = w, b
    levels = [(up_w, up_y)]
    for _level in range(width):
        up_w, up_y = up_w.view(N, -1, 2), up_y.view(N, -1, 2)

        up_w_left, up_w_right = up_w[..., 0], up_w[..., 1]
        up_w = up_w_left + up_w_right

        up_y_left, up_y_right = up_y[..., 0], up_y[..., 1]
        up_y = (up_y_left + up_w_right).logaddexp(up_y_right)

        levels.append((up_w, up_y))

    # down
    up_w, up_y = levels.pop()
    from_left_y = torch.full_like(up_y, -10000) # root starts with -inf
    while levels:
        up_w, up_y = levels.pop()

        up_w_left, up_y_left = up_w.view(N, -1, 2)[..., 0], up_y.view(N, -1, 2)[..., 0]
        from_left_y = torch.stack([
            from_left_y,
            (from_left_y + up_w_left).logaddexp(up_y_left)
        ], dim=-1).view(N, -1)

    return (from_left_y + up_w).logaddexp(up_y)


def scanrec_sequential(w: torch.Tensor, b: torch.Tensor):
    """
    Solve the following first order recurrence relation sequentially:
    y[0] = b[0]
    y[i] = b[i] + w[i] * y[i-1]
    """
    N, T = w.shape
    ys = torch.zeros_like(w)
    ys[..., 0] = b[..., 0]
    for i in range(1, T):
        ys[..., i] = b[..., i] + w[..., i] * ys[..., i-1]
        #print(i, ys[..., i], '=', b[..., i], '+', w[..., i], '*', ys[..., i-1])
    return ys


def scanrec_log_sequential(w: torch.Tensor, b: torch.Tensor):
    """
    Solve the following first order recurrence relation sequentially:
    y[0] = b[0]
    y[i] = b[i] `logaddexp` (w[i] + y[i-1])
    """
    width = w.shape[-1]
    ys = torch.zeros_like(w)
    ys[..., 0] = b[..., 0]
    for i in range(1, width):
        ys[..., i] = b[..., i].logaddexp(w[..., i] + ys[..., i-1])
        #print(i, ys[..., i], '=', b[..., i], '`logaddexp`', w[..., i], '+', ys[..., i-1])
    return ys


def test_scan():
    torch.manual_seed(1337)

    xs = torch.tensor([6,4,16,10,16,14,2,8])
    ys = scan(xs)
    print(ys, ys.shape)
    assert torch.equal(scan(xs), scanrec(torch.ones_like(xs), xs))

    for _ in range(100):
        xs = torch.randint(0, 65535, (1, 16384))
        #xs = torch.randint(0, 65535, (1, 32))
        width = xs.shape[-1]
        xs = pad_to_power_of_2(xs)
        ys1 = scan(xs)[:width]
        ys2 = torch.cumsum(xs, dim=-1)
        ys3 = scanrec(torch.ones_like(xs), xs)[:width]

        assert torch.equal(ys1, ys2)
        assert torch.equal(ys2, ys3)


@torch.inference_mode()
def test_scanrec():
    torch.manual_seed(1337)

    for _ in range(20):
        w = torch.randint(1, 65535, (1, 16384))
        b = torch.randint(1, 65535, (1, 16384))
        width = w.shape[-1]
        w = pad_to_power_of_2(w)
        b = pad_to_power_of_2(b)
        ys1 = scanrec_sequential(w, b)[:width]
        ys2 = scanrec(w, b)[:width]

        assert torch.equal(ys1, ys2)


@torch.inference_mode()
def test_scanrec_log():
    torch.manual_seed(1337)

    for _ in range(20):
        #w = torch.randn(1, 65535).abs()
        #b = torch.randn(1, 65535).abs()

        w = torch.randn(1, 32).abs()
        b = torch.randn(1, 32).abs()

        width = w.shape[-1]
        w = pad_to_power_of_2(w)
        b = pad_to_power_of_2(b)
        ys1 = scanrec_log_sequential(w, b)[:, :width]
        ys2 = scanrec_log(w, b)[:, :width]

        #print('ys1', ys1)
        #print('ys2', ys2)

        assert torch.allclose(ys1, ys2)

@torch.inference_mode()
def _test_scanrec_log_batch():
    torch.manual_seed(1337)

    w = torch.randn(7, 64).abs()
    b = torch.randn(7, 64).abs()

    ys1 = scanrec_log_sequential(w, b)
    ys2 = scanrec_log(w, b)

    assert torch.allclose(ys1, ys2)


def test_pad():
    torch.manual_seed(1337)

    assert pad_to_power_of_2(torch.randn(1, 126)).shape[-1] == 128


if __name__ == '__main__':
    import pytest
    pytest.main(["--no-header", "-v", "-s", "--durations", "0", __file__])
