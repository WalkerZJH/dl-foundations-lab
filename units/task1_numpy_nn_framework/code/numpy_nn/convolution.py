from __future__ import annotations

import numpy as np

from .module import Module, Parameter


def _output_size(size: int, kernel: int, padding: int, stride: int) -> int:
    numerator = size + 2 * padding - kernel
    if numerator < 0 or numerator % stride != 0:
        raise ValueError(
            f"Invalid convolution shape: size={size}, kernel={kernel}, "
            f"padding={padding}, stride={stride}"
        )
    return numerator // stride + 1


def _im2col_indices(
    shape: tuple[int, int, int, int], kernel_h: int, kernel_w: int, padding: int, stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    _, channels, height, width = shape
    out_h = _output_size(height, kernel_h, padding, stride)
    out_w = _output_size(width, kernel_w, padding, stride)

    row_offsets = np.repeat(np.arange(kernel_h), kernel_w)
    row_offsets = np.tile(row_offsets, channels)
    col_offsets = np.tile(np.arange(kernel_w), kernel_h * channels)
    channel_indices = np.repeat(np.arange(channels), kernel_h * kernel_w).reshape(-1, 1)

    output_rows = stride * np.repeat(np.arange(out_h), out_w)
    output_cols = stride * np.tile(np.arange(out_w), out_h)
    rows = row_offsets.reshape(-1, 1) + output_rows.reshape(1, -1)
    cols = col_offsets.reshape(-1, 1) + output_cols.reshape(1, -1)
    return channel_indices, rows, cols, out_h, out_w


def im2col(
    inputs: np.ndarray, kernel_h: int, kernel_w: int, padding: int, stride: int
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, int, int]]:
    indices = _im2col_indices(inputs.shape, kernel_h, kernel_w, padding, stride)
    channels, rows, cols, _, _ = indices
    padded = np.pad(
        inputs,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
    )
    columns = padded[:, channels, rows, cols]
    columns = columns.transpose(1, 2, 0).reshape(kernel_h * kernel_w * inputs.shape[1], -1)
    return columns, indices


def col2im(
    columns: np.ndarray,
    shape: tuple[int, int, int, int],
    kernel_h: int,
    kernel_w: int,
    padding: int,
    stride: int,
    indices: tuple[np.ndarray, np.ndarray, np.ndarray, int, int],
) -> np.ndarray:
    batch_size, channels_count, height, width = shape
    channel_indices, rows, cols, out_h, out_w = indices
    padded = np.zeros(
        (batch_size, channels_count, height + 2 * padding, width + 2 * padding),
        dtype=columns.dtype,
    )
    reshaped = columns.reshape(channels_count * kernel_h * kernel_w, out_h * out_w, batch_size)
    reshaped = reshaped.transpose(2, 0, 1)
    np.add.at(padded, (slice(None), channel_indices, rows, cols), reshaped)
    if padding == 0:
        return padded
    return padded[:, :, padding:-padding, padding:-padding]


class _Conv2DBase(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        rng: np.random.Generator,
        stride: int = 1,
        padding: int = 0,
        dtype: np.dtype = np.dtype(np.float64),
    ) -> None:
        super().__init__()
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weight = Parameter.from_array(
            rng.normal(
                0.0,
                scale,
                (out_channels, in_channels, kernel_size, kernel_size),
            ),
            dtype=dtype,
        )
        self.bias = Parameter.from_array(np.zeros(out_channels, dtype=dtype))
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding


class Conv2DNaive(_Conv2DBase):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._inputs: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._inputs = inputs
        batch_size, _, height, width = inputs.shape
        filters = self.weight.data.shape[0]
        out_h = _output_size(height, self.kernel_size, self.padding, self.stride)
        out_w = _output_size(width, self.kernel_size, self.padding, self.stride)
        padded = np.pad(
            inputs,
            ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
        )
        output = np.empty((batch_size, filters, out_h, out_w), dtype=inputs.dtype)
        for n in range(batch_size):
            for f in range(filters):
                for row in range(out_h):
                    h_start = row * self.stride
                    for col in range(out_w):
                        w_start = col * self.stride
                        window = padded[
                            n,
                            :,
                            h_start : h_start + self.kernel_size,
                            w_start : w_start + self.kernel_size,
                        ]
                        output[n, f, row, col] = (
                            np.sum(window * self.weight.data[f]) + self.bias.data[f]
                        )
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._inputs is None:
            raise RuntimeError("Conv2DNaive.backward requires a preceding forward call")
        inputs = self._inputs
        padded = np.pad(
            inputs,
            ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
        )
        grad_padded = np.zeros_like(padded)
        self.weight.grad.fill(0.0)
        self.bias.grad[...] = grad_output.sum(axis=(0, 2, 3))
        for n in range(inputs.shape[0]):
            for f in range(self.weight.data.shape[0]):
                for row in range(grad_output.shape[2]):
                    h_start = row * self.stride
                    for col in range(grad_output.shape[3]):
                        w_start = col * self.stride
                        value = grad_output[n, f, row, col]
                        window = padded[
                            n,
                            :,
                            h_start : h_start + self.kernel_size,
                            w_start : w_start + self.kernel_size,
                        ]
                        self.weight.grad[f] += window * value
                        grad_padded[
                            n,
                            :,
                            h_start : h_start + self.kernel_size,
                            w_start : w_start + self.kernel_size,
                        ] += self.weight.data[f] * value
        if self.padding == 0:
            return grad_padded
        return grad_padded[:, :, self.padding : -self.padding, self.padding : -self.padding]


class Conv2DVectorized(_Conv2DBase):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._cache: tuple[
            tuple[int, int, int, int],
            np.ndarray,
            tuple[np.ndarray, np.ndarray, np.ndarray, int, int],
        ] | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        columns, indices = im2col(
            inputs, self.kernel_size, self.kernel_size, self.padding, self.stride
        )
        _, _, _, out_h, out_w = indices
        result = self.weight.data.reshape(self.weight.data.shape[0], -1) @ columns
        result += self.bias.data.reshape(-1, 1)
        output = result.reshape(self.weight.data.shape[0], out_h, out_w, inputs.shape[0])
        self._cache = (inputs.shape, columns, indices)
        return output.transpose(3, 0, 1, 2)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("Conv2DVectorized.backward requires a preceding forward call")
        input_shape, columns, indices = self._cache
        flattened_grad = grad_output.transpose(1, 2, 3, 0).reshape(
            self.weight.data.shape[0], -1
        )
        self.bias.grad[...] = flattened_grad.sum(axis=1)
        self.weight.grad[...] = (flattened_grad @ columns.T).reshape(self.weight.data.shape)
        grad_columns = self.weight.data.reshape(self.weight.data.shape[0], -1).T @ flattened_grad
        return col2im(
            grad_columns,
            input_shape,
            self.kernel_size,
            self.kernel_size,
            self.padding,
            self.stride,
            indices,
        )


class MaxPool2D(Module):
    def __init__(self, pool_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        self._cache: tuple[tuple[int, ...], np.ndarray] | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        batch_size, channels, height, width = inputs.shape
        out_h = _output_size(height, self.pool_size, 0, self.stride)
        out_w = _output_size(width, self.pool_size, 0, self.stride)
        output = np.empty((batch_size, channels, out_h, out_w), dtype=inputs.dtype)
        maxima = np.empty((batch_size, channels, out_h, out_w), dtype=np.int64)
        for row in range(out_h):
            h_start = row * self.stride
            for col in range(out_w):
                w_start = col * self.stride
                window = inputs[
                    :,
                    :,
                    h_start : h_start + self.pool_size,
                    w_start : w_start + self.pool_size,
                ].reshape(batch_size, channels, -1)
                maxima[:, :, row, col] = np.argmax(window, axis=2)
                output[:, :, row, col] = np.max(window, axis=2)
        self._cache = (inputs.shape, maxima)
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("MaxPool2D.backward requires a preceding forward call")
        input_shape, maxima = self._cache
        grad_inputs = np.zeros(input_shape, dtype=grad_output.dtype)
        batch_indices = np.arange(input_shape[0])[:, None]
        channel_indices = np.arange(input_shape[1])[None, :]
        for row in range(grad_output.shape[2]):
            h_start = row * self.stride
            for col in range(grad_output.shape[3]):
                w_start = col * self.stride
                index = maxima[:, :, row, col]
                row_offset = index // self.pool_size
                col_offset = index % self.pool_size
                grad_inputs[
                    batch_indices,
                    channel_indices,
                    h_start + row_offset,
                    w_start + col_offset,
                ] += grad_output[:, :, row, col]
        return grad_inputs
