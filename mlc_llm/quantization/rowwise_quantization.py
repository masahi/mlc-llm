from dataclasses import dataclass
from typing import List, Literal, Optional

from tvm import relax, te, tir, topi
from tvm.script import tir as T

from . import tir_utils
from .quantization import QuantizationSpec
from .quantization import FQuantize, FDequantize


@dataclass
class RowwiseQuantizationSpec(QuantizationSpec):
    """The quantization specification for group quantization algorithm."""

    def get_quantize_func(
        self, _
    ) -> Optional[FQuantize]:
        return encoding_func(
            nbit=4,
            storage_nbit=8,
            transpose=True,
            dtype=self.dtype,
        )

    def get_dequantize_func(
        self,
        param_info: relax.TensorStructInfo,
        qparam_info: List[relax.TensorStructInfo],
    ) -> Optional[FDequantize]:
        return decoding_func(
            nbit=4,
            storage_nbit=8,
            dtype=self.dtype,
        )


# fmt: off
def encoding_func(nbit: int, storage_nbit: int, transpose: bool, dtype: str = "float32") -> FQuantize:
    def te_encode_sym(weight: te.Tensor):
        n_float_per_int = storage_nbit // nbit
        max_int_value = (1 << (nbit - 1)) - 1

        scale_min_shape = (weight.shape[0],)
        k = te.reduce_axis((0, weight.shape[1]), name="k")
        max_abs_value = te.compute(shape=scale_min_shape, fcompute=lambda i: te.max(te.abs(weight[i, k]), axis=k), name="max_abs_value")

        def f_compute_scale(i):
            max_value = tir.max(tir.Cast(dtype, max_abs_value[i]), tir.const(1e-4, dtype))
            return (max_value / tir.const(max_int_value + 1, dtype))

        scale = te.compute(shape=scale_min_shape, fcompute=f_compute_scale, name="scale")
        storage_dtype = ("int" + str(storage_nbit))

        def f_scale_weight(i, j):
            w_scaled = tir.round(tir.Cast(dtype, weight[i, j]) / scale[i])
            w_scaled = T.min(T.max(w_scaled, tir.const(-max_int_value - 1, dtype)), tir.const(max_int_value, dtype)).astype(storage_dtype)
            return w_scaled & tir.const((1 << nbit) - 1, storage_dtype)

        k = te.reduce_axis((0, n_float_per_int), name="k")
        reducer = te.comm_reducer(fcombine=lambda x, y: tir.bitwise_or(x, y), fidentity=lambda dtype: tir.const(0, storage_dtype), name="bitwise_or")
        n_i32 = tir.ceildiv(weight.shape[0], n_float_per_int)

        if transpose:
            w_gathered = te.compute(shape=(weight.shape[1], n_i32), fcompute=lambda j, i: reducer(tir.if_then_else(i * n_float_per_int + k < weight.shape[0], f_scale_weight(i * n_float_per_int + k, j) << (k.astype(storage_dtype) * tir.const(nbit, storage_dtype)), tir.const(0, storage_dtype)), axis=k), name="w_gathered")
        else:
            w_gathered = te.compute(shape=(n_i32, weight.shape[1]), fcompute=lambda i, j: reducer(tir.if_then_else(i * n_float_per_int + k < weight.shape[0], f_scale_weight(i * n_float_per_int + k, j) << (k.astype(storage_dtype) * tir.const(nbit, storage_dtype)), tir.const(0, storage_dtype)), axis=k), name="w_gathered")

        return w_gathered, topi.cast(scale, "float16")

    return te_encode_sym


def _tir_packed_int_to_int_to_float(storage_nbit: int):
    storage_dtype = "int" + str(storage_nbit)

    def f_convert(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str):
        assert val.dtype == storage_dtype
        mask = tir.const((1 << nbit) - 1, "int32")
        unextended = (val >> (pos.astype("int32") * tir.const(nbit, "int32"))) & mask
        return tir.Cast(dtype, (unextended << tir.const(32 - nbit, "int32")) >> tir.const(32 - nbit, "int32"))

    return f_convert


def decoding_func(nbit: int, storage_nbit: int, dtype: str = "float32"):
    def te_decode_sym(data, scale):
        n_float_per_int = storage_nbit // nbit
        def f_decode_sym(i, j):
            f_convert = _tir_packed_int_to_int_to_float(storage_nbit)
            data_float = f_convert(nbit, data[i, j // n_float_per_int], j % n_float_per_int, dtype="float16")
            scale_float = scale[j]
            return data_float * scale_float

        shape = (data.shape[0], data.shape[1] * n_float_per_int)
        w = te.compute(shape=shape, fcompute=f_decode_sym, name="decode")
        return w

    return te_decode_sym

# fmt: on
