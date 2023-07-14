"""Relax quantization passes."""

from typing import List

import tvm
from tvm import relax, te, tir, topi
from tvm.ir.module import IRModule
from tvm.relax.analysis import remove_all_unused
from tvm.relax.expr_functor import PyExprMutator, mutator
from tvm.relax.op.builtin import stop_lift_params
from tvm.script import tir as T
from .quantization import (
    decoding_after_taking_func,
    encoding_func as groupwise_encoding_func,
)


# fmt: off
def _tir_packed_int_to_int_to_float(storage_nbit: int):
    storage_dtype = "int" + str(storage_nbit)

    def f_convert(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str):
        assert val.dtype == storage_dtype
        mask = tir.const((1 << nbit) - 1, "int32")
        unextended = (val >> (pos.astype("int32") * tir.const(nbit, "int32"))) & mask
        return tir.Cast(dtype, (unextended << tir.const(32 - nbit, "int32")) >> tir.const(32 - nbit, "int32"))

    return f_convert


def encoding_func(nbit: int, storage_nbit: int, dtype: str = "float32"):
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
            if n_float_per_int == 1:
                return w_scaled
            return w_scaled & tir.const((1 << nbit) - 1, storage_dtype)

        n_i32 = tir.ceildiv(weight.shape[0], n_float_per_int)

        if n_float_per_int == 1:
            w_gathered = te.compute(shape=(weight.shape[1], n_i32), fcompute=lambda j, i: f_scale_weight(i, j), name="w_gathered")
        else:
            k = te.reduce_axis((0, n_float_per_int), name="k")
            reducer = te.comm_reducer(fcombine=lambda x, y: tir.bitwise_or(x, y), fidentity=lambda dtype: tir.const(0, storage_dtype), name="bitwise_or")
            w_gathered = te.compute(shape=(weight.shape[1], n_i32), fcompute=lambda j, i: reducer(tir.if_then_else(i * n_float_per_int + k < weight.shape[0], f_scale_weight(i * n_float_per_int + k, j) << (k.astype(storage_dtype) * tir.const(nbit, storage_dtype)), tir.const(0, storage_dtype)), axis=k), name="w_gathered")

        return w_gathered, topi.cast(scale, "float16")

    return te_encode_sym


def decoding_func(nbit: int, storage_nbit: int, dtype: str = "float32"):
    def te_decode_sym(data, scale):
        n_float_per_int = storage_nbit // nbit
        def f_decode_sym(i, j):
            if n_float_per_int == 1:
                data_float = tir.Cast("float16", data[i, j])
            else:
                f_convert = _tir_packed_int_to_int_to_float(storage_nbit)
                data_float = f_convert(nbit, data[i, j // n_float_per_int], j % n_float_per_int, dtype="float16")

            scale_float = scale[j]
            return data_float * scale_float

        shape = (data.shape[0], data.shape[1] * n_float_per_int)
        w = te.compute(shape=shape, fcompute=f_decode_sym, name="decode")
        return w

    return te_decode_sym


# fmt: on


@tvm.transform.module_pass(opt_level=0, name="RowWiseQuantize")
class RowWiseQuantize:
    def __init__(
        self,
        nbit: int = 4,
        dtype: str = "float32",
    ) -> None:
        self.nbit = nbit
        self.dtype = dtype

    def transform_module(self, mod: IRModule, _) -> IRModule:
        @mutator
        class QuantizeMutator(PyExprMutator):
            def __init__(
                self,
                mod: IRModule,
                nbit: int,
                dtype: str,
            ):
                super().__init__(mod)
                self.mod = mod
                self._params = set()
                self.nbit = nbit
                self.storage_nbit = 8
                self.dtype = dtype

            def transform(self) -> IRModule:
                for global_var, func in self.mod.functions.items():
                    if not isinstance(func, relax.Function):
                        continue
                    if not "num_input" in func.attrs:
                        continue
                    num_inputs = func.attrs["num_input"]
                    for i in range(int(num_inputs), len(func.params)):
                        self._params.add(func.params[i])
                    updated_func = self.visit_expr(func)
                    updated_func = remove_all_unused(updated_func)
                    self.builder_.update_func(global_var, updated_func)
                return self.builder_.get()

            def emit_encoding(self, x: relax.Expr) -> List[relax.Expr]:
                encoded_data = self.builder_.emit_te(
                    encoding_func(
                        self.nbit,
                        self.storage_nbit,
                        dtype=self.dtype,
                    ),
                    x,
                    primfunc_name_hint="encode",
                )

                packed_weight = self.builder_.normalize(encoded_data[0])
                encoded_weight = relax.call_pure_packed(
                    "cutlass.ft_preprocess_weight",
                    packed_weight,
                    80,
                    self.nbit == 4,
                    sinfo_args=packed_weight.struct_info,
                )

                decode_args = []
                decode_args.append(self.builder_.emit(encoded_weight))
                decode_args.append(
                    self.builder_.emit(relax.TupleGetItem(encoded_data, 1))
                )
                for i, arg in enumerate(decode_args):
                    decode_args[i] = self.builder_.emit(stop_lift_params(arg))
                return decode_args

            def quantize_matmul(self, call: relax.Call):
                call_arg = self.lookup_binding(call.args[1])
                if call.struct_info.dtype == "float32":
                    return call

                def emit(weight):
                    decode_args = self.emit_encoding(weight)

                    quantized_permute_dims = self.builder_.call_te(
                        decoding_func(
                            self.nbit,
                            self.storage_nbit,
                            dtype=self.dtype,
                        ),
                        *decode_args,
                        primfunc_name_hint="decode"
                    )
                    return relax.op.matmul(
                        call.args[0],
                        quantized_permute_dims,
                        out_dtype=call.attrs.out_dtype,
                    )

                if call_arg.op == tvm.ir.Op.get("relax.permute_dims"):
                    if (
                        call_arg.attrs.axes is not None
                        or call_arg.args[0].struct_info.ndim != 2
                    ):
                        return call

                    return emit(call_arg.args[0])

                if call_arg.op == tvm.ir.Op.get("relax.concat"):
                    if call_arg.attrs.axis != 1 or call_arg.struct_info.ndim != 2:
                        return call

                    encode_arg = self.builder_.normalize(
                        relax.op.permute_dims(call_arg)
                    )
                    return emit(encode_arg)

                return call

            def emit_groupwise_encoding(self, x: relax.Expr) -> List[relax.Expr]:
                encoded_data = self.builder_.emit_te(
                    groupwise_encoding_func(
                        True,  # sym
                        32,  # group size, hardcoded
                        self.nbit,
                        "int{}".format(self.nbit),  # mode,
                        self.storage_nbit,
                        transpose=False,
                        dtype="float16",
                    ),
                    x,
                    primfunc_name_hint="encode",
                )

                decode_args = []
                decode_args.append(
                    self.builder_.emit(relax.TupleGetItem(encoded_data, 0))
                )
                decode_args.append(
                    self.builder_.emit(relax.TupleGetItem(encoded_data, 1))
                )

                for i, arg in enumerate(decode_args):
                    decode_args[i] = self.builder_.emit(stop_lift_params(arg))
                return decode_args

            def quantize_take(self, call: relax.Call):
                if (
                    call.attrs.axis is not None
                    and call.attrs.axis.value != 0
                    or call.args[0].struct_info.ndim != 2
                    or call.args[0] not in self._params
                ):
                    return call

                decode_args = self.emit_groupwise_encoding(call.args[0])
                decode_args += (call.args[1],)
                return self.builder_.call_te(
                    decoding_after_taking_func(
                        True,  # sym
                        32,  # group size, hardcoded
                        self.nbit,
                        "int{}".format(self.nbit),  # mode
                        self.storage_nbit,
                        call.args[0].struct_info.shape[-1],
                        dtype="float16",
                    ),
                    *decode_args,
                    primfunc_name_hint="take_decode"
                )

            def visit_call_(self, call):
                call = self.visit_expr_post_order(call)

                if call.op == tvm.ir.Op.get("relax.matmul"):
                    return self.quantize_matmul(call)
                elif call.op == tvm.ir.Op.get("relax.take"):
                    return self.quantize_take(call)
                else:
                    return call

        return QuantizeMutator(mod, self.nbit, self.dtype).transform()
