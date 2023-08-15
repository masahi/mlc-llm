import numpy as np

import tvm
from tvm import relax
from tvm.relax.backend.contrib.cutlass import partition_for_cutlass
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T
from tvm import dlight as dl

from mlc_llm import utils


artifact_path = "dist/vicuna-v1-7b-q4f16_ft/"


def build():
    @I.ir_module
    class Module:
        @T.prim_func
        def split_rotary(
            var_A: T.handle,
            cos: T.Buffer((2048, 128), "float16"),
            sin: T.Buffer((2048, 128), "float16"),
            var_positions: T.handle,
            var_T_split: T.handle,
            var_T_split_1: T.handle,
            var_T_split_2: T.handle,
        ):
            T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
            # with T.block("root")
            num_tokens = T.int64()
            A = T.match_buffer(var_A, (num_tokens, T.int64(12288)), "float16")
            positions = T.match_buffer(var_positions, (num_tokens,), "int64")
            T_split = T.match_buffer(var_T_split, (num_tokens, T.int64(4096)), "float16")
            T_split_1 = T.match_buffer(var_T_split_1, (num_tokens, T.int64(4096)), "float16")
            T_split_2 = T.match_buffer(var_T_split_2, (num_tokens, T.int64(4096)), "float16")

            for ax0, ax1 in T.grid(num_tokens, T.int64(4096)):
                with T.block("T_split"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(
                        A[v_ax0, v_ax1],
                        A[v_ax0, v_ax1 + T.int64(4096)],
                        A[v_ax0, v_ax1 + T.int64(8192)],
                        positions[v_ax0],
                    )
                    T.writes(
                        T_split[v_ax0, v_ax1], T_split_1[v_ax0, v_ax1], T_split_2[v_ax0, v_ax1]
                    )
                    T_split[v_ax0, v_ax1] = cos[positions[v_ax0], v_ax1 % T.int64(128)] * A[
                        v_ax0, v_ax1
                    ] + sin[positions[v_ax0], v_ax1 % T.int64(128)] * T.Select(
                        T.int64(64) <= v_ax1 % T.int64(128),
                        A[v_ax0, v_ax1 - T.int64(64)],
                        A[v_ax0, v_ax1 + T.int64(64)] * T.float16(-1),
                    )
                    T_split_1[v_ax0, v_ax1] = cos[positions[v_ax0], v_ax1 % T.int64(128)] * A[
                        v_ax0, v_ax1 + T.int64(4096)
                    ] + sin[positions[v_ax0], v_ax1 % T.int64(128)] * T.Select(
                        T.int64(64) <= v_ax1 % T.int64(128),
                        A[v_ax0, v_ax1 + T.int64(4096) - T.int64(64)],
                        A[v_ax0, v_ax1 + T.int64(4096) + T.int64(64)] * T.float16(-1),
                    )
                    T_split_2[v_ax0, v_ax1] = A[v_ax0, v_ax1 + T.int64(8192)]

        @T.prim_func(private=True)
        def decode_qkv(
            A: T.Buffer((T.int64(4096), T.int64(6144)), "int8"),
            B: T.Buffer((T.int64(12288),), "float16"),
            T_decode: T.Buffer((T.int64(4096), T.int64(12288)), "float16"),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            for i, j in T.grid(T.int64(4096), T.int64(12288)):
                with T.block("decode"):
                    v_i, v_j = T.axis.remap("SS", [i, j])
                    T.reads(A[v_i, v_j // T.int64(2)], B[v_j])
                    T.writes(T_decode[v_i, v_j])
                    T_decode[v_i, v_j] = (
                        T.Cast(
                            "float16",
                            T.shift_right(
                                T.shift_left(
                                    T.bitwise_and(
                                        T.shift_right(
                                            T.Cast("int32", A[v_i, v_j // T.int64(2)]),
                                            T.Cast("int32", v_j % T.int64(2)) * 4,
                                        ),
                                        15,
                                    ),
                                    28,
                                ),
                                28,
                            ),
                        )
                        * B[v_j]
                    )

        @T.prim_func(private=True)
        def decode_o_proj(
            A: T.Buffer((T.int64(4096), T.int64(2048)), "int8"),
            B: T.Buffer((T.int64(4096),), "float16"),
            T_decode: T.Buffer((T.int64(4096), T.int64(4096)), "float16"),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            for i, j in T.grid(T.int64(4096), T.int64(4096)):
                with T.block("decode"):
                    v_i, v_j = T.axis.remap("SS", [i, j])
                    T.reads(A[v_i, v_j // T.int64(2)], B[v_j])
                    T.writes(T_decode[v_i, v_j])
                    T_decode[v_i, v_j] = (
                        T.Cast(
                            "float16",
                            T.shift_right(
                                T.shift_left(
                                    T.bitwise_and(
                                        T.shift_right(
                                            T.Cast("int32", A[v_i, v_j // T.int64(2)]),
                                            T.Cast("int32", v_j % T.int64(2)) * 4,
                                        ),
                                        15,
                                    ),
                                    28,
                                ),
                                28,
                            ),
                        )
                        * B[v_j]
                    )

        @R.function
        def main_qkv_rotary(
            x: R.Tensor(("num_tokens", 4096), dtype="float16"),
            weight: R.Tensor((4096, 6144), dtype="int8"),
            scales: R.Tensor((12288,), dtype="float16"),
            cos: R.Tensor((2048, 128), "float16"),
            sin: R.Tensor((2048, 128), "float16"),
            positions: R.Tensor(("num_tokens",), "int64"),
        ) -> R.Tuple(
            [
                R.Tensor(("num_tokens", 4096), dtype="float16"),
                R.Tensor(("num_tokens", 4096), dtype="float16"),
                R.Tensor(("num_tokens", 4096), dtype="float16"),
            ]
        ):
            R.func_attr({"num_input": 3})
            cls = Module
            num_tokens = T.int64()
            with R.dataflow():
                lv6 = R.call_tir(
                    cls.decode_qkv,
                    (weight, scales),
                    out_sinfo=R.Tensor((4096, 12288), dtype="float16"),
                )
                lv1_1: R.Tensor((num_tokens, 12288), dtype="float16") = R.matmul(
                    x, lv6, out_dtype="float16"
                )
                out = R.call_tir(
                    cls.split_rotary,
                    (lv1_1, cos, sin, positions),
                    out_sinfo=[
                        R.Tensor((num_tokens, 4096), dtype="float16"),
                        R.Tensor((num_tokens, 4096), dtype="float16"),
                        R.Tensor((num_tokens, 4096), dtype="float16"),
                    ],
                )
                R.output(out)
            return out

        @R.function
        def main_o_proj(
            x: R.Tensor(("num_tokens", 4096), dtype="float16"),
            weight: R.Tensor((4096, 2048), dtype="int8"),
            scales: R.Tensor((4096,), dtype="float16"),
            residual: R.Tensor(("num_tokens", 4096), dtype="float16"),
        ) -> R.Tensor(("num_tokens", 4096), dtype="float16"):
            R.func_attr({"num_input": 3})
            cls = Module
            num_tokens = T.int64()
            with R.dataflow():
                lv6 = R.call_tir(
                    cls.decode_o_proj,
                    (weight, scales),
                    out_sinfo=R.Tensor((4096, 4096), dtype="float16"),
                )
                lv1_1: R.Tensor((num_tokens, 4096), dtype="float16") = R.matmul(
                    x, lv6, out_dtype="float16"
                )
                out = R.add(lv1_1, residual)
                R.output(out)
            return out

    mod = partition_for_cutlass(Module)

    mod = relax.transform.RunCodegen(
        {"cutlass": {"sm": 80, "find_first_valid": False}},
        ["main_qkv_rotary", "main_o_proj"],
    )(mod)

    target = tvm.target.Target("nvidia/nvidia-a10g")

    with target:
        mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)

    ex = relax.build(mod, target=target)

    ex.export_library(artifact_path + "llama_qkv_ft.so")

    print(mod)


def test():
    tvm_ex = tvm.runtime.load_module(artifact_path + "llama_qkv_ft.so")
    dev = tvm.device("cuda", 0)
    vm = relax.VirtualMachine(tvm_ex, dev)
    const_params = utils.load_params(artifact_path, dev)

    batch_size = 16
    hidden_dim = 4096

    inp_np = np.random.randn(batch_size, hidden_dim).astype("float16")
    positions_np = np.ones(
        batch_size,
    ).astype("int64")

    inp = tvm.nd.array(inp_np, dev)
    positions = tvm.nd.array(positions_np, dev)
    cos = const_params[-2]
    sin = const_params[-1]

    param_idx = 2

    for i in range(32):
        print(i)
        weight_qkv = const_params[param_idx]
        scale_qkv = const_params[param_idx + 1]
        weight_o_proj = const_params[param_idx + 2]
        scale_o_proj = const_params[param_idx + 3]
        param_idx += 10

        q, k, v = vm["main_qkv_rotary"](inp, weight_qkv, scale_qkv, cos, sin, positions)
        print(q.numpy(), k.numpy(), v.numpy())
        out = vm["main_o_proj"](inp, weight_o_proj, scale_o_proj, inp).numpy()


build()
