# Copyright © 2026 Huawei Technologies Co., Ltd.

import torch


def profiling_test(fn_triton, device_type, device):
    if device_type == 'npu':
        import torch_npu

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            l2_cache=False
        )

        with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.NPU],
                with_stack=False,
                record_shapes=False,
                profile_memory=False,
                schedule=torch_npu.profiler.schedule(wait=1,
                                                     warmup=1,
                                                     active=10,
                                                     repeat=1,
                                                     skip_first=1),
                experimental_config=experimental_config,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result_dir")
        ) as prof:
            for i in range(13):
                fn_triton()
                torch.npu.synchronize()
                prof.step()

    elif device_type == 'cuda':
        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=False,
                record_shapes=False,
                profile_memory=False,
                schedule=torch.profiler.schedule(wait=1,
                                                 warmup=1,
                                                 active=10,
                                                 repeat=1,
                                                 skip_first=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler("./result_dir")
        ) as prof:
            for i in range(13):
                fn_triton()
                torch.cuda.synchronize()
                prof.step()

        print("\n========== GPU 性能统计 ==========")
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

    else:
        print("警告：未检测到 GPU 或 NPU，跳过性能 profiling")

    print(f"性能 profiling 完成！")
