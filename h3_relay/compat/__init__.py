"""Version-bounded upstream runtime fixes; no workflow nodes or model policies."""
def install():
    import importlib.metadata
    from comfy.cli_args import args
    if args.cpu:
        return
    try:
        affected = importlib.metadata.version("comfy-kitchen") == "0.2.35"
    except importlib.metadata.PackageNotFoundError:
        return
    if affected:
        import torch
        if torch.cuda.is_available():
            import comfy_kitchen.backends.triton.quantization as upstream
            from . import triton_int8_large as fixed
            upstream._int8_matmul_dequant_kernel = fixed._int8_matmul_dequant_kernel
            upstream._int8_matmul_dequant_per_row_kernel = fixed._int8_matmul_dequant_per_row_kernel
