目前我们只支持qwen2的全量化

1、为了方便后续非量化，我们首先将QKV/Gate_up_proj进行合并：
    
    python merge_qkv.py

2、如果需要进行smoothquant，请按照如下示例进行

    cd smooth
    python main_smooth.py 
    // 其中smooth_part=['qkv_proj','gate_up_proj']可以控制smooth的范围，可以任意添加o_proj和down_proj，示例:smooth_part=['qkv_proj','gate_up_proj','o_proj','down_proj']

3、navie quantize

    python examples/example_mixtral.py 