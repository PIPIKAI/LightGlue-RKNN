import onnx
from collections import Counter

def analyze_onnx_model(onnx_path):
    """完整分析 ONNX 模型"""
    model = onnx.load(onnx_path)
    
    print("=" * 60)
    print(f"📄 模型文件: {onnx_path}")
    print("=" * 60)
    
    # 1. Opset 信息
    print("\n🔧 Opset 版本:")
    for opset in model.opset_import:
        domain = opset.domain if opset.domain else "ai.onnx (default)"
        print(f"  {domain}: v{opset.version}")
    
    # 2. 输入输出信息
    print("\n📥 输入:")
    for inp in model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: {shape}")
    
    print("\n📤 输出:")
    for out in model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: {shape}")
    
    # 3. 算子统计
    op_types = [node.op_type for node in model.graph.node]
    op_counter = Counter(op_types)
    
    print(f"\n📊 模型统计:")
    print(f"  总节点数: {len(model.graph.node)}")
    print(f"  算子种类: {len(op_counter)}")
    
    print(f"\n📋 算子详情 (共 {len(op_counter)} 种):")
    for op_type, count in op_counter.most_common():
        print(f"  {op_type:30s} : {count:4d}")
    
    # 4. 检查问题算子
    problematic_ops = {
        "LayerNormalization": "可能不兼容低版本 opset",
        "Attention": "需要 opset 14+",
        "GridSample": "某些推理引擎不支持",
    }
    
    print("\n⚠️  潜在兼容性问题:")
    found_issues = False
    for op, issue in problematic_ops.items():
        if op in op_counter:
            print(f"  ⚠️  {op} ({op_counter[op]}次): {issue}")
            found_issues = True
    
    if not found_issues:
        print("  ✅ 未发现已知问题算子")
    
    print("=" * 60)

# 使用
analyze_onnx_model("output/onnx/superpoint_lightglue_end2end_simplified.onnx")