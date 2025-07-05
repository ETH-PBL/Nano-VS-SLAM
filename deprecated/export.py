from torch._export import capture_pre_autograd_graph
from torch.export import export, ExportedProgram
from executorch.exir import EdgeProgramManager, to_edge
from executorch.exir import ExecutorchBackendConfig, ExecutorchProgramManager
from executorch.exir.passes import MemoryPlanningPass
import torch
from kp2dtiny.models.kp2dtiny import KP2DTinyV2,  KP2D_TINY_V2, KP2D_TINY_ATT_V2
import torch

model = KP2DTinyV2(**KP2D_TINY_ATT_V2, nClasses=28)

model = model.cpu()
model.eval()

example_args = (torch.randn(1, 3, 256, 256),)
pre_autograd_aten_dialect = capture_pre_autograd_graph(model, example_args)
print("Pre-Autograd ATen Dialect Graph")
print(pre_autograd_aten_dialect)

from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)

quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
prepared_graph = prepare_pt2e(pre_autograd_aten_dialect, quantizer)
# calibrate with a sample dataset
converted_graph = convert_pt2e(prepared_graph)
print("Quantized Graph")
print(converted_graph)

aten_dialect: ExportedProgram = export(converted_graph, example_args)
print("ATen Dialect Graph")
print(aten_dialect)


edge_program: EdgeProgramManager = to_edge(aten_dialect)
print("Edge Dialect Graph")
print(edge_program.exported_program())


executorch_program: ExecutorchProgramManager = edge_program.to_executorch(
    ExecutorchBackendConfig(
        passes=[],  # User-defined passes
    )
)

with open("model.pte", "wb") as file:
    file.write(executorch_program.buffer)