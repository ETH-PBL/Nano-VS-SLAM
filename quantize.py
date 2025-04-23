import torch

from torch._export import capture_pre_autograd_graph
from torch.export import export, ExportedProgram

def calibrate(model, data_loader, num_batches=100):
    model.eval()

    with torch.no_grad():
        for i,sample in enumerate(data_loader):
            image = sample['image']
            model(image)
            if i > num_batches:
                break


def quantize_4_executorch(model):
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

    return aten_dialect

def to_executorch(aten_dialect):
    from executorch.exir import ExecutorchBackendConfig, ExecutorchProgramManager
    from executorch.exir.passes import MemoryPlanningPass
    import executorch.exir as exir
    edge_program: exir.EdgeProgramManager = exir.to_edge(aten_dialect)
    executorch_program: exir.ExecutorchProgramManager = edge_program.to_executorch(
        ExecutorchBackendConfig(
            passes=[],  # User-defined passes
        )
    )
    with open("model.pte", "wb") as file:
        file.write(executorch_program.buffer)
    
def quantize(model, dataset_val, backend='fbgemm'):
    torch.backends.quantized.engine = backend
    model.training = False
    model.eval()
    #model.fuse()
    model.qconfig = torch.ao.quantization.get_default_qconfig(backend)


    model_prepared = torch.ao.quantization.prepare(model)
    print("Calibrating model...")
    calibrate(model_prepared, dataset_val)
    model_quantized = torch.ao.quantization.convert(model_prepared)
    return model_quantized

def save(model, out_path):
    torch.jit.save(torch.jit.script(model), out_path)
