import onnx
from onnxruntime.quantization import quantize_static, QuantType
from onnxruntime import quantization
import torch
import torch
from torch.utils.data import Dataset
model_fp32 = './checkpoints/KP2Dtiny_STM_q.onnx'
model_quant = './checkpoints/KP2Dtiny_STM_q.quant.onnx'

class RandomNoiseDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        noise = torch.randn(1, 3, 120, 160)
        return noise

calib_ds = RandomNoiseDataset(size=100)


class QuntizationDataReader(quantization.CalibrationDataReader):
    def __init__(self, torch_ds, batch_size, input_name):

        self.torch_dl = torch.utils.data.DataLoader(torch_ds, batch_size=batch_size, shuffle=False)

        self.input_name = input_name
        self.datasize = len(self.torch_dl)

        self.enum_data = iter(self.torch_dl)

    def to_numpy(self, pt_tensor):
        return pt_tensor.detach().cpu().numpy() if pt_tensor.requires_grad else pt_tensor.cpu().numpy()

    def get_next(self):
        batch = next(self.enum_data, None)
        if batch is not None:
          return {self.input_name: self.to_numpy(batch[0])}
        else:
          return None

    def rewind(self):
        self.enum_data = iter(self.torch_dl)

qdr = QuntizationDataReader(calib_ds, batch_size=1, input_name='image')


quantized_model = quantize_static(model_fp32, model_quant,qdr, weight_type=QuantType.QInt8,)
