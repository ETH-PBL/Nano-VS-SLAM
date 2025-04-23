from kp2dtiny.models.kp2dtiny import KP2DTinyV2, KP2D_TINY_V2, KP2D_TINY_ATT_V2, KP2D_TINY_F_V2
from quantize import to_executorch, quantize_4_executorch

model = KP2DTinyV2(**KP2D_TINY_V2)
aten_dialect = to_executorch(model)
quantize_4_executorch(aten_dialect)

