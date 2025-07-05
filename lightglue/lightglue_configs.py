LG_KP2D_TINY_S = {
    "name": "lightglue",  # just for interfacing
    "input_dim": 32,  # input descriptor dimension (autoselected from weights)
    "descriptor_dim": 32,
    "n_layers": 4,
}

LG_KP2D_TINY_F = {
    "name": "lightglue",  # just for interfacing
    "input_dim": 64,  # input descriptor dimension (autoselected from weights)
    "descriptor_dim": 64,
    "n_layers": 4,
}

LG_KP2D_TINY_ATT = {
    "name": "lightglue",  # just for interfacing
    "input_dim": 32,  # input descriptor dimension (autoselected from weights)
    "descriptor_dim": 32,
    "n_layers": 4,
}


LIGHT_GLUE_CONFIGS = {"S": LG_KP2D_TINY_S,
                    "F": LG_KP2D_TINY_F,
                    "A": LG_KP2D_TINY_ATT}

def get_light_glue_config(config):
    if config not in LIGHT_GLUE_CONFIGS:
        raise ValueError("Config not supported")
    return LIGHT_GLUE_CONFIGS[config]