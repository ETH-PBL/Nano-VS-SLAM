import torch

new_class_mapping = {
    0: "None",
    1: "Architecture",  # wall
    2: "Architecture",  # building, edifice
    3: "Nature",  # sky
    4: "Architecture",  # floor, flooring
    5: "Nature",  # tree
    6: "Architecture",  # ceiling
    7: "Architecture",  # road, route
    8: "Furniture",  # bed
    9: "Architecture",  # windowpane, window
    10: "Nature",  # grass
    11: "Furniture",  # cabinet
    12: "Architecture",  # sidewalk, pavement
    13: "Person",  # person, individual, someone, somebody, mortal, soul
    14: "Nature",  # earth, ground
    15: "Architecture",  # door, double door
    16: "Furniture",  # table
    17: "Nature",  # mountain, mount
    18: "Nature",  # plant, flora, plant life
    19: "Furniture",  # curtain, drape, drapery, mantle, pall
    20: "Furniture",  # chair
    21: "Vehicles",  # car, auto, automobile, machine, motorcar
    22: "Nature",  # water
    23: "Static Stuff",  # painting, picture
    24: "Furniture",  # sofa, couch, lounge
    25: "Furniture",  # shelf
    26: "Architecture",  # house
    27: "Nature",  # sea
    28: "Furniture",  # mirror
    29: "Furniture",  # rug, carpet, carpeting
    30: "Nature",  # field
    31: "Furniture",  # armchair
    32: "Furniture",  # seat
    33: "Architecture",  # fence, fencing
    34: "Furniture",  # desk
    35: "Nature",  # rock, stone
    36: "Furniture",  # wardrobe, closet, press
    37: "Furniture",  # lamp
    38: "Furniture",  # bathtub, bathing tub, bath, tub
    39: "Architecture",  # railing, rail
    40: "Furniture",  # cushion
    41: "Static Stuff",  # base, pedestal, stand
    42: "Static Stuff",  # box
    43: "Architecture",  # column, pillar
    44: "Static Stuff",  # signboard, sign
    45: "Furniture",  # chest of drawers, chest, bureau, dresser
    46: "Furniture",  # counter
    47: "Nature",  # sand
    48: "Furniture",  # sink
    49: "Architecture",  # skyscraper
    50: "Furniture",  # fireplace, hearth, open fireplace
    51: "Furniture",  # refrigerator, icebox
    52: "Static Stuff",  # grandstand, covered stand
    53: "Architecture",  # path
    54: "Architecture",  # stairs, steps
    55: "Architecture",  # runway
    56: "Static Stuff",  # case, display case, showcase, vitrine
    57: "Furniture",  # pool table, billiard table, snooker table
    58: "Furniture",  # pillow
    59: "Architecture",  # screen door, screen
    60: "Architecture",  # stairway, staircase
    61: "Nature",  # river
    62: "Architecture",  # bridge, span
    63: "Furniture",  # bookcase
    64: "Furniture",  # blind, screen
    65: "Furniture",  # coffee table, cocktail table
    66: "Furniture",  # toilet, can, commode, crapper, pot, potty, stool, throne
    67: "Nature",  # flower
    68: "Static Stuff",  # book
    69: "Nature",  # hill
    70: "Furniture",  # bench
    71: "Furniture",  # countertop
    72: "Furniture",  # stove, kitchen stove, range, kitchen range, cooking stove
    73: "Nature",  # palm, palm tree
    74: "Furniture",  # kitchen island
    75: "Dynamic Stuff",  # computer, computing machine, computing device, data processor, electronic computer, information processing system
    76: "Furniture",  # swivel chair
    77: "Vehicles",  # boat
    78: "Static Stuff",  # bar
    79: "Dynamic Stuff",  # arcade machine
    80: "Architecture",  # hovel, hut, hutch, shack, shanty
    81: "Vehicles",  # bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle
    82: "Dynamic Stuff",  # towel
    83: "Static Stuff",  # light, light source
    84: "Vehicles",  # truck, motortruck
    85: "Static Stuff",  # tower
    86: "Furniture",  # chandelier, pendant, pendent
    87: "Static Stuff",  # awning, sunshade, sunblind
    88: "Static Stuff",  # streetlight, street lamp
    89: "Static Stuff",  # booth, cubicle, stall, kiosk
    90: "Dynamic Stuff",  # television, television receiver, television set, tv, tv set, idiot box, boob tube, telly, goggle box
    91: "Vehicles",  # airplane, aeroplane, plane
    92: "Architecture",  # dirt track
    93: "Dynamic Stuff",  # apparel, wearing apparel, dress, clothes
    94: "Static Stuff",  # pole
    95: "Nature",  # land, ground, soil
    96: "Architecture",  # bannister, banister, balustrade, balusters, handrail
    97: "Dynamic Stuff",  # escalator, moving staircase, moving stairway
    98: "Furniture",  # ottoman, pouf, pouffe, puff, hassock
    99: "Dynamic Stuff",  # bottle
    100: "Static Stuff",  # buffet, counter, sideboard
    101: "Static Stuff",  # poster, posting, placard, notice, bill, card
    102: "Static Stuff",  # stage
    103: "Vehicles",  # van
    104: "Vehicles",  # ship
    105: "Static Stuff",  # fountain
    106: "Dynamic Stuff",  # conveyer belt, conveyor belt, conveyer, conveyor, transporter
    107: "Static Stuff",  # canopy
    108: "Dynamic Stuff",  # washer, automatic washer, washing machine
    109: "Dynamic Stuff",  # plaything, toy
    110: "Dynamic Stuff",  # swimming pool, swimming bath, natatorium
    111: "Furniture",  # stool
    112: "Dynamic Stuff",  # barrel, cask
    113: "Dynamic Stuff",  # basket, handbasket
    114: "Nature",  # waterfall, falls
    115: "Dynamic Stuff",  # tent, collapsible shelter
    116: "Dynamic Stuff",  # bag
    117: "Vehicles",  # minibike, motorbike
    118: "Furniture",  # cradle
    119: "Furniture",  # oven
    120: "Dynamic Stuff",  # ball
    121: "Dynamic Stuff",  # food, solid food
    122: "Architecture",  # step, stair
    123: "Static Stuff",  # tank, storage tank
    124: "Static Stuff",  # trade name, brand name, brand, marque
    125: "Dynamic Stuff",  # microwave, microwave oven
    126: "Nature",  # pot, flowerpot
    127: "Dynamic Stuff",  # animal, animate being, beast, brute, creature, fauna
    128: "Vehicles",  # bicycle, bike, wheel, cycle
    129: "Nature",  # lake
    130: "Dynamic Stuff",  # dishwasher, dish washer, dishwashing machine
    131: "Static Stuff",  # screen, silver screen, projection screen
    132: "Dynamic Stuff",  # blanket, cover
    133: "Static Stuff",  # sculpture
    134: "Furniture",  # Hood, exhaust hood
    135: "Furniture",  # Sconce
    136: "Furniture",  # Vase
    137: "Static Stuff",  # Traffic light, traffic signal, stoplight
    138: "Static Stuff",  # Tray
    139: "Static Stuff",  # Ashcan, trash can, garbage can, wastebin
    140: "Furniture",  # Fan
    141: "Architecture",  # Pier, wharf, wharfage, dock
    142: "Static Stuff",  # CRT screen
    143: "Static Stuff",  # Plate
    144: "Static Stuff",  # Monitor, monitoring device
    145: "Static Stuff",  # Bulletin board, notice board
    146: "Furniture",  # Shower
    147: "Furniture",  # Radiator
    148: "Furniture",  # Glass, drinking glass
    149: "Static Stuff",  # Clock
    150: "Static Stuff",  # Flag
}

# Convert this mapping to indices
class_to_index = {
    "None": 0,
    "Person": 1,
    "Vehicles": 2,
    "Architecture": 3,
    "Furniture": 4,
    "Nature": 5,
    "Dynamic Stuff": 6,
    "Static Stuff": 7,
}


def get_mapping():
    num_original_classes = 151
    # Create the final mapping dictionary
    final_mapping = {
        original: class_to_index[new_class]
        for original, new_class in new_class_mapping.items()
    }
    new_class_indices = torch.full((num_original_classes,), 0, dtype=torch.uint8)

    for original_class, new_class_index in final_mapping.items():
        # Subtract 1 from the original class because class indices are 1-based in your dataset
        new_class_indices[original_class] = new_class_index
    return new_class_indices
