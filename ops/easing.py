# SPDX-License-Identifier: MIT
# Helpers adapted from ComfyUI-Apt_Preset; original notice in licenses/MIT-Apt.txt.
import math

def easeInBack(t):
    s = 1.70158
    return t * t * ((s + 1) * t - s)

def easeInOutSinSquared(t):
    if t < 0.5:
        return 0.5 * (1 - math.cos(t * 2 * math.pi))
    else:
        return 0.5 * (1 + math.cos((t - 0.5) * 2 * math.pi))

def easeOutBack(t):
    s = 1.70158
    return ((t - 1) * t * ((s + 1) * t + s)) + 1

def easeInOutBack(t):
    s = 1.70158 * 1.525
    if t < 0.5:
        return (t * t * (t * (s + 1) - s)) * 2
    return ((t - 2) * t * ((s + 1) * t + s) + 2) * 2

def easeInElastic(t):
    if t == 0:
        return 0
    if t == 1:
        return 1
    p = 0.3
    s = p / 4
    return -(math.pow(2, 10 * (t - 1)) * math.sin((t - 1 - s) * (2 * math.pi) / p))

def easeOutElastic(t):
    if t == 0:
        return 0
    if t == 1:
        return 1
    p = 0.3
    s = p / 4
    return math.pow(2, -10 * t) * math.sin((t - s) * (2 * math.pi) / p) + 1

def easeInOutElastic(t):
    if t == 0:
        return 0
    if t == 1:
        return 1
    p = 0.3 * 1.5
    s = p / 4
    t = t * 2
    if t < 1:
        return -0.5 * (
            math.pow(2, 10 * (t - 1)) * math.sin((t - 1 - s) * (2 * math.pi) / p)
        )
    return (
        0.5 * math.pow(2, -10 * (t - 1)) * math.sin((t - 1 - s) * (2 * math.pi) / p)
        + 1
    )

def easeInBounce(t):
    return 1 - easeOutBounce(1 - t)

def easeOutBounce(t):
    if t < (1 / 2.75):
        return 7.5625 * t * t
    elif t < (2 / 2.75):
        t -= 1.5 / 2.75
        return 7.5625 * t * t + 0.75
    elif t < (2.5 / 2.75):
        t -= 2.25 / 2.75
        return 7.5625 * t * t + 0.9375
    else:
        t -= 2.625 / 2.75
        return 7.5625 * t * t + 0.984375

def easeInOutBounce(t):
    if t < 0.5:
        return easeInBounce(t * 2) * 0.5
    return easeOutBounce(t * 2 - 1) * 0.5 + 0.5

def easeInQuart(t):
    return t * t * t * t

def easeOutQuart(t):
    t -= 1
    return -(t**2 * t * t - 1)

def easeInOutQuart(t):
    t *= 2
    if t < 1:
        return 0.5 * t * t * t * t
    t -= 2
    return -0.5 * (t**2 * t * t - 2)

def easeInCubic(t):
    return t * t * t

def easeOutCubic(t):
    t -= 1
    return t**2 * t + 1

def easeInOutCubic(t):
    t *= 2
    if t < 1:
        return 0.5 * t * t * t
    t -= 2
    return 0.5 * (t**2 * t + 2)

def easeInCirc(t):
    return -(math.sqrt(1 - t * t) - 1)

def easeOutCirc(t):
    t -= 1
    return math.sqrt(1 - t**2)

def easeInOutCirc(t):
    t *= 2
    if t < 1:
        return -0.5 * (math.sqrt(1 - t**2) - 1)
    t -= 2
    return 0.5 * (math.sqrt(1 - t**2) + 1)

def easeInSine(t):
    return -math.cos(t * (math.pi / 2)) + 1

def easeOutSine(t):
    return math.sin(t * (math.pi / 2))

def easeInOutSine(t):
    return -0.5 * (math.cos(math.pi * t) - 1)

def easeLinear(t):
    return t

easing_functions = {
    "Linear": easeLinear,
    "Sine_In": easeInSine,
    "Sine_Out": easeOutSine,
    "Sine_InOut": easeInOutSine,
    "Sin_Squared": easeInOutSinSquared,
    "Quart_In": easeInQuart,
    "Quart_Out": easeOutQuart,
    "Quart_InOut": easeInOutQuart,
    "Cubic_In": easeInCubic,
    "Cubic_Out": easeOutCubic,
    "Cubic_InOut": easeInOutCubic,
    "Circ_In": easeInCirc,
    "Circ_Out": easeOutCirc,
    "Circ_InOut": easeInOutCirc,
    "Back_In": easeInBack,
    "Back_Out": easeOutBack,
    "Back_InOut": easeInOutBack,
    "Elastic_In": easeInElastic,
    "Elastic_Out": easeOutElastic,
    "Elastic_InOut": easeInOutElastic,
    "Bounce_In": easeInBounce,
    "Bounce_Out": easeOutBounce,
    "Bounce_InOut": easeInOutBounce,
}

EASING_TYPES = list(easing_functions.keys())

def apply_easing(value, easing_type):
    function_ease = easing_functions.get(easing_type)
    if function_ease:
        return function_ease(value)

    raise ValueError(f"Unknown easing type: {easing_type}")
