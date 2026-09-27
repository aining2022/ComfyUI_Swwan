# SPDX-License-Identifier: MIT
# Helpers adapted from ComfyUI-Apt_Preset; original notice in licenses/MIT-Apt.txt.

class AnyType(str):
    def __eq__(self, _) -> bool:
        return True
    def __ne__(self, __value: object) -> bool:
        return False

ANY_TYPE = AnyType("*")

any_type = AnyType("*")
