# SPDX-License-Identifier: GPL-3.0-only
# Immutable upstream test reference; see THIRD_PARTY_NOTICES.md for source revisions.
class ColorConverterGuhai:
    """颜色转换节点 - 孤海"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "色值": ("STRING", {
                    "default": "#FFFFFF",
                    "multiline": False
                }),
                "转换后": (["#HEX", "HEX", "RGB", "HSL"], {
                    "default": "#HEX"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "COLORCODE")
    RETURN_NAMES = ("字符串", "颜色控件")
    FUNCTION = "convert_color"
    CATEGORY = "孤海工具箱"
    OUTPUT_NODE = False

    def normalize_symbols(self, text):
        """将全角符号转换为半角符号"""
        text = text.replace('，', ',')
        text = text.replace('（', '(')
        text = text.replace('）', ')')
        text = text.replace('　', ' ')
        return text

    def parse_color(self, color_str):
        """解析多种格式的颜色值"""
        color_str = str(color_str).strip().lower()
        color_str = self.normalize_symbols(color_str)
        color_str = re.sub(r'\s+', '', color_str)
        
        hex_match = re.match(r'^#?([0-9a-f]{3}|[0-9a-f]{6})$', color_str)
        if hex_match:
            hex_code = hex_match.group(1)
            if len(hex_code) == 3:
                hex_code = ''.join([c*2 for c in hex_code])
            return self.hex_to_rgb(hex_code)
        
        rgb_pattern = r'^[\(（]?\s*(\d{1,3})\s*[，,]\s*(\d{1,3})\s*[，,]\s*(\d{1,3})\s*[\)）]?$'
        rgb_match = re.match(rgb_pattern, color_str)
        if rgb_match:
            r, g, b = map(int, rgb_match.groups())
            if 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255:
                return (r/255, g/255, b/255)
        
        hsl_pattern = r'^[\(（]?\s*(\d{1,3})\s*[，,]\s*(\d{1,3})%\s*[，,]\s*(\d{1,3})%\s*[\)）]?$'
        hsl_match = re.match(hsl_pattern, color_str)
        if hsl_match:
            h, s, l_val = map(float, hsl_match.groups())
            h = h / 360.0
            s = s / 100.0
            l_val = l_val / 100.0
            return self.hsl_to_rgb_normalized(h, s, l_val)
        
        try:
            if re.match(r'^[01]\.\d+\s*[，,]\s*[01]\.\d+\s*[，,]\s*[01]\.\d+$', color_str):
                parts = re.split(r'[，,]\s*', color_str)
                if len(parts) == 3:
                    r, g, b = map(float, parts)
                    if 0 <= r <= 1 and 0 <= g <= 1 and 0 <= b <= 1:
                        return (r, g, b)
        except:
            pass
        
        return (1.0, 1.0, 1.0)
    
    def hex_to_rgb(self, hex_code):
        """十六进制转RGB(0-1范围)"""
        hex_code = hex_code.lstrip('#')
        if len(hex_code) == 3:
            hex_code = ''.join([c*2 for c in hex_code])
        r = int(hex_code[0:2], 16) / 255.0
        g = int(hex_code[2:4], 16) / 255.0
        b = int(hex_code[4:6], 16) / 255.0
        return (r, g, b)
    
    def rgb_to_hex(self, r, g, b, with_hash=True):
        """RGB(0-1范围)转十六进制"""
        r_int = int(min(max(r * 255, 0), 255))
        g_int = int(min(max(g * 255, 0), 255))
        b_int = int(min(max(b * 255, 0), 255))
        hex_code = f"{r_int:02x}{g_int:02x}{b_int:02x}"
        return f"#{hex_code}" if with_hash else hex_code
    
    def rgb_to_hsl_normalized(self, r, g, b):
        """RGB(0-1范围)转HSL(0-360, 0-100%, 0-100%)"""
        h, l_val, s = colorsys.rgb_to_hls(r, g, b)
        h = (h * 360) % 360
        s = s * 100
        l_val = l_val * 100
        return h, s, l_val
    
    def hsl_to_rgb_normalized(self, h, s, l_val):
        """HSL(0-1范围)转RGB(0-1范围)"""
        r, g, b = colorsys.hls_to_rgb(h, l_val, s)
        return (r, g, b)
    
    def convert_color(self, 色值, 转换后):
        """转换颜色格式"""
        rgb_normalized = self.parse_color(色值)
        
        if 转换后 == "#HEX":
            result_str = self.rgb_to_hex(*rgb_normalized, with_hash=True)
        elif 转换后 == "HEX":
            result_str = self.rgb_to_hex(*rgb_normalized, with_hash=False)
        elif 转换后 == "RGB":
            r = int(rgb_normalized[0] * 255)
            g = int(rgb_normalized[1] * 255)
            b = int(rgb_normalized[2] * 255)
            result_str = f"{r},{g},{b}"
        elif 转换后 == "HSL":
            h, s, l_val = self.rgb_to_hsl_normalized(*rgb_normalized)
            result_str = f"{int(round(h))},{int(round(s))}%,{int(round(l_val))}%"
        else:
            result_str = "#ffffff"
        
        color_control = self.rgb_to_hex(*rgb_normalized, with_hash=True)
        
        return (result_str, color_control)
