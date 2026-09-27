"""ComfyUI_Swwan: independent image and workflow tools."""
from pathlib import Path
import folder_paths
from .registry import build_registry

__version__ = '1.0.0'
fonts_path = str(Path(__file__).with_name('fonts'))
existing, extensions = folder_paths.folder_names_and_paths.get('swwan_fonts', ([], set()))
folder_paths.folder_names_and_paths['swwan_fonts'] = (
    list(dict.fromkeys([fonts_path, *existing])), set(extensions) | {'.otf', '.ttf'}
)
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = build_registry(__name__)
WEB_DIRECTORY = './web/js'
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
