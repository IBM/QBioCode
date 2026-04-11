from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import types


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name: str, relative_path: str):
    spec = spec_from_file_location(module_name, REPO_ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {relative_path}")

    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def ensure_package(package_name: str, relative_path: str):
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(REPO_ROOT / relative_path)]
        sys.modules[package_name] = package
    return package
