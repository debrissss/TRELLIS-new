import importlib

__attributes = {
    'SparseStructure': 'sparse_structure',

    'SparseFeat2Render': 'sparse_feat2render',
    'SLat2Render':'structured_latent2render',
    'Slat2RenderGeo':'structured_latent2render',
    
    'SparseStructureLatent': 'sparse_structure_latent',
    'TextConditionedSparseStructureLatent': 'sparse_structure_latent',
    'ImageConditionedSparseStructureLatent': 'sparse_structure_latent',
    # ControlNet 改动：注册返回 raw 3D occupancy 的数据集副本。
    'SparseStructureLatent_ControlNet': 'sparse_structure_latent_ControlNet',
    'TextConditionedSparseStructureLatent_ControlNet': 'sparse_structure_latent_ControlNet',
    'ImageConditionedSparseStructureLatent_ControlNet': 'sparse_structure_latent_ControlNet',
    # ControlNet 改动：FaceScan 专用 paired 数据入口，分别读取 normal map、
    # control occupancy 和由完整 target mesh 预编码的 SS latent。
    'ImageConditionedFaceScanSparseStructureLatent_ControlNet': 'sparse_structure_latent_ControlNet',
    
    'SLat': 'structured_latent',
    'TextConditionedSLat': 'structured_latent',
    'ImageConditionedSLat': 'structured_latent',
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


# For Pylance
if __name__ == '__main__':
    from .sparse_structure import SparseStructure
    
    from .sparse_feat2render import SparseFeat2Render
    from .structured_latent2render import (
        SLat2Render,
        Slat2RenderGeo,
    )
    
    from .sparse_structure_latent import (
        SparseStructureLatent,
        TextConditionedSparseStructureLatent,
        ImageConditionedSparseStructureLatent,
    )
    from .sparse_structure_latent_ControlNet import (
        SparseStructureLatent_ControlNet,
        TextConditionedSparseStructureLatent_ControlNet,
        ImageConditionedSparseStructureLatent_ControlNet,
        ImageConditionedFaceScanSparseStructureLatent_ControlNet,
    )
    from .structured_latent import (
        SLat,
        TextConditionedSLat,
        ImageConditionedSLat,
    )
