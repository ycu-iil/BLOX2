# lazy import
def __getattr__(name):
    if name == "LassoPredictor":
        from .lasso import LassoPredictor
        return LassoPredictor
    if name == "RidgePredictor":
        from .ridge import RidgePredictor
        return RidgePredictor
    if name == "RandomForestPredictor":
        from .random_forest import RandomForestPredictor
        return RandomForestPredictor
    if name == "SVRPredictor":
        from .svr import SVRPredictor
        return SVRPredictor
    if name == "TabPFNPredictor":
        from .tabpfn import TabPFNPredictor
        return TabPFNPredictor
    if name == "LightGBMPredictor":
        from .lightgbm import LightGBMPredictor
        return LightGBMPredictor
    raise AttributeError(f"module {__name__} has no attribute {name}")