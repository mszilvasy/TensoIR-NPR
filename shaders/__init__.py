from .specular_only import SpecularOnly
from .blinn_phong import BlinnPhong
from .gooch import Gooch

shader_dict = {
    'specular_only': SpecularOnly,
    'blinn_phong': BlinnPhong,
    'gooch': Gooch
}
