from gaussian_renderer.render import render
from gaussian_renderer.neilf import render_neilf
from gaussian_renderer.neilf_composite import render_neilf_composite
from gaussian_renderer.depth_render import render_with_depth


render_fn_dict = {
    "render": render,
    "neilf": render_neilf,
    "neilf_composite": render_neilf_composite,
    "depth": render_with_depth,
}