from edge_detection.ip_edge_detection import *
from edge_detection.scan_edges import ScanEdges

edge_detection_dict = {'normals': normals, 'canny': canny, 'sobel': sobel}


class EdgeDetection:
    def __init__(self, method, ckpt, **kwargs):
        self.active = method != 'none'
        if method == 'scan':
            self.scan_edges = ScanEdges(**kwargs)
            self.scan_edges.load(ckpt)
            self.detect = self.scan
        elif self.active:
            self.detect = edge_detection_dict[method]

    def scan(self, *args, **kwargs):
        H, W = kwargs['hw']
        depth_edge_mask = self.scan_edges(kwargs['rays'], (H, W)).reshape(H*W).cpu()  # [H*W]
        depth_edge_map = torch.where(depth_edge_mask, torch.zeros((H*W,)), torch.ones((H*W,)))  # [H*W]
        normal_edge_mask = torch.full((H*W,), False)  # [H*W]
        normal_edge_map = torch.zeros((H*W,))  # [H*W]
        return depth_edge_map, depth_edge_mask, normal_edge_map, normal_edge_mask
