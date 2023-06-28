from edge_detection.geometric_edge_detection import geometric
from edge_detection.ip_edge_detection import *
from edge_detection.scan_edges import ScanEdges

edge_detection_dict = {'normals': normals, 'canny': canny, 'sobel': sobel, 'geometric': geometric}


class EdgeDetection:
    def __init__(self, method, ckpt, **kwargs):
        self.active = method != 'none'
        self.suffix = "_" + method if self.active else ""
        if method == 'scan':
            self.scan_edges = ScanEdges(**kwargs)
            self.scan_edges.load(ckpt)
            self.detect = self.scan
        elif self.active:
            self.detect = edge_detection_dict[method]

    def scan(self, *args, **kwargs):
        H, W = kwargs['hw']
        depth_edges_raw, depth_edges = self.scan_edges(kwargs['rays'], (W, H),
                                                       acc_threshold=args[0],
                                                       inclusion_threshold=args[1],
                                                       near_weight=args[2],
                                                       far_weight=args[3],
                                                       line_style=args[4] if len(args) > 4 else 'euclidean')
        depth_edges_raw = depth_edges_raw.reshape(H * W).cpu()
        depth_edges = depth_edges.reshape(H * W).cpu()
        normal_edges_raw = torch.zeros((H * W,))  # [H*W]
        normal_edges = torch.zeros((H * W,))  # [H*W]
        return depth_edges_raw, depth_edges, normal_edges_raw, normal_edges
