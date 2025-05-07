# models/graph.py

import torch
import torch.nn.functional as F


class GraphBuilder:
    """构建图像块图结构用于GCN处理"""

    def __init__(self, patch_size=16, neighbor=4):
        """
        Args:
            patch_size: 图像块大小（正方形）
            neighbor: 邻接节点数量控制（未使用，保留配置）
        """
        self.patch_size = patch_size
        self.neighbor = neighbor

    def build(self, x):
        """从输入图像构建图结构（节点特征 + 边索引）"""
        B, C, H, W = x.shape
        p = self.patch_size
        N = (H // p) * (W // p)  # Number of patches per image

        # Extract image patches using unfold
        patches = F.unfold(x, p, stride=p)  # [B, C*p*p, N]
        patches = patches.view(B, C, p, p, N).permute(0, 4, 1, 2, 3)  # [B, N, C, p, p]
        patches = patches.mean(dim=(3, 4))  # [B, N, C]

        # Generate coordinate grid for original image
        coord = torch.stack(torch.meshgrid(
            torch.arange(H, device=x.device, dtype=torch.float32),
            torch.arange(W, device=x.device, dtype=torch.float32)
        ), dim=-1)  # [H, W, 2]

        # Compute center coordinates for each patch
        y_centers = torch.linspace(p // 2, H - p // 2, H // p, device=x.device)
        x_centers = torch.linspace(p // 2, W - p // 2, W // p, device=x.device)
        grid_y, grid_x = torch.meshgrid(y_centers, x_centers, indexing='ij')
        coord_centers = torch.stack([grid_x, grid_y], dim=-1)  # [H/p, W/p, 2]
        coord_centers = coord_centers.view(1, N, 2).repeat(B, 1, 1)  # [B, N, 2]

        # Combine patch features and coordinates
        nodes = torch.cat([patches, coord_centers], dim=-1)  # [B, N, C+2]
        nodes = nodes.view(B * N, -1)  # [B*N, C+2]

        # Generate edge index based on grid neighborhood
        edge_index = self._grid_edges(B, H // p, W // p)

        return nodes, edge_index.to(x.device)

    def _grid_edges(self, batch, h, w):
        """为网格结构生成邻接边索引"""
        nodes_per_sample = h * w
        edges = []
        for b in range(batch):
            offset = b * nodes_per_sample
            for i in range(h):
                for j in range(w):
                    current = i * w + j + offset
                    # Top neighbor
                    if i > 0:
                        neighbor = (i - 1) * w + j + offset
                        edges.append([current, neighbor])
                    # Bottom neighbor
                    if i < h - 1:
                        neighbor = (i + 1) * w + j + offset
                        edges.append([current, neighbor])
                    # Left neighbor
                    if j > 0:
                        neighbor = i * w + (j - 1) + offset
                        edges.append([current, neighbor])
                    # Right neighbor
                    if j < w - 1:
                        neighbor = i * w + (j + 1) + offset
                        edges.append([current, neighbor])

        # Fallback: self-loops if no edges found
        if not edges:
            for b in range(batch):
                offset = b * (h * w)
                for i in range(h * w):
                    edges.append([offset + i, offset + i])

        return torch.tensor(edges, dtype=torch.long).t().contiguous()