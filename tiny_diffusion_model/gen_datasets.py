import numpy as np
import torch
from torch.utils.data import Dataset


class BarnsleyFern(Dataset):
    def __init__(self, n_points: int):
        X, Y = self._generate_fern(n_points)
        data_np = np.stack((X, Y), axis=1)
        self.data = torch.from_numpy(data_np).float()

    def _generate_fern(self, n):
        x, y = 0, 0
        X, Y = [], []
        for _ in range(n):
            r = np.random.rand()
            if r < 0.01:
                x, y = 0, 0.16 * y
            elif r < 0.86:
                x_new = 0.85 * x + 0.04 * y
                y = -0.04 * x + 0.85 * y + 1.6
                x = x_new
            elif r < 0.93:
                x_new = 0.2 * x - 0.26 * y
                y = 0.23 * x + 0.22 * y + 1.6
                x = x_new
            else:
                x_new = -0.15 * x + 0.28 * y
                y = 0.26 * x + 0.24 * y + 0.44
                x = x_new
            X.append(x)
            Y.append(y)
        return np.array(X), np.array(Y)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SierpinskiTriangle(Dataset):
    def __init__(self, n_points: int):
        self.vertices = np.array([[0, 0], [0.5, np.sqrt(3)/2], [1, 0]])
        self.data = self._generate_triangle(n_points)

    def _generate_triangle(self, n):
        point = np.array([0.0, 0.0])
        points = []
        for _ in range(n):
            vertex = self.vertices[np.random.randint(0, 3)]
            point = (point + vertex) / 2
            points.append(point)
        return torch.tensor(points).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class KochSnowflake(Dataset):
    def __init__(self, n_points: int = 10000, levels: int = 5):
        self.data = self._generate_snowflake(levels, n_points)

    def _koch_curve(self, p1, p2, level):
        if level == 0:
            return [p1, p2]
        else:
            p1 = np.array(p1)
            p2 = np.array(p2)
            delta = (p2 - p1) / 3.0
            p3 = p1 + delta
            p5 = p2 - delta

            angle = np.pi / 3
            rotation = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle),  np.cos(angle)]
            ])
            p4 = p3 + rotation @ delta

            return (
                self._koch_curve(p1, p3, level - 1)[:-1] +
                self._koch_curve(p3, p4, level - 1)[:-1] +
                self._koch_curve(p4, p5, level - 1)[:-1] +
                self._koch_curve(p5, p2, level - 1)
            )

    def _generate_snowflake(self, level, n_points):
        # Initial equilateral triangle
        height = np.sqrt(3) / 2
        A = (0.0, 0.0)
        B = (1.0, 0.0)
        C = (0.5, height)
        # Generate all three sides
        points = (
            self._koch_curve(A, B, level)[:-1] +
            self._koch_curve(B, C, level)[:-1] +
            self._koch_curve(C, A, level)
        )
        points = np.array(points)
        # Uniformly sample if needed
        if len(points) > n_points:
            idx = np.random.choice(len(points), size=n_points, replace=False)
            points = points[idx]
        return torch.tensor(points).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



class SwissRoll(Dataset):
    def __init__(self, n_points: int, noise=0.05):
        self.data = self._generate_roll(n_points, noise)

    def _generate_roll(self, n, noise):
        t = 1.5 * np.pi * (1 + 2 * np.random.rand(n))
        x = t * np.cos(t)
        y = 21 * np.random.rand(n)
        z = t * np.sin(t)
        x += np.random.normal(0, noise, n)
        z += np.random.normal(0, noise, n)
        # Just keep x and z for 2D
        return torch.tensor(np.stack([x, z], axis=1)).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
