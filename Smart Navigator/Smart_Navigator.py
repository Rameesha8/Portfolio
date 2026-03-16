import tkinter as tk
from collections import deque
import heapq

ROWS = 20
COLS = 20
CELL_SIZE = 25

class SmartNavigator:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Navigator - Multi-Algorithm")
        self.canvas = tk.Canvas(root, width=COLS * CELL_SIZE, height=ROWS * CELL_SIZE, bg="lightblue")
        self.canvas.pack()

        self.rects = [[None for _ in range(COLS)] for _ in range(ROWS)]
        self.grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]
        self.start = None
        self.end = None

        for r in range(ROWS):
            for c in range(COLS):
                x1, y1 = c * CELL_SIZE, r * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
                rect = self.canvas.create_rectangle(x1, y1, x2, y2, fill="lightblue", outline="gray")
                self.rects[r][c] = rect

        self.canvas.bind("<Button-1>", self.add_wall)
        self.canvas.bind("<Button-3>", self.set_start_or_end)

        self.bfs_button = tk.Button(root, text="Run BFS", command=self.run_bfs)
        self.bfs_button.pack(side=tk.LEFT)

        self.dfs_button = tk.Button(root, text="Run DFS", command=self.run_dfs)
        self.dfs_button.pack(side=tk.LEFT)

        self.astar_button = tk.Button(root, text="Run A*", command=self.run_astar)
        self.astar_button.pack(side=tk.LEFT)

    def add_wall(self, event):
        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE
        if self.grid[row][col] == 0:
            self.grid[row][col] = 1
            self.canvas.itemconfig(self.rects[row][col], fill="black")

    def set_start_or_end(self, event):
        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE
        if self.start is None:
            self.start = (row, col)
            self.canvas.itemconfig(self.rects[row][col], fill="green")
            print("Start set to:", self.start)
        elif self.end is None:
            self.end = (row, col)
            self.canvas.itemconfig(self.rects[row][col], fill="red")
            print("End set to:", self.end)

    def run_bfs(self):
        self.reset_path()
        self.search_path(self.bfs)

    def run_dfs(self):
        self.reset_path()
        self.search_path(self.dfs)

    def run_astar(self):
        self.reset_path()
        self.search_path(self.astar)

    def reset_path(self):
        for r in range(ROWS):
            for c in range(COLS):
                if self.grid[r][c] == 0 and (r, c) != self.start and (r, c) != self.end:
                    self.canvas.itemconfig(self.rects[r][c], fill="lightblue")

    def search_path(self, algorithm):
        if self.start is None or self.end is None:
            print("Start and End points must be set!")
            return
        prev = algorithm()
        if prev:
            print("Path found!")
            self.trace_path(prev)
        else:
            print("No path found!")

    def bfs(self):
        queue = deque()
        queue.append(self.start)
        visited = [[False for _ in range(COLS)] for _ in range(ROWS)]
        prev = [[None for _ in range(COLS)] for _ in range(ROWS)]
        visited[self.start[0]][self.start[1]] = True

        while queue:
            row, col = queue.popleft()
            if (row, col) == self.end:
                return prev
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                r, c = row + dr, col + dc
                if 0 <= r < ROWS and 0 <= c < COLS and not visited[r][c] and self.grid[r][c] == 0:
                    queue.append((r, c))
                    visited[r][c] = True
                    prev[r][c] = (row, col)
                    self.canvas.itemconfig(self.rects[r][c], fill="yellow")
                    self.canvas.update()
        return None

    def dfs(self):
        stack = [self.start]
        visited = [[False for _ in range(COLS)] for _ in range(ROWS)]
        prev = [[None for _ in range(COLS)] for _ in range(ROWS)]
        visited[self.start[0]][self.start[1]] = True

        while stack:
            row, col = stack.pop()
            if (row, col) == self.end:
                return prev
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                r, c = row + dr, col + dc
                if 0 <= r < ROWS and 0 <= c < COLS and not visited[r][c] and self.grid[r][c] == 0:
                    stack.append((r, c))
                    visited[r][c] = True
                    prev[r][c] = (row, col)
                    self.canvas.itemconfig(self.rects[r][c], fill="orange")
                    self.canvas.update()
        return None

    def astar(self):
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        heap = [(0, self.start)]
        came_from = [[None for _ in range(COLS)] for _ in range(ROWS)]
        g_score = [[float('inf') for _ in range(COLS)] for _ in range(ROWS)]
        g_score[self.start[0]][self.start[1]] = 0

        while heap:
            _, current = heapq.heappop(heap)
            if current == self.end:
                return came_from
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                r, c = current[0] + dr, current[1] + dc
                if 0 <= r < ROWS and 0 <= c < COLS and self.grid[r][c] == 0:
                    tentative_g = g_score[current[0]][current[1]] + 1
                    if tentative_g < g_score[r][c]:
                        g_score[r][c] = tentative_g
                        f_score = tentative_g + heuristic((r, c), self.end)
                        heapq.heappush(heap, (f_score, (r, c)))
                        came_from[r][c] = current
                        self.canvas.itemconfig(self.rects[r][c], fill="purple")
                        self.canvas.update()
        return None

    def trace_path(self, prev):
        row, col = self.end
        while (row, col) != self.start:
            row, col = prev[row][col]
            if (row, col) != self.start:
                self.canvas.itemconfig(self.rects[row][col], fill="blue")
                self.canvas.update()

# Run the app
root = tk.Tk()
app = SmartNavigator(root)
root.mainloop()