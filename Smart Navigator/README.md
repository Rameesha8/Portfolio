# Smart Navigator – AI Pathfinding Visualizer

## 1. Introduction
The **Smart Navigator** project is an interactive pathfinding visualization tool developed using Python and the Tkinter library. It enables users to visually understand how different pathfinding algorithms—**Breadth-First Search (BFS)**, **Depth-First Search (DFS)**, and **A***—work to find the shortest path in a 2D grid.  

Users can define obstacles (walls), a start point, and an end point on the grid. Once set, users can run any of the available algorithms to see the **step-by-step pathfinding process** in action.

---

## 2. Objectives
- Provide a **visual and educational tool** for understanding pathfinding algorithms.  
- Simulate and **compare the performance** of BFS, DFS, and A* Search.  
- Give users a **hands-on experience** with how heuristics influence intelligent search.  

---

## 3. Tools and Technologies Used

| Component           | Details                      |
|-------------------|-------------------------------|
| Programming Language | Python 3.x                  |
| GUI Framework        | Tkinter (Python standard GUI) |
| Algorithms Implemented | BFS, DFS, A* Search         |

---

## 4. System Design

### Grid Layout
- The grid is a **20x20 matrix** (400 cells).  
- Each cell can be a **free path**, **wall**, **start**, or **end**.  

### User Interactions
- **Left Click:** Place wall (obstacle).  
- **Right Click:**  
  - First click sets the **start point** (green).  
  - Second click sets the **end point** (red).  

### Buttons
- **Run BFS:** Executes Breadth-First Search.  
- **Run DFS:** Executes Depth-First Search.  
- **Run A\*:** Executes A* Search algorithm using Manhattan distance as heuristic.  

---

## 5. Algorithms Explained

### Breadth-First Search (BFS)
- Explores equally in all directions from the start.  
- Guarantees **shortest path** in unweighted graphs.  
- Uses a **queue (FIFO)**.

### Depth-First Search (DFS)
- Explores as deep as possible before backtracking.  
- **Does not guarantee shortest path**.  
- Uses a **stack (LIFO)**.

### A* Search
- Uses both **cost so far (g(n))** and **estimated cost to goal (h(n))**.  
- **Heuristic Used:** Manhattan distance: `|x1 − x2| + |y1 − y2|`.  
- Guarantees the **shortest path** if heuristic is admissible.  
- Combines **priority queue** and **heuristic function**.

---

## 6. Features
- **Interactive and user-friendly interface**.  
- **Visual differences** in algorithm behaviors.  
- Uses **colors** to indicate:  
  - Walls: **black**  
  - Start: **green**  
  - End: **red**  
  - Visited nodes: **yellow/orange/purple** depending on algorithm  
  - Final path: **blue**  

---

## 7. Conclusion
The **Smart Navigator** provides an effective way to **visualize and understand** the working of fundamental pathfinding algorithms. It successfully demonstrates how **BFS, DFS, and A*** differ in approach and efficiency.  

With simple improvements and additions, this tool can be extended into a more **robust pathfinding simulation and educational utility**.