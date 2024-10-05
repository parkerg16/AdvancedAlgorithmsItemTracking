import sys
import heapq
import pandas as pd
import random
from math import sqrt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsRectItem,
    QGraphicsTextItem, QVBoxLayout, QWidget, QPushButton, QLabel, QComboBox, QSpinBox,
    QFileDialog, QInputDialog, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QFont
import os
import json


def generate_warehouse_data(num_aisles=5, max_shelves_per_aisle=10, save_to_csv=True, filename="warehouse_layout.csv"):
    """Generate warehouse data with aisles and shelves and save it as a CSV file."""
    shelf_locations = ['A', 'B', 'C', 'D']

    inventory_data = []

    for aisle_num in range(1, num_aisles + 1):
        num_shelves = random.randint(2, max_shelves_per_aisle)

        for shelf_num in range(1, num_shelves + 1):
            is_shelf_empty = random.choice([True, False])

            for shelf_location in shelf_locations:
                if is_shelf_empty:
                    inventory_data.append({
                        'Aisle_Number': f"Aisle_{aisle_num}",
                        'Shelf_Number': f"Shelf_{shelf_num}",
                        'Shelf_Location': shelf_location,
                        'Item': None,
                        'Quantity': 0,
                        'Is_Shelf_Empty': True
                    })
                else:
                    inventory_data.append({
                        'Aisle_Number': f"Aisle_{aisle_num}",
                        'Shelf_Number': f"Shelf_{shelf_num}",
                        'Shelf_Location': shelf_location,
                        'Item': f"Item_{random.randint(1000, 9999)}",
                        'Quantity': random.randint(1, 100),
                        'Is_Shelf_Empty': False
                    })

    # Convert the inventory data into a Pandas DataFrame
    df_inventory = pd.DataFrame(inventory_data)

    # Optionally save the DataFrame to a CSV file for viewing
    if save_to_csv:
        df_inventory.to_csv(filename, index=False)
        print(f"Warehouse layout saved to {filename}")

    return df_inventory


class Node(QGraphicsRectItem):
    def __init__(self, x, y, size, name, parent_window):
        super().__init__(0, 0, size, size)
        self.setPos(x * size, y * size)
        self.name = name
        self.setBrush(QBrush(QColor(255, 255, 255)))  # White by default
        self.parent_window = parent_window
        self.is_obstacle = False
        self.is_start = False
        self.is_end = False
        self.is_aisle = False  # New flag for aisles

    def mousePressEvent(self, event):
        """Handle node click based on the current mode (start, end, barrier)."""
        mode = self.parent_window.current_mode
        if mode == 'start':
            self.parent_window.set_start_node(self)
        elif mode == 'end':
            self.parent_window.set_end_node(self)
        elif mode == 'barrier':
            self.set_as_barrier()

    def mouseMoveEvent(self, event):
        """Allow for dragging to set barriers in barrier mode."""
        mode = self.parent_window.current_mode
        if mode == 'barrier':
            self.set_as_barrier()

    def set_as_barrier(self):
        """Mark the node as a barrier (yellow), unless it is an aisle or start/end node."""
        if not self.is_aisle and not self.is_start and not self.is_end:
            self.setBrush(QBrush(QColor(255, 255, 0)))  # Yellow for barriers
            self.is_obstacle = True
    def set_visited(self):
        self.setBrush(QBrush(QColor(0, 255, 0)))  # Green when visited

    def set_path(self):
        self.setBrush(QBrush(QColor(0, 0, 255)))  # Blue when part of the path

    def reset(self):
        """Reset the node to its default state (white or aisle color)."""
        if not self.is_obstacle and not self.is_start and not self.is_end:
            if not self.is_aisle:
                self.setBrush(QBrush(QColor(255, 255, 255)))  # White for reset
            else:
                # Reset to the original color based on aisle orientation
                if self.parent_window.orientation_type == 'vertical':
                    color = QColor(150, 150, 250)  # Light blue
                else:
                    color = QColor(150, 0, 250)  # Purple
                self.setBrush(QBrush(color))

    def set_as_start(self):
        """Mark the node as the start (green)."""
        self.is_start = True
        self.setBrush(QBrush(QColor(0, 255, 0)))  # Green for start

    def set_as_end(self):
        """Mark the node as the end (red)."""
        self.is_end = True
        self.setBrush(QBrush(QColor(255, 0, 0)))  # Red for end

    def set_as_aisle(self, color=None):
        """Mark the node as an aisle with specified color."""
        if color is None:
            color = QColor(150, 150, 250)  # Default light blue for vertical aisles
        self.setBrush(QBrush(color))
        self.is_aisle = True
        self.is_obstacle = True  # Aisles are obstacles
        # Store the orientation type for reset
        if color == QColor(150, 150, 250):
            self.parent_window.orientation_type = 'vertical'
        else:
            self.parent_window.orientation_type = 'horizontal'


class WarehouseVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.num_aisles = 5  # Default number of aisles
        self.max_shelves_per_aisle = 10  # Default max shelves per aisle

        self.node_size = 80
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene, self)

        self.grid = []
        self.spacing = 1  # Default spacing between aisles (hallway size)
        self.current_mode = None  # Track mode for start, end, or barrier selection

        # Directory to store scenarios
        self.scenario_dir = "scenarios"
        if not os.path.exists(self.scenario_dir):
            os.makedirs(self.scenario_dir)

        # Dropdown to adjust spacing between aisles
        self.spacing_dropdown = QComboBox(self)
        self.spacing_dropdown.addItems([str(i) for i in range(1, 4)])  # Dropdown for spacing (1 to 3)
        self.spacing_dropdown.currentIndexChanged.connect(self.update_spacing)

        # Dropdown to select the algorithm
        self.algorithm_dropdown = QComboBox(self)
        self.algorithm_dropdown.addItems(
            ["Manhattan Distance", "Euclidean Distance", "Modified Euclidean (1.2x Y Priority)",
             "Aisle-Aware Heuristic", "BFS", "DFS", "A* with Branch Pruning", "Jump Point Search"])

        # Dropdown to select the warehouse layout
        self.layout_dropdown = QComboBox(self)
        self.layout_dropdown.addItems(["Vertical Aisles", "Horizontal Aisles", "Mixed Aisles"])

        # Spin boxes to adjust the number of aisles and shelves
        self.aisle_spinbox = QSpinBox(self)
        self.aisle_spinbox.setRange(1, 20)
        self.aisle_spinbox.setValue(self.num_aisles)
        self.aisle_spinbox.valueChanged.connect(self.update_num_aisles)

        self.shelf_spinbox = QSpinBox(self)
        self.shelf_spinbox.setRange(1, 50)
        self.shelf_spinbox.setValue(self.max_shelves_per_aisle)
        self.shelf_spinbox.valueChanged.connect(self.update_num_shelves)

        # Buttons for selecting start, end, and barrier modes
        self.start_button = QPushButton("Set Start", self)
        self.start_button.clicked.connect(self.set_mode_start)

        self.end_button = QPushButton("Set End", self)
        self.end_button.clicked.connect(self.set_mode_end)

        self.barrier_button = QPushButton("Set Barriers", self)
        self.barrier_button.clicked.connect(self.set_mode_barrier)

        # Search button
        self.search_button = QPushButton("Search Path", self)
        self.search_button.clicked.connect(self.handle_search)

        # Button to generate the layout
        self.generate_button = QPushButton("Generate Warehouse Layout", self)
        self.generate_button.clicked.connect(self.handle_generate_layout)

        # Clear all button
        self.clear_button = QPushButton("Clear All", self)
        self.clear_button.clicked.connect(self.clear_all)

        # Save scenario button
        self.save_button = QPushButton("Save Scenario", self)
        self.save_button.clicked.connect(self.save_scenario)

        # Load scenario dropdown
        self.load_dropdown = QComboBox(self)
        self.load_dropdown.addItem("Select Scenario")  # Default placeholder
        self.load_dropdown.currentIndexChanged.connect(self.load_scenario)

        # Counter label for searched nodes
        self.counter_label = QLabel("Nodes Searched: 0", self)

        # Layout setup
        layout = QVBoxLayout()
        layout.addWidget(self.view)
        layout.addWidget(QLabel("Select Aisle Spacing:", self))
        layout.addWidget(self.spacing_dropdown)
        layout.addWidget(QLabel("Select Warehouse Layout:", self))
        layout.addWidget(self.layout_dropdown)
        layout.addWidget(QLabel("Number of Aisles:", self))
        layout.addWidget(self.aisle_spinbox)
        layout.addWidget(QLabel("Max Shelves per Aisle:", self))
        layout.addWidget(self.shelf_spinbox)
        layout.addWidget(QLabel("Select Algorithm:", self))
        layout.addWidget(self.algorithm_dropdown)
        layout.addWidget(self.start_button)
        layout.addWidget(self.end_button)
        layout.addWidget(self.barrier_button)
        layout.addWidget(self.search_button)
        layout.addWidget(self.generate_button)
        layout.addWidget(self.save_button)
        layout.addWidget(QLabel("Load Scenario:", self))
        layout.addWidget(self.load_dropdown)
        layout.addWidget(self.clear_button)
        layout.addWidget(self.counter_label)
        # Add zoom buttons
        self.zoom_in_button = QPushButton("Zoom In", self)
        self.zoom_in_button.clicked.connect(self.zoom_in)
        layout.addWidget(self.zoom_in_button)

        self.zoom_out_button = QPushButton("Zoom Out", self)
        self.zoom_out_button.clicked.connect(self.zoom_out)
        layout.addWidget(self.zoom_out_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.grid_size = 12  # Initial grid size
        self.init_grid()

        self.load_scenarios()  # Load existing scenarios into the dropdown

        self.show()

    def zoom_in(self):
        """Zoom in the view by scaling up."""
        self.view.scale(1.2, 1.2)  # Scale up by 20%

    def zoom_out(self):
        """Zoom out the view by scaling down."""
        self.view.scale(1 / 1.2, 1 / 1.2)  # Scale down by 20%

    def adjust_zoom(self):
        """Adjust the zoom level to fit the entire grid within the view."""
        # Reset any existing transformations
        self.view.resetTransform()

        # Calculate the total size of the grid
        total_width = self.grid_size * self.node_size
        total_height = self.grid_size * self.node_size

        # Get the size of the view
        view_width = self.view.viewport().width()
        view_height = self.view.viewport().height()

        # Calculate the scaling factors
        scale_x = view_width / total_width
        scale_y = view_height / total_height

        # Use the smaller scaling factor to fit both dimensions
        scale = min(scale_x, scale_y) * 0.9  # Use 90% to add some padding

        self.view.scale(scale, scale)

    def init_grid(self):
        """Initialize an empty grid of nodes with a buffer around the warehouse."""
        # Clear the existing scene and grid
        self.scene.clear()
        self.grid = []

        for y in range(self.grid_size):
            row = []
            for x in range(self.grid_size):
                node_name = f"Node ({x},{y})"
                node = Node(x, y, self.node_size, node_name, self)
                self.scene.addItem(node)
                row.append(node)
            self.grid.append(row)

    def reset_grid(self):
        """Reset the path and visited nodes but keep start, end, and barriers."""
        for row in self.grid:
            for node in row:
                if not node.is_start and not node.is_end and not node.is_obstacle:
                    node.reset()

    def clear_all(self):
        """Clear the entire grid, including start, end, and barriers."""
        for row in self.grid:
            for node in row:
                node.is_start = False
                node.is_end = False
                node.is_obstacle = False
                node.is_aisle = False
                node.reset()
        self.start_node = None
        self.end_node = None

    def update_spacing(self):
        """Update the spacing between aisles based on dropdown selection."""
        self.spacing = int(self.spacing_dropdown.currentText())

    def update_num_aisles(self, value):
        self.num_aisles = value

    def update_num_shelves(self, value):
        self.max_shelves_per_aisle = value

    def handle_generate_layout(self):
        selected_layout = self.layout_dropdown.currentText()
        if selected_layout == "Vertical Aisles":
            orientation = 'vertical'
        elif selected_layout == "Horizontal Aisles":
            orientation = 'horizontal'
        elif selected_layout == "Mixed Aisles":
            orientation = 'mixed'
        else:
            orientation = 'vertical'  # Default to vertical
        self.generate_warehouse_layout(orientation)

    def generate_warehouse_layout(self, orientation='vertical'):
        """Generate the warehouse layout with adjustable aisle spacing and label items."""
        self.clear_all()

        # Generate warehouse data
        warehouse_data = generate_warehouse_data(
            num_aisles=self.num_aisles,
            max_shelves_per_aisle=self.max_shelves_per_aisle,
            save_to_csv=False
        )

        aisles = warehouse_data['Aisle_Number'].nunique()
        aisle_positions = []

        vertical_positions = set()
        horizontal_positions = set()

        if orientation == 'vertical':
            for aisle_num in range(1, aisles + 1):
                x = 2 + (aisle_num - 1) * (self.spacing + 1)
                vertical_positions.add(x)
                aisle_positions.append(('vertical', x))
            max_x = max(vertical_positions) + 2
            max_y = 2 + self.max_shelves_per_aisle
        elif orientation == 'horizontal':
            for aisle_num in range(1, aisles + 1):
                y = 2 + (aisle_num - 1) * (self.spacing + 1)
                horizontal_positions.add(y)
                aisle_positions.append(('horizontal', y))
            max_x = 2 + self.max_shelves_per_aisle
            max_y = max(horizontal_positions) + 2
        elif orientation == 'mixed':
            num_vertical_aisles = (aisles + 1) // 2
            num_horizontal_aisles = aisles // 2

            for i in range(num_vertical_aisles):
                x = 2 + i * (self.spacing + 2)  # Ensure 1-block space by adding 2
                vertical_positions.add(x)
                aisle_positions.append(('vertical', x))

            for i in range(num_horizontal_aisles):
                y = 2 + i * (self.spacing + 2)
                # Adjust y to ensure it's offset from x positions
                while y in vertical_positions or y + 1 in vertical_positions:
                    y += 1  # Shift y to avoid overlap and maintain 1-block space
                horizontal_positions.add(y)
                aisle_positions.append(('horizontal', y))

            max_x = max(max(vertical_positions, default=0), 2 + self.max_shelves_per_aisle) + 2
            max_y = max(max(horizontal_positions, default=0), 2 + self.max_shelves_per_aisle) + 2
        else:
            max_x = max_y = 12  # Default grid size

        # Update grid size
        self.grid_size = max(int(max_x), int(max_y)) + 2  # Add buffer
        self.init_grid()  # Reinitialize the grid with new size

        for i, row in warehouse_data.iterrows():
            aisle_num = int(row['Aisle_Number'].split('_')[1])  # Extract aisle number
            shelf_num = int(row['Shelf_Number'].split('_')[1])  # Extract shelf number
            shelf_loc = row['Shelf_Location']
            item = row['Item'] if row['Item'] is not None else "Empty"

            # Skip if aisle number exceeds available positions
            if aisle_num - 1 >= len(aisle_positions):
                continue

            orientation_type, pos = aisle_positions[aisle_num - 1]

            if orientation_type == 'vertical':
                x = pos
                y = 2 + (shelf_num - 1)  # Adjust for grid layout (y-axis for shelves)
                aisle_color = QColor(150, 150, 250)  # Light blue
            else:  # Horizontal
                x = 2 + (shelf_num - 1)
                y = pos
                aisle_color = QColor(150, 0, 250)  # Purple color for horizontal aisles

            # Ensure coordinates are within grid bounds
            if x >= self.grid_size or y >= self.grid_size:
                continue  # Skip if out of bounds

            # Set the node as aisle with appropriate color
            self.grid[y][x].set_as_aisle(aisle_color)

            # Define the item number
            item_number = item.split('_')[-1] if item != "Empty" else "Empty"

            # Add the item label to the node (only the number part of the item)
            label = QGraphicsTextItem(item_number)

            # Scale down the font size
            font = QFont()
            font.setPointSize(8)  # You can adjust the font size here
            label.setFont(font)

            # Adjust the position of the label based on the shelf location
            padding = 5  # Small padding for neat positioning

            node_x = self.grid[y][x].pos().x()
            node_y = self.grid[y][x].pos().y()

            if shelf_loc == 'A':
                label.setPos(node_x + padding, node_y + padding)  # Top-left
            elif shelf_loc == 'B':
                label.setPos(node_x + self.node_size / 2 + padding, node_y + padding)  # Top-right
            elif shelf_loc == 'C':
                label.setPos(node_x + padding, node_y + self.node_size / 2 + padding)  # Bottom-left
            elif shelf_loc == 'D':
                label.setPos(node_x + self.node_size / 2 + padding, node_y + self.node_size / 2 + padding)  # Bottom-right

            self.scene.addItem(label)

        self.adjust_zoom()  # Adjust the zoom level

    def set_mode_start(self):
        """Set mode to start node selection."""
        self.current_mode = 'start'

    def set_mode_end(self):
        """Set mode to end node selection."""
        self.current_mode = 'end'

    def set_mode_barrier(self):
        """Set mode to barrier node selection."""
        self.current_mode = 'barrier'

    def set_start_node(self, node):
        """Set the selected node as the start node."""
        if hasattr(self, 'start_node') and self.start_node:
            self.start_node.is_start = False  # Reset the previous start node
            self.start_node.reset()
        self.start_node = node
        self.start_node.set_as_start()

    def set_end_node(self, node):
        """Set the selected node as the end node."""
        if hasattr(self, 'end_node') and self.end_node:
            self.end_node.is_end = False  # Reset the previous end node
            self.end_node.reset()
        self.end_node = node
        self.end_node.set_as_end()

    def handle_search(self):
        """Handle search between start and end nodes."""
        # Check if start and end nodes are set
        if not hasattr(self, 'start_node') or self.start_node is None:
            self.counter_label.setText("Start node not set.")
            return
        if not hasattr(self, 'end_node') or self.end_node is None:
            self.counter_label.setText("End node not set.")
            return

        # Clear previous path and visited nodes
        self.reset_grid()

        # Get the start and end node coordinates
        start = (int(self.start_node.pos().x() // self.node_size),
                 int(self.start_node.pos().y() // self.node_size))
        end = (int(self.end_node.pos().x() // self.node_size),
               int(self.end_node.pos().y() // self.node_size))

        # Determine which algorithm to run
        selected_algorithm = self.algorithm_dropdown.currentText()

        if selected_algorithm == "Jump Point Search":
            path = self.run_jps(start, end)
        elif selected_algorithm == "BFS":
            path = self.bfs(start, end)
        elif selected_algorithm == "DFS":
            path = self.dfs(start, end)
        elif selected_algorithm == "A* with Branch Pruning":
            path = self.run_astar_with_pruning(start, end)
        else:
            path = self.run_astar(start, end)

        if path:
            self.search_path = path
            self.visualize_path_step_by_step()
        else:
            self.counter_label.setText("No path found.")

    def run_astar(self, start, end):
        """Runs the A* algorithm to find the path between start and end nodes."""
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, end)}
        self.nodes_searched = 0  # Reset node search count

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                self.counter_label.setText(f"Nodes Searched: {self.nodes_searched}")
                return self.reconstruct_path(came_from, current)

            # Mark the node as visited visually and count it
            self.grid[current[1]][current[0]].set_visited()
            self.nodes_searched += 1
            self.counter_label.setText(f"Nodes Searched: {self.nodes_searched}")
            QApplication.processEvents()  # Update the UI in real-time

            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        self.counter_label.setText("No path found.")
        return None  # If no path found, return None

    def run_jps(self, start, end):
        """Runs the Jump Point Search algorithm."""
        self.start_pos = start
        self.end_pos = end
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        self.nodes_searched = 0

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                return self.reconstruct_path(came_from, current)

            self.grid[current[1]][current[0]].set_visited()
            self.nodes_searched += 1
            self.counter_label.setText(f"Nodes Searched: {self.nodes_searched}")
            QApplication.processEvents()

            neighbors = self.identify_successors(current)

            for neighbor, cost in neighbors:
                tentative_g_score = g_score[current] + cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, neighbor))

        self.counter_label.setText("No path found.")
        return None

    def identify_successors(self, node):
        """Identify successors in Jump Point Search."""
        successors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-directional movement

        for direction in directions:
            jump_point = self.jump(node, direction)
            if jump_point:
                cost = self.distance(node, jump_point)
                successors.append((jump_point, cost))
        return successors

    def jump(self, current, direction):
        x, y = current
        dx, dy = direction

        next_x, next_y = x + dx, y + dy

        if not self.is_valid_position(next_x, next_y) or self.is_obstacle(next_x, next_y):
            return None

        if (next_x, next_y) == self.end_pos:
            return (next_x, next_y)

        # Forced neighbor detection
        if dx != 0:
            # Moving horizontally
            if (self.is_traversable(next_x, y - 1) and not self.is_traversable(x, y - 1)) or \
                    (self.is_traversable(next_x, y + 1) and not self.is_traversable(x, y + 1)):
                return (next_x, next_y)
        elif dy != 0:
            # Moving vertically
            if (self.is_traversable(x - 1, next_y) and not self.is_traversable(x - 1, y)) or \
                    (self.is_traversable(x + 1, next_y) and not self.is_traversable(x + 1, y)):
                return (next_x, next_y)

        # Continue moving in the same direction
        return self.jump((next_x, next_y), direction)

    def is_valid_position(self, x, y):
        """Check if position is within grid bounds."""
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def is_obstacle(self, x, y):
        """Check if position is an obstacle."""
        if not self.is_valid_position(x, y):
            return True
        return self.grid[y][x].is_obstacle

    def is_traversable(self, x, y):
        """Check if the position is within bounds and not an obstacle."""
        return self.is_valid_position(x, y) and not self.is_obstacle(x, y)

    def distance(self, node_a, node_b):
        """Calculate distance between two nodes."""
        (x1, y1) = node_a
        (x2, y2) = node_b
        return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def is_near_aisle(self, node):
        """Check if the node is near an aisle and return True if so."""
        (x, y) = node
        # Check adjacent nodes to see if any are aisles
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        for nx, ny in neighbors:
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                if self.grid[ny][nx].is_aisle:
                    return True
        return False

    def bfs(self, start, end):
        """Breadth-First Search algorithm to find the shortest path."""
        queue = [(start, [])]  # Queue of tuples (node, path)
        visited = set()
        self.nodes_searched = 0  # Reset node search count

        while queue:
            (current, path) = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            # Mark the node as visited visually and count it
            self.grid[current[1]][current[0]].set_visited()
            self.nodes_searched += 1
            self.counter_label.setText(f"Nodes Searched: {self.nodes_searched}")
            QApplication.processEvents()  # Update the UI in real-time

            path = path + [current]

            if current == end:
                return path

            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    queue.append((neighbor, path))

        return None  # No path found

    def dfs(self, start, end):
        """Depth-First Search algorithm to find the path."""
        stack = [(start, [])]  # Stack of tuples (node, path)
        visited = set()
        self.nodes_searched = 0  # Reset node search count

        while stack:
            (current, path) = stack.pop()
            if current in visited:
                continue
            visited.add(current)

            # Mark the node as visited visually and count it
            self.grid[current[1]][current[0]].set_visited()
            self.nodes_searched += 1
            self.counter_label.setText(f"Nodes Searched: {self.nodes_searched}")
            QApplication.processEvents()  # Update the UI in real-time

            path = path + [current]

            if current == end:
                return path

            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    stack.append((neighbor, path))

        return None  # No path found

    def heuristic(self, node, goal):
        """Heuristic function that supports different types of heuristics, including custom ones."""
        (x1, y1) = node
        (x2, y2) = goal

        selected_distance = self.algorithm_dropdown.currentText()

        # Base heuristics (Manhattan, Euclidean, Modified Euclidean)
        if selected_distance == "Manhattan Distance":
            return abs(x1 - x2) + abs(y1 - y2)

        elif selected_distance == "Euclidean Distance":
            return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        elif selected_distance == "Modified Euclidean (1.2x Y Priority)":
            return sqrt((x1 - x2) ** 2 + (1.2 * (y1 - y2)) ** 2)

        # Custom heuristic: Aisle-Aware Heuristic
        elif selected_distance == "Aisle-Aware Heuristic":
            # Base heuristic can be Manhattan or Euclidean, let's use Manhattan
            base_heuristic = abs(x1 - x2) + abs(y1 - y2)

            # Add a penalty if the node is near an aisle
            penalty = 0
            if self.is_near_aisle(node):
                penalty = 10  # Example penalty value, you can adjust this

            return base_heuristic + penalty

        # Default fallback to Manhattan distance if no valid heuristic is selected
        return abs(x1 - x2) + abs(y1 - y2)

    def get_neighbors(self, node):
        """Get the valid neighbors of the current node."""
        (x, y) = node
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                if not self.grid[ny][nx].is_obstacle:
                    neighbors.append((nx, ny))
        return neighbors

    def reconstruct_path(self, came_from, current):
        """Reconstructs the path from the A* or JPS result."""
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(current)
        path.reverse()
        return path

    def visualize_path_step_by_step(self):
        """Visualize the path traversal with a delay."""
        self.step_index = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_step)
        self.timer.start(200)  # 200 ms delay

        # Display the total path length in addition to nodes searched
        total_path_length = len(self.search_path)
        self.counter_label.setText(f"Nodes Searched: {self.nodes_searched}, Path Length: {total_path_length}")

    def update_step(self):
        """Update the grid one step at a time."""
        if self.step_index < len(self.search_path):
            x, y = self.search_path[self.step_index]
            if (x, y) != (int(self.start_node.pos().x() // self.node_size),
                          int(self.start_node.pos().y() // self.node_size)):  # Don't change the start node color
                self.grid[y][x].set_path()
            self.step_index += 1
        else:
            self.timer.stop()  # Stop once the entire path is visualized

    def save_scenario(self):
        """Save the current warehouse layout and state as a scenario."""
        # Prompt the user to enter a scenario name
        scenario_name, ok = QInputDialog.getText(self, "Save Scenario", "Enter scenario name:")
        if ok and scenario_name:
            # Prepare the scenario data
            scenario_data = {
                'grid_size': self.grid_size,
                'spacing': self.spacing,
                'num_aisles': self.num_aisles,
                'max_shelves_per_aisle': self.max_shelves_per_aisle,
                'layout_type': self.layout_dropdown.currentText(),
                'start_node': self.get_node_position(self.start_node) if hasattr(self, 'start_node') else None,
                'end_node': self.get_node_position(self.end_node) if hasattr(self, 'end_node') else None,
                'nodes': []
            }

            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    node = self.grid[y][x]
                    node_data = {
                        'x': x,
                        'y': y,
                        'is_obstacle': node.is_obstacle,
                        'is_aisle': node.is_aisle,
                        'color': node.brush().color().name()
                    }
                    scenario_data['nodes'].append(node_data)

            # Save the scenario data to a JSON file
            scenario_file = os.path.join(self.scenario_dir, f"{scenario_name}.json")
            with open(scenario_file, 'w') as f:
                json.dump(scenario_data, f)

            # Update the load dropdown
            self.load_scenarios()
            QMessageBox.information(self, "Scenario Saved", f"Scenario '{scenario_name}' saved successfully.")

    def get_node_position(self, node):
        """Get the grid coordinates of a node."""
        if node:
            x = int(node.pos().x() // self.node_size)
            y = int(node.pos().y() // self.node_size)
            return {'x': x, 'y': y}
        return None

    def load_scenario(self):
        """Load the selected scenario from the dropdown."""
        scenario_name = self.load_dropdown.currentText()
        if scenario_name == "Select Scenario":
            return

        scenario_file = os.path.join(self.scenario_dir, f"{scenario_name}.json")
        if not os.path.exists(scenario_file):
            QMessageBox.warning(self, "Error", f"Scenario file '{scenario_name}' not found.")
            return

        with open(scenario_file, 'r') as f:
            scenario_data = json.load(f)

        # Load the scenario data
        self.grid_size = scenario_data['grid_size']
        self.spacing = scenario_data['spacing']
        self.num_aisles = scenario_data['num_aisles']
        self.max_shelves_per_aisle = scenario_data['max_shelves_per_aisle']

        # Update UI elements
        self.spacing_dropdown.setCurrentText(str(self.spacing))
        self.aisle_spinbox.setValue(self.num_aisles)
        self.shelf_spinbox.setValue(self.max_shelves_per_aisle)
        self.layout_dropdown.setCurrentText(scenario_data['layout_type'])

        # Reinitialize the grid
        self.init_grid()

        # Set nodes
        for node_data in scenario_data['nodes']:
            x = node_data['x']
            y = node_data['y']
            node = self.grid[y][x]
            node.is_obstacle = node_data['is_obstacle']
            node.is_aisle = node_data['is_aisle']
            color = QColor(node_data['color'])
            node.setBrush(QBrush(color))

        # Set start and end nodes
        self.start_node = None
        self.end_node = None

        if scenario_data['start_node']:
            x = scenario_data['start_node']['x']
            y = scenario_data['start_node']['y']
            self.start_node = self.grid[y][x]
            self.start_node.set_as_start()

        if scenario_data['end_node']:
            x = scenario_data['end_node']['x']
            y = scenario_data['end_node']['y']
            self.end_node = self.grid[y][x]
            self.end_node.set_as_end()

        self.adjust_zoom()  # Adjust the zoom level

        QMessageBox.information(self, "Scenario Loaded", f"Scenario '{scenario_name}' loaded successfully.")

    def load_scenarios(self):
        """Load the list of saved scenarios into the dropdown."""
        self.load_dropdown.blockSignals(True)  # Temporarily block signals to prevent triggering load_scenario

        self.load_dropdown.clear()
        self.load_dropdown.addItem("Select Scenario")  # Default placeholder

        scenario_files = [f for f in os.listdir(self.scenario_dir) if f.endswith('.json')]
        scenario_names = [os.path.splitext(f)[0] for f in scenario_files]

        self.load_dropdown.addItems(scenario_names)

        self.load_dropdown.blockSignals(False)  # Re-enable signals


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WarehouseVisualizer()
    sys.exit(app.exec())
