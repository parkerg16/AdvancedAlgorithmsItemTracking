import sys
import heapq
import pandas as pd
import random
from math import sqrt
from datetime import datetime
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsRectItem,
    QGraphicsTextItem, QVBoxLayout, QWidget, QPushButton, QLabel, QComboBox, QSpinBox,
    QFileDialog, QInputDialog, QMessageBox, QCheckBox, QGraphicsItem, QHBoxLayout
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QFont
import os
import json
import time  # Import time module for benchmarking


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
        """
        Initialize a Node instance.

        Parameters:
            x (int): The x-coordinate in the grid.
            y (int): The y-coordinate in the grid.
            size (int): The size of the node (width and height).
            name (str): The name identifier for the node.
            parent_window (WarehouseVisualizer): Reference to the main window.
        """
        super().__init__(0, 0, size, size)
        self.setPos(x * size, y * size)
        self.name = name
        self.setBrush(QBrush(QColor(255, 255, 255)))  # Default color: White
        self.parent_window = parent_window
        self.is_obstacle = False
        self.is_start = False
        self.is_end = False
        self.edge_weight = 1  # Default edge weight
        self.is_aisle = False
        self.last_scroll_time = time.time()  # For throttling wheel events
        self.original_aisle_color = None
        # Enable focus to capture wheel events
        self.setFlags(QGraphicsRectItem.GraphicsItemFlag.ItemIsFocusable)

        # Initialize edge weight label
        self.weight_label = QGraphicsTextItem(str(self.edge_weight), self)
        weight_font = QFont()
        weight_font.setPointSize(10)
        weight_font.setBold(True)
        self.weight_label.setFont(weight_font)

        # Center the weight text within the node
        self.update_weight_label_position()

        # Initialize item label (for shelf items)
        self.item_label = QGraphicsTextItem("", self)
        item_font = QFont()
        item_font.setPointSize(8)
        self.item_label.setFont(item_font)
        self.item_label.setDefaultTextColor(QColor(0, 0, 0))  # Black color for item labels

        # Initially hide the weight label if the node is an aisle
        if self.is_aisle:
            self.weight_label.hide()
        if self.is_aisle:
            self.item_label.hide()

    def update_weight_label_position(self):
        """Center the weight label within the node."""
        text_rect = self.weight_label.boundingRect()
        node_rect = self.rect()
        self.weight_label.setPos(
            (node_rect.width() - text_rect.width()) / 2,
            (node_rect.height() - text_rect.height()) / 2
        )

    def update_item_label_position(self):
        """Center the item label within the node."""
        text_rect = self.item_label.boundingRect()
        node_rect = self.rect()
        self.item_label.setPos(
            (node_rect.width() - text_rect.width()) / 2,
            (node_rect.height() - text_rect.height()) / 2
        )

    def detect_negative_cycle_spfa(self, x, y, new_weight):
        """
        Check if changing the weight of node at (x,y) to new_weight would create a negative cycle.
        Uses SPFA (Shortest Path Faster Algorithm) for efficient negative cycle detection.

        Parameters:
            x (int): X coordinate of the node being changed
            y (int): Y coordinate of the node being changed
            new_weight (float): The new weight to test

        Returns:
            bool: True if a negative cycle would be created, False otherwise
        """
        # Store the original weight
        original_weight = self.edge_weight

        # Temporarily set the new weight
        self.edge_weight = new_weight

        # Get grid size from parent window
        grid_size = self.parent_window.grid_size

        # Initialize distance and count arrays
        distance = {(i, j): float('inf') for i in range(grid_size) for j in range(grid_size)}
        count = {(i, j): 0 for i in range(grid_size) for j in range(grid_size)}

        # Start from the changed node
        start = (x, y)
        distance[start] = 0

        # Initialize queue with start node
        queue = [start]
        in_queue = {(i, j): False for i in range(grid_size) for j in range(grid_size)}
        in_queue[start] = True

        def get_neighbors(node):
            px, py = node
            neighbors = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4-directional neighbors
                nx, ny = px + dx, py + dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    if not self.parent_window.grid[ny][nx].is_obstacle:
                        neighbors.append((nx, ny))
            return neighbors

        # Run SPFA
        while queue:
            current = queue.pop(0)
            in_queue[current] = False

            # Check each neighbor
            for next_node in get_neighbors(current):
                weight = self.parent_window.grid[next_node[1]][next_node[0]].edge_weight
                if distance[current] + weight < distance[next_node]:
                    distance[next_node] = distance[current] + weight
                    count[next_node] += 1

                    # If a node has been relaxed more times than number of nodes,
                    # we've found a negative cycle
                    if count[next_node] >= grid_size * grid_size:
                        # Restore original weight before returning
                        self.edge_weight = original_weight
                        return True

                    if not in_queue[next_node]:
                        queue.append(next_node)
                        in_queue[next_node] = True

        # Restore original weight before returning
        self.edge_weight = original_weight
        return False

    def set_edge_weight(self, new_weight):
        """
        Set edge weight and check for negative cycles only under specific conditions.
        """
        x = int(self.pos().x() / self.parent_window.node_size)
        y = int(self.pos().y() / self.parent_window.node_size)

        # Trigger cycle check only if necessary
        if (
                new_weight < 0 and
                not self.is_aisle and
                not self.parent_window.is_generating_warehouse and
                (self.edge_weight >= 0 or new_weight < self.edge_weight)  # If newly negative or decreasing further
        ):
            # Check for negative cycles
            if self.detect_negative_cycle_spfa(x, y, new_weight):
                # Revert to a safe value if a cycle is detected
                new_weight = max(1, self.edge_weight)
                QMessageBox.warning(
                    self.parent_window,
                    "Invalid Weight",
                    "This weight change would create a negative cycle and has been reverted."
                )

        # Update the edge weight and appearance
        self.edge_weight = new_weight
        self.update_color_from_weight()

    def update_color_from_weight(self):
        """
        Update node color based on its current edge weight without cycle detection.
        """
        if not self.is_aisle:
            red_intensity = max(0, 255 - (self.edge_weight - 1) * 25)
            self._color = QColor(255, red_intensity, red_intensity)
            self.setBrush(QBrush(self._color))

            if self.weight_label:
                self.weight_label.setPlainText(str(self.edge_weight))
                self.update_weight_label_position()

    def set_item_label(self, text):
        """
        Set the item label text.

        Parameters:
            text (str): The text to display as the item label.
        """
        if self.is_aisle:
            self.item_label.hide()
            return

        self.item_label.setPlainText(text)
        self.update_item_label_position()
        if text:
            self.item_label.show()
        else:
            self.item_label.hide()

    def wheelEvent(self, event):
        """
        Handle mouse wheel events to adjust the edge weight with throttling.
        """
        try:
            current_time = time.time()
            if current_time - self.last_scroll_time > 0.1:  # 100ms delay
                delta_y = event.delta()
                if delta_y != 0:
                    delta = delta_y / 120  # Standard scroll value
                    old_weight = self.edge_weight
                    new_weight = min(10, max(-10, self.edge_weight + int(delta)))
                    self.set_edge_weight(new_weight)  # Use new method
                    print(f"Edge weight changed from {old_weight} to {self.edge_weight}")
                self.last_scroll_time = current_time
        except Exception as e:
            print(f"Error in wheelEvent: {e}")

    def mousePressEvent(self, event):
        """
        Handle mouse press events to set the node as start, end, or barrier based on current mode.

        Parameters:
            event (QGraphicsSceneMouseEvent): The mouse press event.
        """
        mode = self.parent_window.current_mode
        if mode == 'start':
            self.parent_window.set_start_node(self)
        elif mode == 'end':
            self.parent_window.set_end_node(self)
        elif mode == 'barrier':
            self.set_as_barrier()

    def mouseMoveEvent(self, event):
        """
        Handle mouse move events to set nodes as barriers when in barrier mode.

        Parameters:
            event (QGraphicsSceneMouseEvent): The mouse move event.
        """
        mode = self.parent_window.current_mode
        if mode == 'barrier':
            self.set_as_barrier()

    def set_as_barrier(self):
        """
        Mark the node as a barrier (yellow) unless it's an aisle, start, or end node.
        """
        if not self.is_aisle and not self.is_start and not self.is_end:
            self.setBrush(QBrush(QColor(255, 255, 0)))  # Yellow for barriers
            self.is_obstacle = True
            if self.weight_label:
                self.weight_label.hide()  # Hide weight label when node becomes a barrier
            if self.item_label:
                self.item_label.hide()  # Hide item label when node becomes a barrier

    def set_relaxed(self):
        """Mark the node as relaxed (orange) during Bellman-Ford relaxation."""
        if not self.is_aisle:
            self.setBrush(QBrush(QColor(255, 165, 0)))  # Orange for relaxation
        else:
            if not self.is_start and not self.is_end:
                self.set_as_aisle(self.brush().color())

    def set_visited(self):
        """Mark the node as visited (green) unless it's an aisle."""
        if not self.is_aisle:
            self.setBrush(QBrush(QColor(144, 238, 144)))  # Light green
        else:
            if not self.is_start and not self.is_end:
                self.set_as_aisle(self.brush().color())

    def set_path(self):
        """Mark the node as part of the path (blue) unless it's an aisle."""
        if not self.is_aisle:
            self.setBrush(QBrush(QColor(0, 0, 255)))  # Blue when part of the path
        else:
            if not self.is_start and not self.is_end:
                self.set_as_aisle(self.brush().color())  # Retain aisle color
            # If it's start or end, do not change the color

    def reset(self):
        """Reset the node to its default visual state."""
        if not self.is_obstacle and not self.is_start and not self.is_end:
            if not self.is_aisle:
                self._color = QColor(255, 255, 255)  # White
                self.setBrush(QBrush(self._color))
                if self.weight_label:
                    self.weight_label.show()
                if self.item_label:
                    self.item_label.show()
                self.update_color_from_weight()  # Update color based on current weight
            else:
                if self.original_aisle_color:
                    self.set_as_aisle(self.original_aisle_color)
                else:
                    self.set_as_aisle(QColor(150, 150, 250))

    def set_as_start(self):
        """
        Set the node as the start node (green) and update relevant states.
        """
        # Prevent setting the same node as both start and end
        if self.parent_window.end_node == self:
            return

        # Reset previous start node if it exists
        if self.parent_window.start_node:
            self.parent_window.start_node.is_start = False
            self.parent_window.start_node.reset()

        # Set the new start node
        self.is_start = True
        self.is_obstacle = False  # Start node should not be an obstacle
        self.setBrush(QBrush(QColor(0, 255, 0)))  # Green for start
        if self.weight_label:
            self.weight_label.hide()  # Hide weight label when node becomes start
        if self.item_label:
            self.item_label.hide()  # Hide item label when node becomes start

        # Update reference in parent window
        self.parent_window.start_node = self

    def set_as_end(self):
        """
        Set the node as the end node (red) and update relevant states.
        """
        # Prevent setting the same node as both start and end
        if self.parent_window.start_node == self:
            return

        # Reset previous end node if it exists
        if self.parent_window.end_node:
            self.parent_window.end_node.is_end = False
            self.parent_window.end_node.reset()

        # Set the new end node
        self.is_end = True
        self.is_obstacle = False  # End node should not be an obstacle
        self.setBrush(QBrush(QColor(255, 0, 0)))  # Red for end
        if self.weight_label:
            self.weight_label.hide()  # Hide weight label when node becomes end
        if self.item_label:
            self.item_label.hide()  # Hide item label when node becomes end

        # Update reference in parent window
        self.parent_window.end_node = self

    def set_as_aisle(self, aisle_color):
        """
        Set the node as an aisle with the specified color.

        Parameters:
            aisle_color (QColor): The color to set for the aisle node.
        """
        self.original_aisle_color = aisle_color
        if self.is_start:
            self.setBrush(QBrush(QColor(0, 255, 0)))  # Green for start
        elif self.is_end:
            self.setBrush(QBrush(QColor(255, 0, 0)))  # Red for end
        else:
            self.setBrush(QBrush(aisle_color))  # Use the passed aisle color
        self.is_aisle = True
        if self.weight_label:
            self.weight_label.hide()
        if self.item_label:
            self.item_label.hide()


class WarehouseVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.is_generating_warehouse = False

        self.num_aisles = 5  # Default number of aisles
        self.max_shelves_per_aisle = 10  # Default max shelves per aisle

        self.node_size = 80
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene, self)

        self.grid = []
        self.spacing = 1  # Default spacing between aisles (hallway size)
        self.current_mode = None  # Track mode for start, end, or barrier selection
        self.start_node = None
        self.end_node = None

        self.orientation_type = 'vertical'  # Default value
        # Directory to store scenarios
        self.scenario_dir = "scenarios"
        if not os.path.exists(self.scenario_dir):
            os.makedirs(self.scenario_dir)

        self.current_diagonal_state = False

        # Initialize all UI elements
        # Dropdowns
        self.spacing_dropdown = QComboBox(self)
        self.spacing_dropdown.addItems([str(i) for i in range(1, 4)])
        self.spacing_dropdown.currentIndexChanged.connect(self.update_spacing)

        self.algorithm_dropdown = QComboBox(self)
        self.algorithm_dropdown.addItems([
            "A* (Manhattan Distance)",
            "A* (Euclidean Distance)",
            "A* (Modified Euclidean 1.2x Y Priority)",
            "Dijkstra's",
            "Bellman-Ford",
            "SPFA",
            "Johnson's",
            "Johnson's with A*",

        ])

        self.layout_dropdown = QComboBox(self)
        self.layout_dropdown.addItems(["Vertical Aisles", "Horizontal Aisles", "Mixed Aisles"])

        self.load_dropdown = QComboBox(self)
        self.load_dropdown.addItem("Select Scenario")
        self.load_dropdown.currentIndexChanged.connect(self.load_scenario)

        self.item_dropdown = QComboBox(self)
        self.item_dropdown.addItem("Select Item")
        self.item_dropdown.currentIndexChanged.connect(self.set_end_node_from_dropdown)

        # Spin boxes
        self.aisle_spinbox = QSpinBox(self)
        self.aisle_spinbox.setRange(1, 50)
        self.aisle_spinbox.setValue(self.num_aisles)
        self.aisle_spinbox.valueChanged.connect(self.update_num_aisles)

        self.shelf_spinbox = QSpinBox(self)
        self.shelf_spinbox.setRange(1, 50)
        self.shelf_spinbox.setValue(self.max_shelves_per_aisle)
        self.shelf_spinbox.valueChanged.connect(self.update_num_shelves)

        self.benchmark_spinbox = QSpinBox(self)
        self.benchmark_spinbox.setRange(1, 1000)
        self.benchmark_spinbox.setValue(1)

        # Buttons
        self.start_button = QPushButton("Set Start", self)
        self.start_button.clicked.connect(self.set_mode_start)

        self.end_button = QPushButton("Set End", self)
        self.end_button.clicked.connect(self.set_mode_end)

        self.barrier_button = QPushButton("Set Barriers", self)
        self.barrier_button.clicked.connect(self.set_mode_barrier)

        self.search_button = QPushButton("Search Path", self)
        self.search_button.clicked.connect(self.handle_search)

        self.generate_button = QPushButton("Generate Warehouse Layout", self)
        self.generate_button.clicked.connect(self.handle_generate_layout)

        self.clear_button = QPushButton("Clear All", self)
        self.clear_button.clicked.connect(self.clear_all)

        self.save_button = QPushButton("Save Scenario", self)
        self.save_button.clicked.connect(self.save_scenario)

        self.zoom_in_button = QPushButton("Zoom In", self)
        self.zoom_in_button.clicked.connect(self.zoom_in)

        self.zoom_out_button = QPushButton("Zoom Out", self)
        self.zoom_out_button.clicked.connect(self.zoom_out)

        self.show_all_paths_button = QPushButton("Show All Paths", self)
        self.show_all_paths_button.clicked.connect(self.handle_show_all_paths)

        # Remove the Benchmark Algorithms button
        # self.benchmark_button = QPushButton("Benchmark Algorithms", self)
        # self.benchmark_button.clicked.connect(self.benchmark_algorithms)

        self.random_benchmark_button = QPushButton("Run Random Benchmarks", self)
        self.random_benchmark_button.clicked.connect(self.run_random_benchmarks)

        # Checkbox
        self.diagonal_checkbox = QCheckBox("Allow Diagonal Neighbors", self)
        self.diagonal_checkbox.stateChanged.connect(self.handle_diagonal_change)

        # Add a new checkbox for using start node or random node in benchmarks
        self.use_start_as_benchmark_start_checkbox = QCheckBox("Use Start Node for Benchmarks", self)
        self.use_start_as_benchmark_start_checkbox.setChecked(True)  # Default: Use start node

        # *** Add a new checkbox for benchmarking mode ***
        self.all_nodes_checkbox = QCheckBox("Benchmark Against All Nodes", self)
        self.all_nodes_checkbox.setChecked(False)  # Default: Benchmark only item nodes

        # Labels
        self.counter_label = QLabel("Nodes Searched: 0", self)
        self.benchmark_label = QLabel("Number of Benchmark Runs:", self)

        # Layout setup
        layout = QVBoxLayout()
        layout.addWidget(self.view)

        # Spacing controls
        layout.addWidget(QLabel("Select Aisle Spacing:", self))
        layout.addWidget(self.spacing_dropdown)

        # Warehouse layout controls
        layout.addWidget(QLabel("Select Warehouse Layout:", self))
        layout.addWidget(self.layout_dropdown)

        # Aisle and shelf controls
        layout.addWidget(QLabel("Number of Aisles:", self))
        layout.addWidget(self.aisle_spinbox)
        layout.addWidget(QLabel("Max Shelves per Aisle:", self))
        layout.addWidget(self.shelf_spinbox)

        # Algorithm selection
        layout.addWidget(QLabel("Select Algorithm:", self))
        layout.addWidget(self.algorithm_dropdown)
        layout.addWidget(self.diagonal_checkbox)
        self.algorithm_dropdown.currentIndexChanged.connect(self.on_algorithm_changed)
        # Node setting buttons
        layout.addWidget(self.start_button)
        layout.addWidget(self.end_button)
        layout.addWidget(self.barrier_button)
        layout.addWidget(self.search_button)
        layout.addWidget(self.generate_button)

        # Save/Load controls
        layout.addWidget(self.save_button)
        layout.addWidget(QLabel("Load Scenario:", self))
        layout.addWidget(self.load_dropdown)
        layout.addWidget(self.clear_button)
        layout.addWidget(self.counter_label)

        # Item selection
        layout.addWidget(QLabel("Select Item as End Node:", self))
        layout.addWidget(self.item_dropdown)

        # Zoom controls
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(self.zoom_in_button)
        zoom_layout.addWidget(self.zoom_out_button)
        layout.addLayout(zoom_layout)

        # Show all paths button
        layout.addWidget(self.show_all_paths_button)

        # Benchmark controls in a group
        benchmark_group = QVBoxLayout()
        benchmark_header = QHBoxLayout()
        benchmark_header.addWidget(self.benchmark_label)
        benchmark_header.addWidget(self.benchmark_spinbox)
        benchmark_group.addLayout(benchmark_header)

        benchmark_buttons = QHBoxLayout()
        # Remove the Benchmark Algorithms button from the layout
        # benchmark_buttons.addWidget(self.benchmark_button)
        benchmark_buttons.addWidget(self.random_benchmark_button)
        benchmark_group.addLayout(benchmark_buttons)

        # Add the new checkbox for using start node in benchmarks
        benchmark_group.addWidget(self.use_start_as_benchmark_start_checkbox)

        # Add the existing checkbox for benchmarking mode
        benchmark_group.addWidget(self.all_nodes_checkbox)

        layout.addLayout(benchmark_group)

        # Set up the main container
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.grid_size = 12  # Initial grid size
        self.init_grid()

        self.load_scenarios()  # Load existing scenarios into the dropdown

        self.show()

        self.algorithm_capabilities = {
            "A* (Manhattan Distance)": {"handles_negative": False},
            "A* (Euclidean Distance)": {"handles_negative": False},
            "A* (Modified Euclidean 1.2x Y Priority)": {"handles_negative": False},
            "Dijkstra's": {"handles_negative": False},
            "Bellman-Ford": {"handles_negative": True},
            "SPFA": {"handles_negative": True},
            "Johnson's": {"handles_negative": True},
            "Johnson's with A*": {"handles_negative": True}
        }

    def handle_diagonal_change(self, state):
        """Handle changes to the diagonal movement checkbox."""
        new_diagonal_state = bool(state)
        if new_diagonal_state != self.current_diagonal_state:
            # Clear the Johnson's graph cache when diagonal movement changes
            if hasattr(self, 'johnsons_graph'):
                delattr(self, 'johnsons_graph')
            if hasattr(self, 'last_grid_state'):
                delattr(self, 'last_grid_state')
            print("Cleared Johnson's graph cache due to diagonal movement change")
            self.current_diagonal_state = new_diagonal_state

            # Reset the grid to clear any existing paths
            self.reset_grid()
    def on_algorithm_changed(self, index):
        """Handle algorithm selection change."""
        # Reset grid while preserving start, end, and barriers
        self.reset_grid()

        # Clear any cached Johnson's graph if it exists
        if hasattr(self, 'johnsons_graph'):
            delattr(self, 'johnsons_graph')
            print("Cleared cached Johnson's graph due to algorithm change.")

        # Check for negative weights compatibility
        has_negative = self.has_negative_weights()
        selected_algorithm = self.algorithm_dropdown.currentText()

        if has_negative and not self.algorithm_capabilities[selected_algorithm]["handles_negative"]:
            compatible_algorithms = [name for name, caps in self.algorithm_capabilities.items()
                                     if caps["handles_negative"]]
            QMessageBox.warning(
                self,
                "Invalid Algorithm Selection",
                f"The selected algorithm '{selected_algorithm}' cannot handle negative edge weights.\n"
                f"Please choose one of these algorithms:\n{', '.join(compatible_algorithms)}"
            )
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
        self.counter_label.setText("Nodes Searched: 0")

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

        # *** Clear Cached Johnson's Graph ***
        if hasattr(self, 'johnsons_graph'):
            del self.johnsons_graph
            print("Cleared cached Johnson's graph during clear_all.")
        if hasattr(self, 'last_grid_state'):
            del self.last_grid_state
            print("Cleared last grid state during clear_all.")

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
        try:
            self.is_generating_warehouse = True
            self.clear_all()

            # *** Clear Cached Johnson's Graph ***
            if hasattr(self, 'johnsons_graph'):
                del self.johnsons_graph
                print("Cleared cached Johnson's graph.")
            if hasattr(self, 'last_grid_state'):
                del self.last_grid_state
                print("Cleared last grid state.")

            # Generate warehouse data
            warehouse_data = generate_warehouse_data(
                num_aisles=self.num_aisles,
                max_shelves_per_aisle=self.max_shelves_per_aisle,
                save_to_csv=True
            )

            # Clear the item dropdown and populate it with new items
            self.item_dropdown.blockSignals(True)  # Block signals while populating the dropdown
            self.item_dropdown.clear()  # Clear old items
            self.item_dropdown.addItem("Select Item")  # Add default placeholder

            self.item_nodes = []  # Initialize item_nodes list

            aisles = warehouse_data['Aisle_Number'].nunique()
            aisle_positions = []

            vertical_positions = set()
            horizontal_positions = set()

            # Determine max_x and max_y to prevent out-of-bounds access
            max_x, max_y = 12, 12  # Default grid size

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

            # Adjust max_x and max_y to ensure they are within reasonable limits
            max_x = min(max_x, 50)  # Set an upper limit for the grid size
            max_y = min(max_y, 50)

            # Update grid size
            self.grid_size = max(int(max_x), int(max_y)) + 2  # Add buffer
            self.init_grid()  # Reinitialize the grid with the new size

            for i, row in warehouse_data.iterrows():
                try:
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
                        print(f"Skipping node at ({x}, {y}) - out of bounds.")
                        continue  # Skip if out of bounds

                    node = self.grid[y][x]
                    node.set_as_aisle(aisle_color)  # Pass the correct aisle color
                    # Set the node's name to include aisle, shelf, and location
                    node.name = f"Aisle_{aisle_num}_Shelf_{shelf_num}_{shelf_loc}"

                    # Add the item to the dropdown if not empty
                    if item != "Empty":
                        self.item_dropdown.addItem(
                            f"{item} (Aisle {aisle_num}, Shelf {shelf_num}, Location {shelf_loc})")

                        # Store the item node for benchmarking
                        node_info = {
                            'node': node,
                            'item': item,
                            'x': x,
                            'y': y
                        }
                        self.item_nodes.append(node_info)

                        # Define the item number
                        item_number = item.split('_')[-1] if item != "Empty" else "Empty"

                        # Set the item label using the Node class method
                        node.set_item_label(item_number)
                except Exception as e:
                    print(f"Error processing row {i}: {e}")

            # Re-enable signals now that dropdown is populated
            self.item_dropdown.blockSignals(False)

            # Adjust the zoom to fit the new layout
            self.adjust_zoom()
            print(f"Item nodes available: {len(self.item_nodes)}")

        except Exception as e:
            print(f"Error in generate_warehouse_layout: {e}")
        finally:
            # Reset the flag after generation is complete
            self.is_generating_warehouse = False

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
        # Prevent setting the same node as both start and end
        if not hasattr(self, 'start_node'):
            self.start_node = None
        if self.end_node == node:
            return

        # Reset the previous start node if it exists
        if self.start_node:
            self.start_node.is_start = False
            self.start_node.reset()

        # Set the new start node
        self.start_node = node
        self.start_node.set_as_start()

    def set_end_node(self, node):
        """Set the selected node as the end node."""
        # Prevent setting the same node as both start and end
        if not hasattr(self, 'end_node'):
            self.end_node = None
        if self.start_node == node:
            return

        # Reset the previous end node if it exists
        if self.end_node:
            self.end_node.is_end = False
            self.end_node.reset()

        # Set the new end node
        self.end_node = node
        self.end_node.set_as_end()

    def set_as_start(self):
        """Mark the node as the start (green)."""
        if self.parent_window.end_node == self:
            return  # Prevent the same node from being both start and end
        if self.parent_window.start_node:
            self.parent_window.start_node.is_start = False
            self.parent_window.start_node.reset()
        self.is_start = True
        self.is_obstacle = False
        self.setBrush(QBrush(QColor(0, 255, 0)))  # Green for start
        self.parent_window.start_node = self

    def set_as_end(self):
        """Mark the node as the end (red)."""
        if self.parent_window.start_node == self:
            return  # Prevent the same node from being both start and end
        if self.parent_window.end_node:
            self.parent_window.end_node.is_end = False
            self.parent_window.end_node.reset()
        self.is_end = True
        self.is_obstacle = False
        self.setBrush(QBrush(QColor(255, 0, 0)))  # Red for end
        self.parent_window.end_node = self

    def set_end_node_from_dropdown(self):
        """Set the end node based on the item selected from the dropdown."""
        selected_item = self.item_dropdown.currentText()
        if selected_item == "Select Item":
            return  # Do nothing if placeholder is selected

        # Extract item details from the dropdown text
        try:
            # Parsing text like: "Item_1234 (Aisle 1, Shelf 2, Location A)"
            item_part, location_part = selected_item.split(' (')
            item_name = item_part.strip()
            location_part = location_part.rstrip(')')
            # location_part is now like "Aisle 1, Shelf 2, Location A"
            location_items = location_part.split(', ')
            aisle_num = int(location_items[0].split(' ')[1])
            shelf_num = int(location_items[1].split(' ')[1])
            shelf_loc = location_items[2].split(' ')[1]

            # Determine orientation
            orientation = self.layout_dropdown.currentText()
            if orientation == "Vertical Aisles":
                orientation_type = 'vertical'
            elif orientation == "Horizontal Aisles":
                orientation_type = 'horizontal'
            elif orientation == "Mixed Aisles":
                orientation_type = 'mixed'
            else:
                orientation_type = 'vertical'  # Default to vertical

            # Reconstruct aisle positions as in generate_warehouse_layout
            aisle_positions = []
            vertical_positions = set()
            horizontal_positions = set()

            if orientation_type == 'vertical':
                for aisle_idx in range(1, self.num_aisles + 1):
                    x_pos = 2 + (aisle_idx - 1) * (self.spacing + 1)
                    vertical_positions.add(x_pos)
                    aisle_positions.append(('vertical', x_pos))
            elif orientation_type == 'horizontal':
                for aisle_idx in range(1, self.num_aisles + 1):
                    y_pos = 2 + (aisle_idx - 1) * (self.spacing + 1)
                    horizontal_positions.add(y_pos)
                    aisle_positions.append(('horizontal', y_pos))
            elif orientation_type == 'mixed':
                num_vertical_aisles = (self.num_aisles + 1) // 2
                num_horizontal_aisles = self.num_aisles // 2

                for i in range(num_vertical_aisles):
                    x_pos = 2 + i * (self.spacing + 2)
                    vertical_positions.add(x_pos)
                    aisle_positions.append(('vertical', x_pos))

                for i in range(num_horizontal_aisles):
                    y_pos = 2 + i * (self.spacing + 2)
                    while y_pos in vertical_positions or y_pos + 1 in vertical_positions:
                        y_pos += 1
                    horizontal_positions.add(y_pos)
                    aisle_positions.append(('horizontal', y_pos))

            # Now, get the aisle position for the given aisle_num
            if aisle_num - 1 >= len(aisle_positions):
                print(f"Aisle number {aisle_num} exceeds available positions.")
                return

            orientation_type_aisle, pos = aisle_positions[aisle_num - 1]

            if orientation_type_aisle == 'vertical':
                x = pos
                y = 2 + (shelf_num - 1)
            else:
                x = 2 + (shelf_num - 1)
                y = pos

            # Ensure x and y are within grid bounds
            if x >= self.grid_size or y >= self.grid_size:
                print("Calculated position is out of bounds.")
                return

            node = self.grid[y][x]

            # Now, set the end node
            self.set_end_node(node)

        except (IndexError, ValueError) as e:
            print(f"Error parsing selected item: {e}")

    def has_negative_weights(self):
        """Check if any edge weights in the grid are negative."""
        for row in self.grid:
            for node in row:
                if node.edge_weight < 0:
                    return True
        return False

    def handle_search(self):
        """Handle search between start and end nodes."""
        # Check if start and end nodes are set
        has_negative = self.has_negative_weights()
        selected_algorithm = self.algorithm_dropdown.currentText()

        current_weights_state = {(x, y): self.grid[y][x].edge_weight
                                 for x in range(self.grid_size)
                                 for y in range(self.grid_size)}

        if hasattr(self, 'last_weights_state') and self.last_weights_state != current_weights_state:
            if hasattr(self, 'johnsons_graph'):
                delattr(self, 'johnsons_graph')
                print("Cleared cached Johnson's graph due to edge weight changes.")

        self.last_weights_state = current_weights_state

        if has_negative and not self.algorithm_capabilities[selected_algorithm]["handles_negative"]:
            compatible_algorithms = [name for name, caps in self.algorithm_capabilities.items()
                                     if caps["handles_negative"]]
            QMessageBox.warning(
                self,
                "Invalid Algorithm Selection",
                f"The selected algorithm '{selected_algorithm}' cannot handle negative edge weights.\n"
                f"Please choose one of these algorithms:\n{', '.join(compatible_algorithms)}"
            )
            return

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

        # Check if diagonal neighbors are allowed
        diagonal_neighbors = self.diagonal_checkbox.isChecked()

        # Determine which algorithm to run
        selected_algorithm = self.algorithm_dropdown.currentText()

        if selected_algorithm == "Dijkstra's":
            path, self.nodes_searched = self.run_dijkstra(start, end, diagonal_neighbors, visualize=True)
        elif selected_algorithm == "Bellman-Ford":
            path, self.nodes_searched = self.run_bellman_ford(start, end, diagonal_neighbors, visualize=True)
        elif selected_algorithm == "A* (Manhattan Distance)":
            path, self.nodes_searched = self.run_astar(start, end, diagonal_neighbors, visualize=True)
        elif selected_algorithm == "A* (Euclidean Distance)":
            path, self.nodes_searched = self.run_astar(start, end, diagonal_neighbors, visualize=True)
        elif selected_algorithm == "A* (Modified Euclidean 1.2x Y Priority)":
            path, self.nodes_searched = self.run_astar(start, end, diagonal_neighbors, visualize=True)
        elif selected_algorithm == "SPFA":
            path, self.nodes_searched = self.run_spfa(start, end, diagonal_neighbors, visualize=True)
        elif selected_algorithm == "Johnson's":
            path, (mandatory_visits, pathfinding_visits) = self.run_johnsons(start, end, diagonal_neighbors,
                                                                             visualize=True)
        elif selected_algorithm == "Johnson's with A*":
            path, (mandatory_visits, pathfinding_visits) = self.run_johnsons_astar(
                start, end, diagonal_neighbors, visualize=True
            )
            # For visualization purposes, use the sum of both types of visits
            self.nodes_searched = mandatory_visits + pathfinding_visits
            # You might want to show both counts in the counter label
            self.counter_label.setText(f"Mandatory: {mandatory_visits}, Pathfinding: {pathfinding_visits}")

            # For visualization purposes, use the sum of both types of visits
            self.nodes_searched = mandatory_visits + pathfinding_visits
            # You might want to show both counts in the counter label
            self.counter_label.setText(f"Mandatory: {mandatory_visits}, Pathfinding: {pathfinding_visits}")
        else:
            self.counter_label.setText("Invalid algorithm selection.")
            return

        # If a path is found, visualize it step by step
        if path:
            self.search_path = path
            self.visualize_path_step_by_step()
        else:
            self.counter_label.setText("No path found.")

    def get_neighbors_for_reweighting(self, node, diagonal_neighbors=False):
        """
        Helper function to get neighbors for Johnson's algorithms with proper diagonal movement costs.
        Includes checking for barrier jumping with diagonal moves.
        """
        (x, y) = node
        # Define orthogonal and diagonal neighbors with their respective costs
        four_neighbors = [
            ((-1, 0), 1.0),  # Left
            ((1, 0), 1.0),  # Right
            ((0, -1), 1.0),  # Up
            ((0, 1), 1.0)  # Down
        ]

        eight_neighbors = [
            ((-1, -1), 1.414),  # Up-Left (√2)
            ((1, -1), 1.414),  # Up-Right
            ((-1, 1), 1.414),  # Down-Left
            ((1, 1), 1.414)  # Down-Right
        ]

        def is_diagonal_valid(curr_x, curr_y, new_x, new_y):
            """
            Check if a diagonal move is valid by ensuring both orthogonal paths aren't blocked.
            For example, to move diagonally up-right, both the 'up' and 'right' spaces must be clear.
            """
            dx = new_x - curr_x
            dy = new_y - curr_y

            # Check first orthogonal path (horizontal)
            if not (0 <= curr_x + dx < self.grid_size):
                return False
            if self.grid[curr_y][curr_x + dx].is_obstacle:
                return False

            # Check second orthogonal path (vertical)
            if not (0 <= curr_y + dy < self.grid_size):
                return False
            if self.grid[curr_y + dy][curr_x].is_obstacle:
                return False

            return True

        neighbors = []

        # First add orthogonal neighbors (no need for extra checks)
        for (dx, dy), base_cost in four_neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                neighbor_node = self.grid[ny][nx]
                if not neighbor_node.is_obstacle:
                    neighbor_coords = (nx, ny)
                    # Multiply the base movement cost by the node's edge weight
                    total_cost = base_cost * neighbor_node.edge_weight
                    neighbors.append((neighbor_coords, total_cost))

        # Then add diagonal neighbors if enabled and valid
        if diagonal_neighbors:
            for (dx, dy), base_cost in eight_neighbors:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    neighbor_node = self.grid[ny][nx]
                    if not neighbor_node.is_obstacle and is_diagonal_valid(x, y, nx, ny):
                        neighbor_coords = (nx, ny)
                        # Multiply the base movement cost by the node's edge weight
                        total_cost = base_cost * neighbor_node.edge_weight
                        neighbors.append((neighbor_coords, total_cost))

        return neighbors

    def run_johnsons(self, start, end, diagonal_neighbors=False, visualize=True):
        """
        Johnson's algorithm implementation with improved negative weight handling.
        """
        mandatory_visits = 0
        pathfinding_visits = 0

        # Include ALL valid nodes including aisles, just excluding actual obstacles
        current_nodes = [(x, y) for y in range(self.grid_size) for x in range(self.grid_size)
                         if not self.grid[y][x].is_obstacle]

        # Validate start and end are not obstacles
        if self.grid[start[1]][start[0]].is_obstacle or self.grid[end[1]][end[0]].is_obstacle:
            if visualize:
                self.counter_label.setStyleSheet("color: red;")
                self.counter_label.setText("Invalid start or end position")
            return None, (mandatory_visits, pathfinding_visits)

        # PHASE 1: Graph Preprocessing (only if needed)
        if not hasattr(self, 'johnsons_graph') or not hasattr(self, 'last_grid_state'):
            try:
                if visualize:
                    self.counter_label.setStyleSheet("color: green;")
                    self.counter_label.setText("Creating reweighted graph using Bellman-Ford")
                    QApplication.processEvents()

                # Save current grid state
                self.last_grid_state = {(x, y): self.grid[y][x].is_obstacle
                                        for x in range(self.grid_size)
                                        for y in range(self.grid_size)}

                # Step 1: Create graph with virtual node
                virtual_node = (-1, -1)
                modified_graph = {virtual_node: []}

                # Initialize all nodes (including aisles)
                for node in current_nodes:
                    modified_graph[node] = []
                    mandatory_visits += 1
                    if visualize and node != start and node != end:  # Preserve start node color
                        self.grid[node[1]][node[0]].set_visited()
                        self.counter_label.setText(f"Building graph: {mandatory_visits} nodes processed")
                        QApplication.processEvents()

                # Add edges, considering aisle nodes
                for node in current_nodes:
                    modified_graph[virtual_node].append((node, 0))
                    for neighbor, weight in self.get_neighbors_for_reweighting(node, diagonal_neighbors):
                        if neighbor in current_nodes:
                            modified_graph[node].append((neighbor, weight))

                # Step 2: Run modified Bellman-Ford for reweighting
                h_values = {node: float('inf') for node in current_nodes}
                h_values[virtual_node] = 0

                # First pass: |V|-1 iterations of relaxation
                for i in range(len(current_nodes)):
                    updates = False
                    for u in modified_graph:
                        for v, weight in modified_graph[u]:
                            if h_values[u] != float('inf') and h_values[u] + weight < h_values[v]:
                                h_values[v] = h_values[u] + weight
                                updates = True
                                if v != virtual_node:
                                    mandatory_visits += 1
                                    if visualize and v != start and v != end:  # Preserve start node color
                                        self.grid[v[1]][v[0]].set_relaxed()
                                        self.counter_label.setText(f"Running Bellman-Ford: {mandatory_visits} updates")
                                        QApplication.processEvents()
                    if not updates:
                        break

                # Second pass: Check for negative cycles
                for u in modified_graph:
                    for v, weight in modified_graph[u]:
                        if h_values[u] != float('inf') and h_values[u] + weight < h_values[v]:
                            if visualize:
                                self.counter_label.setStyleSheet("color: red;")
                                self.counter_label.setText("Negative cycle detected - cannot proceed")
                                QApplication.processEvents()
                            return None, (mandatory_visits, pathfinding_visits)

                # Step 3: Create reweighted graph with validation
                self.johnsons_graph = {node: [] for node in current_nodes}
                for u in current_nodes:
                    for v, weight in modified_graph[u]:
                        if v != virtual_node:
                            new_weight = weight + h_values[u] - h_values[v]
                            epsilon = 1e-10
                            if new_weight < -epsilon:
                                if visualize:
                                    self.counter_label.setStyleSheet("color: red;")
                                    self.counter_label.setText("Reweighting failed - invalid negative weight")
                                    QApplication.processEvents()
                                return None, (mandatory_visits, pathfinding_visits)
                            new_weight = max(0, new_weight)
                            self.johnsons_graph[u].append((v, new_weight))

            except Exception as e:
                if visualize:
                    self.counter_label.setStyleSheet("color: red;")
                    self.counter_label.setText(f"Error in Johnson's preprocessing: {str(e)}")
                    QApplication.processEvents()
                return None, (mandatory_visits, pathfinding_visits)

        else:
            current_grid_state = {(x, y): self.grid[y][x].is_obstacle
                                  for x in range(self.grid_size)
                                  for y in range(self.grid_size)}

            if current_grid_state != self.last_grid_state:
                delattr(self, 'johnsons_graph')
                return self.run_johnsons(start, end, diagonal_neighbors, visualize)

        # Reset visualization before pathfinding
        if visualize:
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    if (x, y) != start and (x, y) != end:
                        self.grid[y][x].reset()

        # PHASE 2: Pathfinding using Dijkstra's
        if start not in self.johnsons_graph or end not in self.johnsons_graph:
            if visualize:
                self.counter_label.setStyleSheet("color: red;")
                self.counter_label.setText("Start or end node not in cached graph")
            return None, (mandatory_visits, pathfinding_visits)

        distances = {node: float('inf') for node in self.johnsons_graph}
        distances[start] = 0
        predecessors = {node: None for node in self.johnsons_graph}
        pq = [(0, start)]
        visited = set()

        while pq:
            current_dist, current = heapq.heappop(pq)

            if current in visited:
                continue

            visited.add(current)
            pathfinding_visits += 1

            if current == end:
                break

            if visualize and current != start and current != end:
                self.grid[current[1]][current[0]].set_visited()
                self.counter_label.setText(f"Pathfinding visits: {pathfinding_visits}")
                QApplication.processEvents()

            for neighbor, weight in self.johnsons_graph[current]:
                if neighbor not in visited:
                    can_traverse = (
                            not self.grid[neighbor[1]][neighbor[0]].is_aisle  # Regular non-aisle node
                            or neighbor == end  # End node (can be aisle)
                            or current == start  # Moving from start (can go to aisle)
                    )

                    if can_traverse:
                        distance = distances[current] + weight
                        if distance < distances[neighbor]:
                            distances[neighbor] = distance
                            predecessors[neighbor] = current
                            heapq.heappush(pq, (distance, neighbor))

        # Reconstruct path
        if distances[end] != float('inf'):
            path = []
            current = end
            while current is not None:
                path.append(current)
                current = predecessors[current]
            path.reverse()

            if visualize:
                for node in path:
                    if node != start and node != end:
                        self.grid[node[1]][node[0]].set_path()
                        QApplication.processEvents()

            return path, (mandatory_visits, pathfinding_visits)

        if visualize:
            self.counter_label.setStyleSheet("color: red;")
            self.counter_label.setText("No path found")
        return None, (mandatory_visits, pathfinding_visits)

    def run_johnsons_astar(self, start, end, diagonal_neighbors=False, visualize=True):
        """
        Johnson's algorithm implementation with A* pathfinding after reweighting.
        """
        mandatory_visits = 0
        pathfinding_visits = 0

        # Include ALL valid nodes including aisles, just excluding actual obstacles
        current_nodes = [(x, y) for y in range(self.grid_size) for x in range(self.grid_size)
                         if not self.grid[y][x].is_obstacle]

        # Validate start and end are not obstacles
        if self.grid[start[1]][start[0]].is_obstacle or self.grid[end[1]][end[0]].is_obstacle:
            if visualize:
                self.counter_label.setStyleSheet("color: red;")
                self.counter_label.setText("Invalid start or end position")
            return None, (mandatory_visits, pathfinding_visits)

        # PHASE 1: Graph Preprocessing (only if needed)
        if not hasattr(self, 'johnsons_graph') or not hasattr(self, 'last_grid_state'):
            try:
                if visualize:
                    self.counter_label.setStyleSheet("color: green;")
                    self.counter_label.setText("Creating reweighted graph using Bellman-Ford")
                    QApplication.processEvents()

                # Save current grid state
                self.last_grid_state = {(x, y): self.grid[y][x].is_obstacle
                                        for x in range(self.grid_size)
                                        for y in range(self.grid_size)}

                # Step 1: Create graph with virtual node
                virtual_node = (-1, -1)
                modified_graph = {virtual_node: []}

                # Initialize all nodes (including aisles)
                for node in current_nodes:
                    modified_graph[node] = []
                    mandatory_visits += 1
                    if visualize and node != start and node != end:
                        self.grid[node[1]][node[0]].set_visited()
                        self.counter_label.setText(f"Building graph: {mandatory_visits} nodes processed")
                        QApplication.processEvents()

                # Add edges, considering aisle nodes
                for node in current_nodes:
                    modified_graph[virtual_node].append((node, 0))
                    for neighbor, weight in self.get_neighbors_for_reweighting(node, diagonal_neighbors):
                        if neighbor in current_nodes:
                            modified_graph[node].append((neighbor, weight))

                # Step 2: Run modified Bellman-Ford for reweighting
                h_values = {node: float('inf') for node in current_nodes}
                h_values[virtual_node] = 0

                # First pass: |V|-1 iterations of relaxation
                for i in range(len(current_nodes)):
                    updates = False
                    for u in modified_graph:
                        for v, weight in modified_graph[u]:
                            if h_values[u] != float('inf') and h_values[u] + weight < h_values[v]:
                                h_values[v] = h_values[u] + weight
                                updates = True
                                if v != virtual_node:
                                    mandatory_visits += 1
                                    if visualize and v != start and v != end:
                                        self.grid[v[1]][v[0]].set_relaxed()
                                        self.counter_label.setText(f"Running Bellman-Ford: {mandatory_visits} updates")
                                        QApplication.processEvents()
                    if not updates:
                        break

                # Second pass: Check for negative cycles
                for u in modified_graph:
                    for v, weight in modified_graph[u]:
                        if h_values[u] != float('inf') and h_values[u] + weight < h_values[v]:
                            if visualize:
                                self.counter_label.setStyleSheet("color: red;")
                                self.counter_label.setText("Negative cycle detected - cannot proceed")
                                QApplication.processEvents()
                            return None, (mandatory_visits, pathfinding_visits)

                # Step 3: Create reweighted graph with validation
                self.johnsons_graph = {node: [] for node in current_nodes}
                for u in current_nodes:
                    for v, weight in modified_graph[u]:
                        if v != virtual_node:
                            new_weight = weight + h_values[u] - h_values[v]
                            epsilon = 1e-10
                            if new_weight < -epsilon:
                                if visualize:
                                    self.counter_label.setStyleSheet("color: red;")
                                    self.counter_label.setText("Reweighting failed - invalid negative weight")
                                    QApplication.processEvents()
                                return None, (mandatory_visits, pathfinding_visits)
                            new_weight = max(0, new_weight)
                            self.johnsons_graph[u].append((v, new_weight))

            except Exception as e:
                if visualize:
                    self.counter_label.setStyleSheet("color: red;")
                    self.counter_label.setText(f"Error in Johnson's preprocessing: {str(e)}")
                    QApplication.processEvents()
                return None, (mandatory_visits, pathfinding_visits)

        else:
            current_grid_state = {(x, y): self.grid[y][x].is_obstacle
                                  for x in range(self.grid_size)
                                  for y in range(self.grid_size)}

            if current_grid_state != self.last_grid_state:
                delattr(self, 'johnsons_graph')
                return self.run_johnsons_astar(start, end, diagonal_neighbors, visualize)

        # Reset visualization before pathfinding
        if visualize:
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    if (x, y) != start and (x, y) != end:
                        self.grid[y][x].reset()

        # PHASE 2: A* Pathfinding
        if start not in self.johnsons_graph or end not in self.johnsons_graph:
            if visualize:
                self.counter_label.setStyleSheet("color: red;")
                self.counter_label.setText("Start or end node not in cached graph")
            return None, (mandatory_visits, pathfinding_visits)

        g_score = {node: float('inf') for node in self.johnsons_graph}
        g_score[start] = 0
        f_score = {node: float('inf') for node in self.johnsons_graph}
        f_score[start] = self.heuristic(start, end, "Manhattan Distance")

        pq = [(f_score[start], start)]
        predecessors = {node: None for node in self.johnsons_graph}
        visited = set()

        while pq:
            current_f, current = heapq.heappop(pq)

            if current in visited:
                continue

            visited.add(current)
            pathfinding_visits += 1

            if current == end:
                break

            if visualize and current != start and current != end:
                self.grid[current[1]][current[0]].set_visited()
                self.counter_label.setText(f"Pathfinding visits: {pathfinding_visits}")
                QApplication.processEvents()

            for neighbor, weight in self.johnsons_graph[current]:
                if neighbor not in visited:
                    can_traverse = (
                            not self.grid[neighbor[1]][neighbor[0]].is_aisle  # Regular non-aisle node
                            or neighbor == end  # End node (can be aisle)
                            or current == start  # Moving from start (can go to aisle)
                    )

                    if can_traverse:
                        tentative_g_score = g_score[current] + weight
                        if tentative_g_score < g_score[neighbor]:
                            predecessors[neighbor] = current
                            g_score[neighbor] = tentative_g_score
                            f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, end, "Manhattan Distance")
                            heapq.heappush(pq, (f_score[neighbor], neighbor))

        # Reconstruct path
        if g_score[end] != float('inf'):
            path = []
            current = end
            while current is not None:
                path.append(current)
                current = predecessors[current]
            path.reverse()

            if visualize:
                for node in path:
                    if node != start and node != end:
                        self.grid[node[1]][node[0]].set_path()
                        QApplication.processEvents()

            return path, (mandatory_visits, pathfinding_visits)

        if visualize:
            self.counter_label.setStyleSheet("color: red;")
            self.counter_label.setText("No path found")
        return None, (mandatory_visits, pathfinding_visits)


    def bellman_ford_for_johnsons(self, graph, source):
        """Helper function for Johnson's algorithm to compute h values."""
        distances = {node: float('inf') for node in graph}
        distances[source] = 0

        # Relax all edges |V|-1 times
        for _ in range(len(graph) - 1):
            for u in graph:
                for v, weight in graph[u]:
                    if distances[u] + weight < distances[v]:
                        distances[v] = distances[u] + weight

        # Check for negative cycles
        for u in graph:
            for v, weight in graph[u]:
                if distances[u] + weight < distances[v]:
                    return None  # Negative cycle detected

        return distances

    def dijkstra_for_johnsons(self, graph, source):
        """Modified Dijkstra's algorithm for Johnson's algorithm."""
        distances = {node: float('inf') for node in graph}
        distances[source] = 0
        predecessors = {node: None for node in graph}
        open_set = [(0, source)]

        while open_set:
            current_dist, current = heapq.heappop(open_set)

            if current_dist > distances[current]:
                continue

            self.nodes_searched += 1

            for neighbor, weight in graph[current]:
                distance = distances[current] + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current
                    heapq.heappush(open_set, (distance, neighbor))

        return distances, predecessors

    def run_astar(self, start, end, diagonal_neighbors=False, visualize=True, heuristic_type="Manhattan"):
        """Run the A* algorithm from start to end with the specified heuristic."""
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}

        if heuristic_type == "Manhattan":
            h = self.heuristic(start, end, heuristic_type="Manhattan Distance")
        elif heuristic_type == "Euclidean":
            h = self.heuristic(start, end, heuristic_type="Euclidean Distance")
        elif heuristic_type == "Modified Euclidean":
            h = self.heuristic(start, end, heuristic_type="Modified Euclidean")
        else:
            h = self.heuristic(start, end, heuristic_type="Manhattan Distance")  # Default

        f_score = {start: h}
        self.nodes_searched = 0  # Reset node search count

        while open_set:
            current_f, current = heapq.heappop(open_set)

            if current == end:
                path = self.reconstruct_path(came_from, current)
                return path, self.nodes_searched

            if visualize:
                self.grid[current[1]][current[0]].set_visited()
                self.counter_label.setText(f"Nodes Searched: {self.nodes_searched}")
                QApplication.processEvents()  # Update the UI in real-time

            self.nodes_searched += 1

            for neighbor, weight in self.get_neighbors(current, diagonal_neighbors):
                tentative_g_score = g_score[current] + weight

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, end, heuristic_type)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # If open_set is empty and end was not reached
        return None, self.nodes_searched

    def run_spfa(self, start, end, diagonal_neighbors=False, visualize=True):
        """
        Run the Shortest Path Faster Algorithm (SPFA) from start to end.
        Handles negative weights while being typically faster than Bellman-Ford.
        """
        # Initialize distances and predecessors
        distance = {(x, y): float('inf') for x in range(self.grid_size) for y in range(self.grid_size)}
        predecessor = {(x, y): None for x in range(self.grid_size) for y in range(self.grid_size)}
        in_queue = {(x, y): False for x in range(self.grid_size) for y in range(self.grid_size)}

        distance[start] = 0
        queue = [start]
        in_queue[start] = True
        self.nodes_searched = 0

        while queue:
            current = queue.pop(0)
            in_queue[current] = False

            # Visualize current node if needed
            if visualize:
                self.grid[current[1]][current[0]].set_visited()
                self.counter_label.setText(f"Nodes Searched: {self.nodes_searched}")
                QApplication.processEvents()

            self.nodes_searched += 1

            # Process neighbors
            for neighbor_coords, weight in self.get_neighbors(current, diagonal_neighbors):
                if distance[current] + weight < distance[neighbor_coords]:
                    distance[neighbor_coords] = distance[current] + weight
                    predecessor[neighbor_coords] = current

                    # Add to queue if not already in it
                    if not in_queue[neighbor_coords]:
                        queue.append(neighbor_coords)
                        in_queue[neighbor_coords] = True

            # Early exit if we've found the end node
            if current == end:
                break

        # Reconstruct path if end was reached
        if distance[end] != float('inf'):
            path = []
            current = end
            while current is not None:
                path.append(current)
                current = predecessor[current]
            path.reverse()
            return path, self.nodes_searched

        return None, self.nodes_searched

    def run_bellman_ford(self, start, end, diagonal_neighbors=False, visualize=True):
        nodes = [(x, y) for y in range(self.grid_size) for x in range(self.grid_size)]
        edges = []

        # Initialize distances and predecessors
        distance = {node: float('inf') for node in nodes}
        distance[start] = 0
        predecessor = {node: None for node in nodes}

        # Build the list of edges with weights
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                current_node = (x, y)
                for neighbor_coords, weight in self.get_neighbors(current_node, diagonal_neighbors):
                    edges.append((current_node, neighbor_coords, weight))  # Use tuple coordinates

        self.nodes_searched = 0  # Reset the count of nodes searched
        visited_nodes = set()  # To track unique nodes visited

        # Bellman-Ford Algorithm: Relax edges |V|-1 times
        for _ in range(len(nodes) - 1):
            for u, v, weight in edges:
                if distance[u] + weight < distance[v]:
                    distance[v] = distance[u] + weight
                    predecessor[v] = u

                    if v not in visited_nodes:
                        visited_nodes.add(v)
                        self.nodes_searched += 1

                    # Mark the node as visited and update the UI if needed
                    if visualize:
                        self.grid[v[1]][v[0]].set_visited()
                        self.counter_label.setText(f"Nodes Searched: {self.nodes_searched}")
                        QApplication.processEvents()  # Update the UI in real-time

        # Check for negative weight cycles (not applicable in this case but part of Bellman-Ford)
        for u, v, weight in edges:
            if distance[u] + weight < distance[v]:
                print("Graph contains a negative-weight cycle.")
                return None, self.nodes_searched

        # Reconstruct the path from end node to start node
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = predecessor[current]

        path.reverse()  # Reverse the path to go from start to end

        # Visualize the path if needed
        if visualize and path:
            for node in path:
                # Avoid changing the color of start and end nodes
                if node != start and node != end:
                    self.grid[node[1]][node[0]].set_path()
                    QApplication.processEvents()  # Update the UI in real-time

        return path if path[0] == start else None, self.nodes_searched

    def run_dijkstra(self, start, end, diagonal_neighbors=False, visualize=True):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        self.nodes_searched = 0  # Reset node search count

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                # Return path and nodes_searched
                path = self.reconstruct_path(came_from, current)
                return path, self.nodes_searched

            # Mark the node as visited visually and count it
            self.nodes_searched += 1

            if visualize:
                self.grid[current[1]][current[0]].set_visited()
                self.counter_label.setText(f"Nodes Searched: {self.nodes_searched}")
                QApplication.processEvents()  # Update the UI in real-time

            # Get neighbors, considering diagonal movement if enabled
            for neighbor_coords, weight in self.get_neighbors(current, diagonal_neighbors):
                tentative_g_score = g_score[current] + weight  # Use edge weight here

                if neighbor_coords not in g_score or tentative_g_score < g_score[neighbor_coords]:
                    came_from[neighbor_coords] = current
                    g_score[neighbor_coords] = tentative_g_score
                    heapq.heappush(open_set, (g_score[neighbor_coords], neighbor_coords))

        # No path found
        return None, self.nodes_searched

    def is_valid_position(self, x, y):
        """Check if position is within grid bounds."""
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def is_obstacle_func(self, x, y):
        """Check if the node is an obstacle, and ensure aisles are non-traversable except the end node."""
        if not self.is_valid_position(x, y):
            return True
        # If it's an aisle and not the end node, treat it as an obstacle
        if self.grid[y][x].is_aisle and self.grid[y][x] != self.end_node:
            return True
        return self.grid[y][x].is_obstacle

    def is_traversable(self, x, y):
        """Check if the node is traversable (not an obstacle or it's the start/end node)."""
        if not self.is_valid_position(x, y):
            return False
        node = self.grid[y][x]

        # Allow aisles to be traversable only if they are the end node
        if node.is_aisle and node != self.end_node:
            return False

        return not node.is_obstacle

    def distance(self, node_a, node_b):
        """Calculate distance between two nodes."""
        (x1, y1) = node_a
        (x2, y2) = node_b
        return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def heuristic(self, node, goal, heuristic_type="Manhattan"):
        """Heuristic function that supports different types of heuristics."""
        (x1, y1) = node
        (x2, y2) = goal

        if heuristic_type == "Manhattan Distance":
            return abs(x1 - x2) + abs(y1 - y2)
        elif heuristic_type == "Euclidean Distance":
            return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        elif heuristic_type == "Modified Euclidean":
            return sqrt((x1 - x2) ** 2 + (1.2 * (y1 - y2)) ** 2)
        else:
            # Default fallback to Manhattan distance
            return abs(x1 - x2) + abs(y1 - y2)

    def get_neighbors(self, node, diagonal_neighbors=False):
        """Get valid neighboring nodes and their edge weights."""
        (x, y) = node

        # Define orthogonal and diagonal neighbors with their respective costs
        four_neighbors = [
            ((-1, 0), 1.0),  # Left
            ((1, 0), 1.0),  # Right
            ((0, -1), 1.0),  # Up
            ((0, 1), 1.0)  # Down
        ]

        eight_neighbors = [
            ((-1, -1), 1.414),  # Up-Left (√2)
            ((1, -1), 1.414),  # Up-Right
            ((-1, 1), 1.414),  # Down-Left
            ((1, 1), 1.414)  # Down-Right
        ]

        def is_diagonal_valid(curr_x, curr_y, new_x, new_y):
            """
            Check if a diagonal move is valid by ensuring both orthogonal paths aren't blocked.
            For example, to move diagonally up-right, both the 'up' and 'right' spaces must be clear.
            """
            dx = new_x - curr_x
            dy = new_y - curr_y

            # Check first orthogonal path (horizontal)
            if not (0 <= curr_x + dx < self.grid_size):
                return False
            if not self.is_traversable(curr_x + dx, curr_y):
                return False

            # Check second orthogonal path (vertical)
            if not (0 <= curr_y + dy < self.grid_size):
                return False
            if not self.is_traversable(curr_x, curr_y + dy):
                return False

            return True

        neighbors = []

        # First add orthogonal neighbors
        for (dx, dy), base_cost in four_neighbors:
            nx, ny = x + dx, y + dy
            if self.is_valid_position(nx, ny) and self.is_traversable(nx, ny):
                neighbor_coords = (nx, ny)
                neighbor_node = self.grid[ny][nx]
                total_cost = base_cost * neighbor_node.edge_weight
                neighbors.append((neighbor_coords, total_cost))

        # Then add diagonal neighbors if enabled and valid
        if diagonal_neighbors:
            for (dx, dy), base_cost in eight_neighbors:
                nx, ny = x + dx, y + dy
                if self.is_valid_position(nx, ny) and self.is_traversable(nx, ny):
                    if is_diagonal_valid(x, y, nx, ny):
                        neighbor_coords = (nx, ny)
                        neighbor_node = self.grid[ny][nx]
                        total_cost = base_cost * neighbor_node.edge_weight
                        neighbors.append((neighbor_coords, total_cost))

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

    def visualize_next_path(self):
        # First, check if all target nodes have been processed
        if self.current_target_index >= len(self.all_target_nodes_copy):
            self.all_paths_timer.stop()
            QMessageBox.information(self, "Show All Paths", "Completed visualizing all paths.")
            self.set_ui_enabled(True)
            print("Completed visualizing all paths")
            return

        # Now, proceed with visualization
        node = self.all_target_nodes_copy[self.current_target_index]
        print(f"Visualizing path {self.current_target_index + 1} of {len(self.all_target_nodes_copy)}")
        print(f"Visualizing path to node: {node.name}")

        # Set end node to this node
        self.end_node = node

        # Reset grid before each run
        self.reset_grid()
        print("Grid reset before visualizing the current path")

        # Get start and end coordinates
        start = (int(self.start_node.pos().x() // self.node_size),
                 int(self.start_node.pos().y() // self.node_size))
        end = (int(self.end_node.pos().x() // self.node_size),
               int(self.end_node.pos().y() // self.node_size))
        print(f"Start coordinates: {start}, End coordinates: {end}")

        # Determine which algorithm to run based on dropdown selection
        selected_algorithm = self.algorithm_dropdown.currentText()
        print(f"Running algorithm: {selected_algorithm}")

        try:
            # Run the selected algorithm without visualization
            if selected_algorithm == "Dijkstra's":
                path, _ = self.run_dijkstra(start, end, diagonal_neighbors=False, visualize=False)
            elif selected_algorithm == "Bellman-Ford":
                path, _ = self.run_bellman_ford(start, end, diagonal_neighbors=False, visualize=False)
            elif selected_algorithm.startswith("A*"):
                # Determine heuristic type based on selection
                if selected_algorithm == "A* (Manhattan Distance)":
                    heuristic_type = "Manhattan"
                elif selected_algorithm == "A* (Euclidean Distance)":
                    heuristic_type = "Euclidean"
                elif selected_algorithm == "A* (Modified Euclidean 1.2x Y Priority)":
                    heuristic_type = "Modified Euclidean"
                else:
                    heuristic_type = "Manhattan"  # Default heuristic

                path, _ = self.run_astar(start, end, diagonal_neighbors=False, visualize=False,
                                         heuristic_type=heuristic_type)
            elif selected_algorithm == "SPFA":
                path, _ = self.run_spfa(start, end, diagonal_neighbors=False, visualize=False)
            elif selected_algorithm == "Johnson's":
                path, _ = self.run_johnsons(start, end, diagonal_neighbors=False, visualize=False)
            elif selected_algorithm == "Johnson's with A*":
                path, _ = self.run_johnsons_astar(start, end, diagonal_neighbors=False, visualize=False)
            else:
                path = None

            # Visualize the path if found
            if path:
                for node_coords in path:
                    if node_coords != start and node_coords != end:
                        self.grid[node_coords[1]][node_coords[0]].set_path()
                print(f"Path to {node.name} visualized successfully.")
            else:
                print(f"No path found to {node.name}.")

            # Update the counter label
            self.counter_label.setText(
                f"Visualized paths: {self.current_target_index + 1}/{len(self.all_target_nodes_copy)}")
            QApplication.processEvents()  # Update the UI in real-time

        except Exception as e:
            print(f"Exception during path visualization: {e}")
            QMessageBox.warning(self, "Error", f"An error occurred while visualizing the path:\n{e}")
            self.all_paths_timer.stop()
            self.set_ui_enabled(True)
            return

        # Move to the next target node
        self.current_target_index += 1

    def visualize_path_step_by_step(self):
        """Visualize the path traversal with a delay."""
        self.step_index = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_step)
        self.timer.start(100)  # 200 ms delay

        # Display the total path length in addition to nodes searched
        total_path_length = len(self.search_path)

        # Check if we're using Johnson's algorithm
        if self.algorithm_dropdown.currentText() == "Johnson's":
            # Keep the existing counter label text which shows both mandatory and pathfinding visits
            current_text = self.counter_label.text()
            self.counter_label.setText(f"{current_text}, Path Length: {total_path_length}")
        else:
            # Standard display for other algorithms
            self.counter_label.setText(f"Nodes Searched: {self.nodes_searched}, Path Length: {total_path_length}")

    def update_step(self):
        """Update the grid one step at a time."""
        if self.step_index < len(self.search_path):
            x, y = self.search_path[self.step_index]
            node = self.grid[y][x]
            if not node.is_aisle and not node.is_start and not node.is_end:
                node.set_path()
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

    def run_random_benchmarks(self):
        """Run benchmarks with both orthogonal and diagonal movement."""
        has_negative = self.has_negative_weights()
        if has_negative:
            compatible_algorithms = [algo for algo, caps in self.algorithm_capabilities.items()
                                     if caps["handles_negative"]]
            self.algorithm_dropdown.clear()
            self.algorithm_dropdown.addItems(compatible_algorithms)
            QMessageBox.information(
                self,
                "Negative Weights Detected",
                f"Only running algorithms that support negative weights:\n{', '.join(compatible_algorithms)}"
            )

        num_runs = self.benchmark_spinbox.value()

        # Determine target nodes
        if self.all_nodes_checkbox.isChecked():
            target_nodes = [
                node for row in self.grid for node in row
                if not node.is_obstacle and not node.is_aisle and node != self.start_node
            ]
        else:
            target_nodes = [node_info['node'] for node_info in self.item_nodes]

        if not target_nodes:
            QMessageBox.warning(self, "Error", "No target nodes available for benchmarking.")
            return

        use_fixed_start = self.use_start_as_benchmark_start_checkbox.isChecked()

        # Setup start nodes
        if use_fixed_start:
            if not self.start_node:
                QMessageBox.warning(self, "Error", "Start node is not set.")
                return
            start_nodes = [self.start_node] * num_runs
        else:
            traversable_nodes = [
                node for row in self.grid for node in row
                if not node.is_obstacle and not node.is_aisle
            ]
            if not traversable_nodes:
                QMessageBox.warning(self, "Error", "No traversable nodes available.")
                return
            start_nodes = [random.choice(traversable_nodes) for _ in range(num_runs)]

        # Initialize benchmark data structure for both movement types
        benchmark_data = {
            'orthogonal': {
                'num_runs': num_runs,
                'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                'mode': 'Orthogonal Movement',
                'runs': []
            },
            'diagonal': {
                'num_runs': num_runs,
                'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                'mode': 'Diagonal Movement',
                'runs': []
            }
        }

        # Disable UI during benchmarking
        self.set_ui_enabled(False)

        # Run benchmarks for both movement types
        for movement_type in ['orthogonal', 'diagonal']:
            diagonal_enabled = (movement_type == 'diagonal')

            for run in range(1, num_runs + 1):
                start_node = start_nodes[run - 1]
                self.set_start_node(start_node)

                self.counter_label.setText(
                    f"Benchmark Run {run}/{num_runs} - {movement_type.capitalize()} Movement"
                )
                QApplication.processEvents()

                # Run benchmarking with specified movement type
                run_metrics = self.benchmark_single_run(target_nodes, diagonal_enabled)

                benchmark_data[movement_type]['runs'].append({
                    'run_number': run,
                    'start_node': self.get_node_position(start_node),
                    'metrics': run_metrics
                })

        # Save results
        output_filename = f"random_benchmarks_{num_runs}_runs_{benchmark_data['orthogonal']['timestamp']}.json"
        output_filepath = os.path.join(self.scenario_dir, output_filename)

        try:
            with open(output_filepath, 'w') as f:
                json.dump(benchmark_data, f, indent=4)
            QMessageBox.information(self, "Benchmark Saved", f"Results saved to {output_filepath}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save benchmark results:\n{e}")

        # Process and display results for both movement types
        for movement_type in ['orthogonal', 'diagonal']:
            averaged_metrics = self.process_benchmark_data(benchmark_data[movement_type])
            self.display_benchmark_results(averaged_metrics, movement_type)
            self.plot_benchmark_results(
                averaged_metrics,
                num_runs,
                f"{movement_type.capitalize()} Movement"
            )

        # Re-enable UI
        self.set_ui_enabled(True)

        QMessageBox.information(
            self,
            "Benchmarks Completed",
            f"Completed {num_runs} runs with both orthogonal and diagonal movement."
        )

    def process_benchmark_data(self, benchmark_data):
        """
        Process the collected benchmark data to compute average metrics per algorithm.

        Parameters:
            benchmark_data (dict): The raw benchmark data collected from runs.

        Returns:
            dict: A dictionary containing averaged metrics per algorithm.
        """
        # Initialize a dictionary to accumulate metrics
        accumulated_metrics = {}

        for run in benchmark_data['runs']:
            metrics = run['metrics']
            for algorithm, alg_metrics in metrics.items():
                if algorithm not in accumulated_metrics:
                    accumulated_metrics[algorithm] = {
                        'total_path_length': 0,
                        'total_nodes_searched': 0,
                        'total_time_taken': 0,
                        'valid_runs': 0
                    }
                if alg_metrics['avg_path_length'] is not None:
                    accumulated_metrics[algorithm]['total_path_length'] += alg_metrics['avg_path_length']
                    accumulated_metrics[algorithm]['total_nodes_searched'] += alg_metrics['avg_nodes_searched']
                    accumulated_metrics[algorithm]['total_time_taken'] += alg_metrics['avg_time_taken']
                    accumulated_metrics[algorithm]['valid_runs'] += 1

        # Calculate averages
        averaged_metrics = {}
        for algorithm, data in accumulated_metrics.items():
            if data['valid_runs'] > 0:
                averaged_metrics[algorithm] = {
                    'avg_path_length': data['total_path_length'] / data['valid_runs'],
                    'avg_nodes_searched': data['total_nodes_searched'] / data['valid_runs'],
                    'avg_time_taken': data['total_time_taken'] / data['valid_runs']
                }
            else:
                averaged_metrics[algorithm] = {
                    'avg_path_length': None,
                    'avg_nodes_searched': None,
                    'avg_time_taken': None
                }

        return averaged_metrics

    def display_benchmark_results(self, averaged_metrics, movement_type=""):
        """Display benchmark results with movement type specification."""
        results_str = f"Benchmark Results ({movement_type}):\n"

        for algorithm, metrics in averaged_metrics.items():
            results_str += f"\nAlgorithm: {algorithm}\n"
            if metrics['avg_path_length'] is not None:
                results_str += f"  - Average Path Length: {metrics['avg_path_length']:.2f}\n"
                results_str += f"  - Average Nodes Searched: {metrics['avg_nodes_searched']:.2f}\n"
                results_str += f"  - Average Time Taken: {metrics['avg_time_taken']:.4f} seconds\n"

                # Add Johnson's specific metrics if available
                if 'avg_mandatory_visits' in metrics and metrics['avg_mandatory_visits'] is not None:
                    results_str += f"  - Average Mandatory Visits: {metrics['avg_mandatory_visits']:.2f}\n"
                if 'avg_pathfinding_visits' in metrics and metrics['avg_pathfinding_visits'] is not None:
                    results_str += f"  - Average Pathfinding Visits: {metrics['avg_pathfinding_visits']:.2f}\n"
            else:
                results_str += "  - No valid paths found.\n"

        # Show results in a message box
        QMessageBox.information(self, f"Benchmark Results - {movement_type}", results_str)
        print(results_str)  # Also print to console for debugging

    def save_benchmark_results(self, metrics_per_algorithm):
        """Save the benchmark results to a JSON file."""
        # Save the benchmark results in a JSON file
        filename = os.path.join(self.scenario_dir, "benchmark_results.json")
        with open(filename, 'w') as f:
            json.dump(metrics_per_algorithm, f, indent=4)

        QMessageBox.information(self, "Benchmark Saved", f"Benchmark results saved to {filename}")

    def plot_benchmark_results(self, averaged_metrics, num_runs, movement_type):
        """Plot benchmark results with movement type specification."""
        # Create plots directory if it doesn't exist
        plots_dir = os.path.join(self.scenario_dir, "benchmark_plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        graph_size = f"{self.grid_size}x{self.grid_size}"
        layout_type = self.layout_dropdown.currentText().replace(" ", "_")

        # Extract data for plotting
        algorithms = list(averaged_metrics.keys())
        avg_path_length = []
        avg_nodes_searched = []
        avg_time_taken = []

        for alg in algorithms:
            metrics = averaged_metrics[alg]
            avg_path_length.append(metrics['avg_path_length'] if metrics['avg_path_length'] is not None else 0)
            avg_nodes_searched.append(metrics['avg_nodes_searched'] if metrics['avg_nodes_searched'] is not None else 0)
            avg_time_taken.append(metrics['avg_time_taken'] if metrics['avg_time_taken'] is not None else 0)

        # Define plot configurations
        plot_configs = [
            {
                'data': avg_path_length,
                'ylabel': 'Average Path Length',
                'title': f'Average Path Length per Algorithm\n({movement_type}, {num_runs} Runs)',
                'filename': f'average_path_length_{graph_size}_{layout_type}_{movement_type.replace(" ", "_")}_{timestamp}.png',
                'color': 'skyblue'
            },
            {
                'data': avg_nodes_searched,
                'ylabel': 'Average Nodes Searched',
                'title': f'Average Nodes Searched per Algorithm\n({movement_type}, {num_runs} Runs)',
                'filename': f'average_nodes_searched_{graph_size}_{layout_type}_{movement_type.replace(" ", "_")}_{timestamp}.png',
                'color': 'lightgreen'
            },
            {
                'data': avg_time_taken,
                'ylabel': 'Average Time Taken (seconds)',
                'title': f'Average Time Taken per Algorithm\n({movement_type}, {num_runs} Runs)',
                'filename': f'average_time_taken_{graph_size}_{layout_type}_{movement_type.replace(" ", "_")}_{timestamp}.png',
                'color': 'salmon'
            }
        ]

        # Generate and save each plot
        for config in plot_configs:
            plt.figure(figsize=(12, 6))
            bars = plt.bar(algorithms, config['data'], color=config['color'])
            plt.xlabel('Algorithm', fontsize=12)
            plt.ylabel(config['ylabel'], fontsize=12)
            plt.title(config['title'], fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(fontsize=10)
            plt.tight_layout()

            # Add value labels on top of each bar
            for bar in bars:
                height = bar.get_height()
                if height is not None:
                    label = f'{height:.4f}' if "Time Taken" in config['title'] else f'{height:.2f}'
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        height,
                        label,
                        ha='center',
                        va='bottom',
                        fontsize=10
                    )

            # Save the plot
            plot_path = os.path.join(plots_dir, config['filename'])
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

        # Inform the user
        QMessageBox.information(
            self,
            "Benchmark Plots Saved",
            f"Benchmark plots for {movement_type} have been saved in:\n{plots_dir}"
        )
        print(f"Benchmark plots for {movement_type} saved in {plots_dir}")
    def benchmark_single_run(self, target_nodes, diagonal_enabled=False):
        """Modified benchmark_single_run to support diagonal movement."""
        self.reset_grid()

        if not self.start_node:
            return {}

        metrics_per_algorithm = {algorithm: {
            'total_path_length': 0,
            'total_nodes_searched': 0,
            'total_time_taken': 0,
            'valid_runs': 0,
            'total_mandatory_visits': 0 if 'Johnson' in algorithm else None,
            'total_pathfinding_visits': 0 if 'Johnson' in algorithm else None
        } for algorithm in self.algorithm_capabilities.keys()}

        for node in target_nodes:
            if node == self.start_node:
                continue

            self.end_node = node
            self.reset_grid()

            start_coords = (
                int(self.start_node.pos().x() // self.node_size),
                int(self.start_node.pos().y() // self.node_size)
            )
            end_coords = (
                int(self.end_node.pos().x() // self.node_size),
                int(self.end_node.pos().y() // self.node_size)
            )

            for algorithm in self.algorithm_capabilities.keys():
                start_time = time.time()

                if algorithm.startswith("A*"):
                    heuristic_type = algorithm.split("(")[1].split(")")[0].strip()
                    path, nodes_searched = self.run_astar(
                        start_coords, end_coords, diagonal_enabled, False, heuristic_type
                    )
                elif algorithm == "Dijkstra's":
                    path, nodes_searched = self.run_dijkstra(
                        start_coords, end_coords, diagonal_enabled, False
                    )
                elif algorithm == "Bellman-Ford":
                    path, nodes_searched = self.run_bellman_ford(
                        start_coords, end_coords, diagonal_enabled, False
                    )
                elif algorithm == "SPFA":
                    path, nodes_searched = self.run_spfa(
                        start_coords, end_coords, diagonal_enabled, False
                    )
                elif "Johnson" in algorithm:
                    path, (mandatory_visits, pathfinding_visits) = (
                        self.run_johnsons_astar if "with A*" in algorithm else self.run_johnsons
                    )(start_coords, end_coords, diagonal_enabled, False)
                    nodes_searched = pathfinding_visits
                    metrics_per_algorithm[algorithm]['total_mandatory_visits'] += mandatory_visits
                    metrics_per_algorithm[algorithm]['total_pathfinding_visits'] += pathfinding_visits

                time_taken = time.time() - start_time

                if path:
                    path_length = len(path)
                    metrics_per_algorithm[algorithm]['total_path_length'] += path_length
                    metrics_per_algorithm[algorithm]['total_nodes_searched'] += nodes_searched
                    metrics_per_algorithm[algorithm]['total_time_taken'] += time_taken
                    metrics_per_algorithm[algorithm]['valid_runs'] += 1

        # Calculate averages
        return {algo: {
            'avg_path_length': data['total_path_length'] / data['valid_runs'] if data['valid_runs'] > 0 else None,
            'avg_nodes_searched': data['total_nodes_searched'] / data['valid_runs'] if data['valid_runs'] > 0 else None,
            'avg_time_taken': data['total_time_taken'] / data['valid_runs'] if data['valid_runs'] > 0 else None,
            'avg_mandatory_visits': (data['total_mandatory_visits'] / data['valid_runs']
                                     if data['total_mandatory_visits'] is not None and data[
                'valid_runs'] > 0 else None),
            'avg_pathfinding_visits': (data['total_pathfinding_visits'] / data['valid_runs']
                                       if data['total_pathfinding_visits'] is not None and data[
                'valid_runs'] > 0 else None)
        } for algo, data in metrics_per_algorithm.items()}

    def process_benchmark_data(self, benchmark_data):
        """
        Process the collected benchmark data to compute average metrics per algorithm.

        Parameters:
            benchmark_data (dict): The raw benchmark data collected from runs.

        Returns:
            dict: A dictionary containing averaged metrics per algorithm.
        """
        # Initialize a dictionary to accumulate metrics
        accumulated_metrics = {}

        for run in benchmark_data['runs']:
            metrics = run['metrics']
            for algorithm, alg_metrics in metrics.items():
                if algorithm not in accumulated_metrics:
                    accumulated_metrics[algorithm] = {
                        'total_path_length': 0,
                        'total_nodes_searched': 0,
                        'total_time_taken': 0,
                        'valid_runs': 0
                    }
                if alg_metrics['avg_path_length'] is not None:
                    accumulated_metrics[algorithm]['total_path_length'] += alg_metrics['avg_path_length']
                    accumulated_metrics[algorithm]['total_nodes_searched'] += alg_metrics['avg_nodes_searched']
                    accumulated_metrics[algorithm]['total_time_taken'] += alg_metrics['avg_time_taken']
                    accumulated_metrics[algorithm]['valid_runs'] += 1

        # Calculate averages
        averaged_metrics = {}
        for algorithm, data in accumulated_metrics.items():
            if data['valid_runs'] > 0:
                averaged_metrics[algorithm] = {
                    'avg_path_length': data['total_path_length'] / data['valid_runs'],
                    'avg_nodes_searched': data['total_nodes_searched'] / data['valid_runs'],
                    'avg_time_taken': data['total_time_taken'] / data['valid_runs']
                }
            else:
                averaged_metrics[algorithm] = {
                    'avg_path_length': None,
                    'avg_nodes_searched': None,
                    'avg_time_taken': None
                }

        return averaged_metrics

    # *** New Method Addition Start ***
    def handle_show_all_paths(self):
        print("handle_show_all_paths called")
        if not self.start_node:
            print("Start node not set")
            QMessageBox.warning(self, "Error", "Start node not set.")
            return

        # Determine target nodes based on the 'all_nodes_checkbox' state
        if self.all_nodes_checkbox.isChecked():
            self.all_target_nodes = [
                node for row in self.grid for node in row
                if not node.is_obstacle and not node.is_aisle and node != self.start_node
            ]
            print(f"All nodes selected for benchmarking: {len(self.all_target_nodes)} targets")
        else:
            if not hasattr(self, 'item_nodes') or not self.item_nodes:
                print("No item nodes available")
                QMessageBox.warning(self, "Error", "No target nodes available.")
                return
            self.all_target_nodes = [node_info['node'] for node_info in self.item_nodes]
            print(f"Item nodes selected for benchmarking: {len(self.all_target_nodes)} targets")

        if not self.all_target_nodes:
            print("No target nodes available for visualization")
            QMessageBox.warning(self, "Error", "No target nodes available for visualization.")
            return

        reply = QMessageBox.question(self, 'Show All Paths',
                                     f"This operation will visualize paths to {'all traversable nodes' if self.all_nodes_checkbox.isChecked() else 'all item nodes'}. Do you want to proceed?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.No:
            print("User cancelled the Show All Paths operation")
            return

        has_negative = self.has_negative_weights()
        selected_algorithm = self.algorithm_dropdown.currentText()
        print(f"Selected algorithm: {selected_algorithm}, Has negative weights: {has_negative}")

        if has_negative and not self.algorithm_capabilities[selected_algorithm]["handles_negative"]:
            compatible_algorithms = [name for name, caps in self.algorithm_capabilities.items()
                                     if caps["handles_negative"]]
            QMessageBox.warning(
                self,
                "Invalid Algorithm Selection",
                f"The selected algorithm '{selected_algorithm}' cannot handle negative edge weights.\n"
                f"Please choose one of these algorithms:\n{', '.join(compatible_algorithms)}"
            )
            print("Selected algorithm cannot handle negative weights")
            return

        # Disable UI elements to prevent interference during the operation
        self.set_ui_enabled(False)
        print("UI elements disabled for Show All Paths operation")

        # Initialize a list of target nodes
        self.all_target_nodes_copy = self.all_target_nodes.copy()
        self.current_target_index = 0
        print(f"Starting visualization of {len(self.all_target_nodes_copy)} paths")

        # Initialize a timer to handle path visualization sequentially
        self.all_paths_timer = QTimer(self)
        self.all_paths_timer.timeout.connect(self.visualize_next_path)
        self.all_paths_timer.start(200)  # 100 ms delay between paths

        # Optionally, reset any existing paths before starting
        self.reset_grid()
        print("Grid reset for path visualization")


    def set_ui_enabled(self, enabled):
        """Enable or disable UI elements during bulk operations."""
        self.spacing_dropdown.setEnabled(enabled)
        self.algorithm_dropdown.setEnabled(enabled)
        self.layout_dropdown.setEnabled(enabled)
        self.aisle_spinbox.setEnabled(enabled)
        self.shelf_spinbox.setEnabled(enabled)
        self.start_button.setEnabled(enabled)
        self.end_button.setEnabled(enabled)
        self.barrier_button.setEnabled(enabled)
        self.diagonal_checkbox.setEnabled(enabled)
        self.search_button.setEnabled(enabled)
        self.generate_button.setEnabled(enabled)
        self.save_button.setEnabled(enabled)
        self.load_dropdown.setEnabled(enabled)
        self.clear_button.setEnabled(enabled)
        self.item_dropdown.setEnabled(enabled)
        self.zoom_in_button.setEnabled(enabled)
        self.zoom_out_button.setEnabled(enabled)
        # self.benchmark_button.setEnabled(enabled)  # Removed
        self.show_all_paths_button.setEnabled(enabled)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WarehouseVisualizer()
    sys.exit(app.exec())
