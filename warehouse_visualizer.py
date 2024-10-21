import sys
import heapq
import pandas as pd
import random
from math import sqrt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsRectItem,
    QGraphicsTextItem, QVBoxLayout, QWidget, QPushButton, QLabel, QComboBox, QSpinBox,
    QFileDialog, QInputDialog, QMessageBox, QCheckBox, QGraphicsItem
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

    def set_edge_weight_color(self):
        """
        Update the node's color based on its edge weight and update the weight label.
        Aisle nodes retain their designated colors and are not altered by edge weights.
        """
        if not self.is_aisle:
            red_intensity = max(0, 255 - (self.edge_weight - 1) * 25)
            self.setBrush(QBrush(QColor(255, red_intensity, red_intensity)))  # Gradient from red to white

            if self.weight_label:
                self.weight_label.setPlainText(str(self.edge_weight))
                self.update_weight_label_position()
        else:
            # For aisle nodes, ensure their color remains unchanged
            # No action needed here since aisle colors are managed elsewhere
            pass

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

        Parameters:
            event (QWheelEvent): The wheel event.
        """
        try:
            current_time = time.time()
            if current_time - self.last_scroll_time > 0.1:  # 100ms delay
                # Using the deprecated delta() method as per requirement
                delta_y = event.delta()
                if delta_y != 0:
                    delta = delta_y / 120  # Standard scroll value
                    old_weight = self.edge_weight
                    self.edge_weight = min(10, max(1, self.edge_weight + int(delta)))
                    print(f"Edge weight changed from {old_weight} to {self.edge_weight}")
                    self.set_edge_weight_color()
                self.last_scroll_time = current_time
        except AttributeError as ae:
            print(f"AttributeError in wheelEvent: {ae}")
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

    def set_visited(self):
        """Mark the node as visited (green) unless it's an aisle."""
        if not self.is_aisle:
            self.setBrush(QBrush(QColor(0, 255, 0)))  # Green when visited
        else:
            if not self.is_start and not self.is_end:
                self.set_as_aisle(self.brush().color())  # Retain aisle color
            # If it's start or end, do not change the color

    def set_path(self):
        """Mark the node as part of the path (blue) unless it's an aisle."""
        if not self.is_aisle:
            self.setBrush(QBrush(QColor(0, 0, 255)))  # Blue when part of the path
        else:
            if not self.is_start and not self.is_end:
                self.set_as_aisle(self.brush().color())  # Retain aisle color
            # If it's start or end, do not change the color

    def reset(self):
        """Reset the node to its default visual state without altering edge_weight."""
        if not self.is_obstacle and not self.is_start and not self.is_end:
            if not self.is_aisle:
                self.setBrush(QBrush(QColor(255, 255, 255)))  # White for reset
                if self.weight_label:
                    self.weight_label.show()
                if self.item_label:
                    self.item_label.show()
            else:
                # Retain original aisle color by passing the current color
                self.set_as_aisle(self.brush().color())
                if self.weight_label:
                    self.weight_label.hide()
                if self.item_label:
                    self.item_label.hide()
        else:
            # Maintain state for start, end, and barriers
            if self.is_start:
                self.setBrush(QBrush(QColor(0, 255, 0)))  # Green for start
            elif self.is_end:
                self.setBrush(QBrush(QColor(255, 0, 0)))  # Red for end
            elif self.is_obstacle:
                self.setBrush(QBrush(QColor(255, 255, 0)))  # Yellow for barriers

        # Update the visual representation based on the current edge_weight
        if not self.is_aisle:
            self.set_edge_weight_color()

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

        self.orientation_type = 'vertical'  # Default value, can be updated later
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
        self.algorithm_dropdown.addItems([
            "A* (Manhattan Distance)",
            "A* (Euclidean Distance)",
            "A* (Modified Euclidean 1.2x Y Priority)",
            "Dijkstra's",
            "Bellman-Ford"
        ])
        # Dropdown to select the warehouse layout
        self.layout_dropdown = QComboBox(self)
        self.layout_dropdown.addItems(["Vertical Aisles", "Horizontal Aisles", "Mixed Aisles"])

        # Spin boxes to adjust the number of aisles and shelves
        self.aisle_spinbox = QSpinBox(self)
        self.aisle_spinbox.setRange(1, 50)
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

        self.diagonal_checkbox = QCheckBox("Allow Diagonal Neighbors", self)

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

        # Benchmark button
        self.benchmark_button = QPushButton("Benchmark Algorithms", self)
        self.benchmark_button.clicked.connect(self.benchmark_algorithms)

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
        layout.addWidget(self.diagonal_checkbox)
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

        # Dropdown to select an item as the end node
        self.item_dropdown = QComboBox(self)
        self.item_dropdown.addItem("Select Item")  # Default placeholder
        self.item_dropdown.currentIndexChanged.connect(
            self.set_end_node_from_dropdown)  # Connect the dropdown selection to a method
        layout.addWidget(QLabel("Select Item as End Node:", self))
        layout.addWidget(self.item_dropdown)

        # Add zoom buttons
        self.zoom_in_button = QPushButton("Zoom In", self)
        self.zoom_in_button.clicked.connect(self.zoom_in)
        layout.addWidget(self.zoom_in_button)

        self.zoom_out_button = QPushButton("Zoom Out", self)
        self.zoom_out_button.clicked.connect(self.zoom_out)
        layout.addWidget(self.zoom_out_button)

        layout.addWidget(self.benchmark_button)  # Add benchmark button to the layout

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
        try:
            self.clear_all()

            # Generate warehouse data
            warehouse_data = generate_warehouse_data(
                num_aisles=self.num_aisles,
                max_shelves_per_aisle=self.max_shelves_per_aisle,
                save_to_csv=False
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

        # Check if diagonal neighbors are allowed (toggle checkbox)
        diagonal_neighbors = self.diagonal_checkbox.isChecked()

        # Determine which algorithm to run based on dropdown selection
        selected_algorithm = self.algorithm_dropdown.currentText()

        if selected_algorithm == "Dijkstra's":
            path, nodes_searched = self.run_dijkstra(start, end, diagonal_neighbors, visualize=True)
        elif selected_algorithm == "Bellman-Ford":
            path, nodes_searched = self.run_bellman_ford(start, end, diagonal_neighbors, visualize=True)
        elif selected_algorithm == "A* (Manhattan Distance)":
            path, nodes_searched = self.run_astar(start, end, diagonal_neighbors, visualize=True)
        elif selected_algorithm == "A* (Euclidean Distance)":
            path, nodes_searched = self.run_astar(start, end, diagonal_neighbors, visualize=True)
        elif selected_algorithm == "A* (Modified Euclidean 1.2x Y Priority)":
            path, nodes_searched = self.run_astar(start, end, diagonal_neighbors, visualize=True)
        else:
            self.counter_label.setText("Invalid algorithm selection.")
            return

        # If a path is found, visualize it step by step
        if path:
            self.search_path = path
            self.visualize_path_step_by_step()
        else:
            self.counter_label.setText("No path found.")

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

    def is_obstacle(self, x, y):
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

    def heuristic(self, node, goal):
        """Heuristic function that supports different types of heuristics, including custom ones."""
        (x1, y1) = node
        (x2, y2) = goal

        selected_algorithm = self.algorithm_dropdown.currentText()

        # Base heuristics based on the algorithm selected
        if "Manhattan Distance" in selected_algorithm:
            return abs(x1 - x2) + abs(y1 - y2)

        elif "Euclidean Distance" in selected_algorithm:
            return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        elif "Modified Euclidean" in selected_algorithm:
            return sqrt((x1 - x2) ** 2 + (1.2 * (y1 - y2)) ** 2)

        # Default fallback to Manhattan distance if no valid heuristic is selected
        return abs(x1 - x2) + abs(y1 - y2)

    def get_neighbors(self, node, diagonal_neighbors=False):
        """Get valid neighboring nodes and their edge weights."""
        (x, y) = node

        four_neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        eight_neighbors = [(-1, -1), (1, -1), (-1, 1), (1, 1)]

        potential_neighbors = four_neighbors
        if diagonal_neighbors:
            potential_neighbors += eight_neighbors

        neighbors = []
        for dx, dy in potential_neighbors:
            nx, ny = x + dx, y + dy
            if self.is_valid_position(nx, ny) and self.is_traversable(nx, ny):
                neighbor_coords = (nx, ny)
                neighbor_node = self.grid[ny][nx]
                neighbors.append((neighbor_coords, neighbor_node.edge_weight))  # Correctly retrieve edge_weight

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

    def benchmark_algorithms(self):
        """Benchmark the algorithm over all viable target nodes and collect metrics for each algorithm."""
        if not self.start_node:
            QMessageBox.warning(self, "Error", "Start node not set.")
            return

        if not hasattr(self, 'item_nodes') or not self.item_nodes:
            QMessageBox.warning(self, "Error", "No target nodes available.")
            return

        reply = QMessageBox.question(self, 'Benchmark Algorithms',
                                     "This operation may take some time. Do you want to proceed?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.No:
            return

        # Dictionary to store metrics for each algorithm
        metrics_per_algorithm = {}

        # Updated list of available algorithms
        algorithms = [
            "A* (Manhattan Distance)",
            "A* (Euclidean Distance)",
            "A* (Modified Euclidean 1.2x Y Priority)",
            "Dijkstra's",
            "Bellman-Ford"
        ]

        # Go over each algorithm and benchmark it
        for algorithm in algorithms:
            print(f"Benchmarking with {algorithm}...")

            # Initialize metrics dictionary for each algorithm
            metrics_per_algorithm[algorithm] = {
                'items': {},
                'total_path_length': 0,
                'total_nodes_searched': 0,
                'total_time_taken': 0,
                'valid_items': 0
            }

            total_items = len(self.item_nodes)
            processed_items = 0

            for node_info in self.item_nodes:
                node = node_info['node']
                item_name = node_info['item']
                x = node_info['x']
                y = node_info['y']

                # Skip if end node is the same as start node
                if node == self.start_node:
                    continue

                # Set end node to this node
                self.end_node = node

                # Reset grid before each run
                self.reset_grid()

                # Run the algorithm
                start_time = time.time()

                # Get start and end coordinates
                start_coords = (int(self.start_node.pos().x() // self.node_size),
                                int(self.start_node.pos().y() // self.node_size))
                end_coords = (x, y)

                # Call the respective method based on the selected algorithm
                if algorithm == "Dijkstra's":
                    path, nodes_searched = self.run_dijkstra(start_coords, end_coords, diagonal_neighbors=False,
                                                             visualize=False)
                elif algorithm == "Bellman-Ford":
                    path, nodes_searched = self.run_bellman_ford(start_coords, end_coords, diagonal_neighbors=False,
                                                                 visualize=False)
                else:
                    # For all A* algorithms, use the correct heuristic
                    if "Manhattan" in algorithm:
                        path, nodes_searched = self.run_astar(start_coords, end_coords, diagonal_neighbors=False,
                                                              visualize=False)
                    elif "Euclidean" in algorithm:
                        path, nodes_searched = self.run_astar(start_coords, end_coords, diagonal_neighbors=False,
                                                              visualize=False)
                    elif "Modified Euclidean" in algorithm:
                        path, nodes_searched = self.run_astar(start_coords, end_coords, diagonal_neighbors=False,
                                                              visualize=False)

                end_time = time.time()
                time_taken = end_time - start_time  # In seconds

                if path:
                    path_length = len(path)
                else:
                    path_length = None  # No path found

                # Store metrics for the current item
                metrics_per_algorithm[algorithm]['items'][item_name] = {
                    'path_length': path_length,
                    'nodes_searched': nodes_searched,
                    'time_taken': time_taken
                }

                # Update totals if a valid path was found
                if path_length is not None:
                    metrics_per_algorithm[algorithm]['total_path_length'] += path_length
                    metrics_per_algorithm[algorithm]['total_nodes_searched'] += nodes_searched
                    metrics_per_algorithm[algorithm]['total_time_taken'] += time_taken
                    metrics_per_algorithm[algorithm]['valid_items'] += 1

                processed_items += 1

            # Calculate averages
            if metrics_per_algorithm[algorithm]['valid_items'] > 0:
                metrics_per_algorithm[algorithm]['avg_path_length'] = metrics_per_algorithm[algorithm][
                                                                          'total_path_length'] / \
                                                                      metrics_per_algorithm[algorithm]['valid_items']
                metrics_per_algorithm[algorithm]['avg_nodes_searched'] = metrics_per_algorithm[algorithm][
                                                                             'total_nodes_searched'] / \
                                                                         metrics_per_algorithm[algorithm]['valid_items']
                metrics_per_algorithm[algorithm]['avg_time_taken'] = metrics_per_algorithm[algorithm][
                                                                         'total_time_taken'] / \
                                                                     metrics_per_algorithm[algorithm]['valid_items']
            else:
                metrics_per_algorithm[algorithm]['avg_path_length'] = None
                metrics_per_algorithm[algorithm]['avg_nodes_searched'] = None
                metrics_per_algorithm[algorithm]['avg_time_taken'] = None

            print(f"Finished benchmarking {algorithm}.")

        # Display and save results
        self.display_benchmark_results(metrics_per_algorithm)
        self.save_benchmark_results(metrics_per_algorithm)

    def display_benchmark_results(self, metrics_per_algorithm):
        """Display benchmark results for all algorithms."""
        results_str = "Benchmark Results:\n"

        for algorithm, data in metrics_per_algorithm.items():
            results_str += f"\nAlgorithm: {algorithm}\n"
            results_str += f"Average Path Length: {data['avg_path_length']}\n"
            results_str += f"Average Nodes Searched: {data['avg_nodes_searched']}\n"
            results_str += f"Average Time Taken: {data['avg_time_taken']:.4f} seconds\n"

        # Show results in a message box
        QMessageBox.information(self, "Benchmark Results", results_str)

        # Also print to console for debugging
        print(results_str)

    def save_benchmark_results(self, metrics_per_algorithm):
        """Save the benchmark results to a JSON file."""
        # Save the benchmark results in a JSON file
        filename = os.path.join(self.scenario_dir, "benchmark_results.json")
        with open(filename, 'w') as f:
            json.dump(metrics_per_algorithm, f, indent=4)

        QMessageBox.information(self, "Benchmark Saved", f"Benchmark results saved to {filename}")

    def run_astar(self, start, end, diagonal_neighbors=False, visualize=True):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, end)}
        self.nodes_searched = 0

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                path = self.reconstruct_path(came_from, current)
                return path, self.nodes_searched

            self.nodes_searched += 1

            if visualize:
                self.grid[current[1]][current[0]].set_visited()
                self.counter_label.setText(f"Nodes Searched: {self.nodes_searched}")
                QApplication.processEvents()

            for neighbor_coords, weight in self.get_neighbors(current, diagonal_neighbors):
                tentative_g_score = g_score[current] + weight  # Use edge weight here

                if neighbor_coords not in g_score or tentative_g_score < g_score[neighbor_coords]:
                    came_from[neighbor_coords] = current
                    g_score[neighbor_coords] = tentative_g_score
                    f_score[neighbor_coords] = tentative_g_score + self.heuristic(neighbor_coords, end)

                    if neighbor_coords not in [i[1] for i in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor_coords], neighbor_coords))

        return None, self.nodes_searched


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WarehouseVisualizer()
    sys.exit(app.exec())
