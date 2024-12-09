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
import time



def generate_warehouse_data(num_aisles=5, max_shelves_per_aisle=10, save_to_csv=True, filename=None):
    
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


    df_inventory = pd.DataFrame(inventory_data)


    if save_to_csv and filename:
        df_inventory.to_csv(filename, index=False)
        print(f"Warehouse layout saved to {filename}")

    return df_inventory


class Node(QGraphicsRectItem):
    def __init__(self, x, y, size, name, parent_window):
        
        super().__init__(0, 0, size, size)
        self.setPos(x * size, y * size)
        self.name = name
        self.setBrush(QBrush(QColor(255, 255, 255)))
        self.parent_window = parent_window
        self.is_obstacle = False
        self.is_start = False
        self.is_end = False
        self.edge_weight = 1  
        self.is_aisle = False
        self.last_scroll_time = time.time()
        self.original_aisle_color = None

        self.setFlags(QGraphicsRectItem.GraphicsItemFlag.ItemIsFocusable)


        self.weight_label = QGraphicsTextItem(str(self.edge_weight), self)
        weight_font = QFont()
        weight_font.setPointSize(10)
        weight_font.setBold(True)
        self.weight_label.setFont(weight_font)

        
        self.update_weight_label_position()

        
        self.item_label = QGraphicsTextItem("", self)
        item_font = QFont()
        item_font.setPointSize(8)
        self.item_label.setFont(item_font)
        self.item_label.setDefaultTextColor(QColor(0, 0, 0))  

        
        if self.is_aisle:
            self.weight_label.hide()
        if self.is_aisle:
            self.item_label.hide()

    def update_weight_label_position(self):
        
        text_rect = self.weight_label.boundingRect()
        node_rect = self.rect()
        self.weight_label.setPos(
            (node_rect.width() - text_rect.width()) / 2,
            (node_rect.height() - text_rect.height()) / 2
        )

    def update_item_label_position(self):
        
        text_rect = self.item_label.boundingRect()
        node_rect = self.rect()
        self.item_label.setPos(
            (node_rect.width() - text_rect.width()) / 2,
            (node_rect.height() - text_rect.height()) / 2
        )

    def detect_negative_cycle_spfa(self, x, y, new_weight):
        
        
        original_weight = self.edge_weight

        
        self.edge_weight = new_weight

        
        grid_size = self.parent_window.grid_size

        
        distance = {(i, j): float('inf') for i in range(grid_size) for j in range(grid_size)}
        count = {(i, j): 0 for i in range(grid_size) for j in range(grid_size)}

        
        start = (x, y)
        distance[start] = 0

        
        queue = [start]
        in_queue = {(i, j): False for i in range(grid_size) for j in range(grid_size)}
        in_queue[start] = True

        def get_neighbors(node):
            px, py = node
            neighbors = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  
                nx, ny = px + dx, py + dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    if not self.parent_window.grid[ny][nx].is_obstacle:
                        neighbors.append((nx, ny))
            return neighbors

        
        while queue:
            current = queue.pop(0)
            in_queue[current] = False

            
            for next_node in get_neighbors(current):
                weight = self.parent_window.grid[next_node[1]][next_node[0]].edge_weight
                if distance[current] + weight < distance[next_node]:
                    distance[next_node] = distance[current] + weight
                    count[next_node] += 1

                    
                    
                    if count[next_node] >= grid_size * grid_size:
                        
                        self.edge_weight = original_weight
                        return True

                    if not in_queue[next_node]:
                        queue.append(next_node)
                        in_queue[next_node] = True

        
        self.edge_weight = original_weight
        return False

    def set_edge_weight(self, new_weight):
        
        x = int(self.pos().x() / self.parent_window.node_size)
        y = int(self.pos().y() / self.parent_window.node_size)

        
        if (
                new_weight < 0 and
                not self.is_aisle and
                not self.parent_window.is_generating_warehouse and
                (self.edge_weight >= 0 or new_weight < self.edge_weight)  
        ):
            
            if self.detect_negative_cycle_spfa(x, y, new_weight):
                
                new_weight = max(1, self.edge_weight)
                QMessageBox.warning(
                    self.parent_window,
                    "Invalid Weight",
                    "This weight change would create a negative cycle and has been reverted."
                )

        
        self.edge_weight = new_weight
        self.update_color_from_weight()

    def update_color_from_weight(self):
        
        if not self.is_aisle:
            red_intensity = max(0, 255 - (self.edge_weight - 1) * 25)
            self._color = QColor(255, red_intensity, red_intensity)
            self.setBrush(QBrush(self._color))

            if self.weight_label:
                self.weight_label.setPlainText(str(self.edge_weight))
                self.update_weight_label_position()

    def set_item_label(self, text):
        
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
        
        try:
            current_time = time.time()
            if current_time - self.last_scroll_time > 0.1:  
                delta_y = event.delta()
                if delta_y != 0:
                    delta = delta_y / 120  
                    old_weight = self.edge_weight
                    new_weight = min(10, max(-10, self.edge_weight + int(delta)))
                    self.set_edge_weight(new_weight)
                    print(f"Edge weight changed from {old_weight} to {self.edge_weight}")
                self.last_scroll_time = current_time
        except Exception as e:
            print(f"Error in wheelEvent: {e}")

    def mousePressEvent(self, event):
        
        mode = self.parent_window.current_mode
        if mode == 'start':
            self.parent_window.set_start_node(self)
        elif mode == 'end':
            self.parent_window.set_end_node(self)
        elif mode == 'barrier':
            self.set_as_barrier()

    def mouseMoveEvent(self, event):
        
        mode = self.parent_window.current_mode
        if mode == 'barrier':
            self.set_as_barrier()

    def set_as_barrier(self):
        
        if not self.is_aisle and not self.is_start and not self.is_end:
            self.setBrush(QBrush(QColor(255, 255, 0)))  
            self.is_obstacle = True
            if self.weight_label:
                self.weight_label.hide()  
            if self.item_label:
                self.item_label.hide()  

    def set_relaxed(self):
        
        if not self.is_aisle:
            self.setBrush(QBrush(QColor(255, 165, 0)))  
        else:
            if not self.is_start and not self.is_end:
                self.set_as_aisle(self.brush().color())

    def set_visited(self):
        
        if not self.is_aisle:
            self.setBrush(QBrush(QColor(144, 238, 144)))  
        else:
            if not self.is_start and not self.is_end:
                self.set_as_aisle(self.brush().color())

    def set_path(self):
        
        if not self.is_aisle:
            self.setBrush(QBrush(QColor(0, 0, 255)))  
        else:
            if not self.is_start and not self.is_end:
                self.set_as_aisle(self.brush().color())  
            

    def reset(self):
        
        if not self.is_obstacle and not self.is_start and not self.is_end:
            if not self.is_aisle:
                self._color = QColor(255, 255, 255)  
                self.setBrush(QBrush(self._color))
                if self.weight_label:
                    self.weight_label.show()
                if self.item_label:
                    self.item_label.show()
                self.update_color_from_weight()  
            else:
                if self.original_aisle_color:
                    self.set_as_aisle(self.original_aisle_color)
                else:
                    self.set_as_aisle(QColor(150, 150, 250))

    def set_as_start(self):
        
        
        if self.parent_window.end_node == self:
            return

        
        if self.parent_window.start_node:
            self.parent_window.start_node.is_start = False
            self.parent_window.start_node.reset()

        
        self.is_start = True
        self.is_obstacle = False  
        self.setBrush(QBrush(QColor(0, 255, 0)))  
        if self.weight_label:
            self.weight_label.hide()  
        if self.item_label:
            self.item_label.hide()  

        
        self.parent_window.start_node = self

    def set_as_end(self):
        
        
        if self.parent_window.start_node == self:
            return

        
        if self.parent_window.end_node:
            self.parent_window.end_node.is_end = False
            self.parent_window.end_node.reset()

        
        self.is_end = True
        self.is_obstacle = False  
        self.setBrush(QBrush(QColor(255, 0, 0)))  
        if self.weight_label:
            self.weight_label.hide()  
        if self.item_label:
            self.item_label.hide()  

        
        self.parent_window.end_node = self

    def set_as_aisle(self, aisle_color):
        
        self.original_aisle_color = aisle_color
        if self.is_start:
            self.setBrush(QBrush(QColor(0, 255, 0)))  
        elif self.is_end:
            self.setBrush(QBrush(QColor(255, 0, 0)))  
        else:
            self.setBrush(QBrush(aisle_color))  
        self.is_aisle = True
        if self.weight_label:
            self.weight_label.hide()
        if self.item_label:
            self.item_label.hide()


class WarehouseVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.is_generating_warehouse = False


        self.num_aisles = 5  
        self.max_shelves_per_aisle = 10  

        self.node_size = 80
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene, self)


        self.grid = []
        self.spacing = 1  
        self.current_mode = None  
        self.start_node = None
        self.end_node = None
        self.item_nodes = []
        self.orientation_type = 'vertical'  
        
        self.scenario_dir = "scenarios"
        if not os.path.exists(self.scenario_dir):
            os.makedirs(self.scenario_dir)

        self.current_diagonal_state = False

        
        
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


        self.random_benchmark_button = QPushButton("Run Random Benchmarks", self)
        self.random_benchmark_button.clicked.connect(self.run_random_benchmarks)

        
        self.diagonal_checkbox = QCheckBox("Allow Diagonal Neighbors", self)
        self.diagonal_checkbox.stateChanged.connect(self.handle_diagonal_change)

        self.use_start_as_benchmark_start_checkbox = QCheckBox("Use Start Node for Benchmarks", self)
        self.use_start_as_benchmark_start_checkbox.setChecked(True)  


        self.all_nodes_checkbox = QCheckBox("Benchmark Against All Nodes", self)
        self.all_nodes_checkbox.setChecked(False)  

        
        self.counter_label = QLabel("Nodes Searched: 0", self)
        self.benchmark_label = QLabel("Number of Benchmark Runs:", self)

        
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
        self.algorithm_dropdown.currentIndexChanged.connect(self.on_algorithm_changed)
        
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

        
        layout.addWidget(QLabel("Select Item as End Node:", self))
        layout.addWidget(self.item_dropdown)

        
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(self.zoom_in_button)
        zoom_layout.addWidget(self.zoom_out_button)
        layout.addLayout(zoom_layout)

        
        layout.addWidget(self.show_all_paths_button)

        
        benchmark_group = QVBoxLayout()
        benchmark_header = QHBoxLayout()
        benchmark_header.addWidget(self.benchmark_label)
        benchmark_header.addWidget(self.benchmark_spinbox)
        benchmark_group.addLayout(benchmark_header)

        benchmark_buttons = QHBoxLayout()

        benchmark_buttons.addWidget(self.random_benchmark_button)
        benchmark_group.addLayout(benchmark_buttons)

        benchmark_group.addWidget(self.use_start_as_benchmark_start_checkbox)


        benchmark_group.addWidget(self.all_nodes_checkbox)

        layout.addLayout(benchmark_group)

        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.grid_size = 12  
        self.init_grid()

        self.load_scenarios()  

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
        
        new_diagonal_state = bool(state)
        if new_diagonal_state != self.current_diagonal_state:
            print("Diagonal movement state changed.")
            self.current_diagonal_state = new_diagonal_state


    def on_algorithm_changed(self, index):
        
        
        self.reset_grid()

        selected_algorithm = self.algorithm_dropdown.currentText()

        if not selected_algorithm:
            return

        
        if hasattr(self, 'johnsons_graph'):
            delattr(self, 'johnsons_graph')
            print("Cleared cached Johnson's graph due to algorithm change.")

        
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
        
        self.view.scale(1.2, 1.2)  

    def zoom_out(self):
        
        self.view.scale(1 / 1.2, 1 / 1.2)  

    def adjust_zoom(self):
        
        
        self.view.resetTransform()

        
        total_width = self.grid_size * self.node_size
        total_height = self.grid_size * self.node_size


        view_width = self.view.viewport().width()
        view_height = self.view.viewport().height()


        scale_x = view_width / total_width
        scale_y = view_height / total_height


        scale = min(scale_x, scale_y) * 0.9  

        self.view.scale(scale, scale)

    def init_grid(self):
        
        
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
        
        for row in self.grid:
            for node in row:
                if not node.is_start and not node.is_end and not node.is_obstacle:
                    node.reset()
        self.counter_label.setText("Nodes Searched: 0")

    def clear_all(self):
        
        for row in self.grid:
            for node in row:
                node.is_start = False
                node.is_end = False
                node.is_obstacle = False
                node.is_aisle = False
                node.reset()
        self.start_node = None
        self.end_node = None


        if hasattr(self, 'johnsons_graph'):
            del self.johnsons_graph
            print("Cleared cached Johnson's graph during clear_all.")
        if hasattr(self, 'last_grid_state'):
            del self.last_grid_state
            print("Cleared last grid state during clear_all.")

    def update_spacing(self):
        
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
            orientation = 'vertical'  
        self.generate_warehouse_layout(orientation)

    def generate_warehouse_layout(self, orientation='vertical', unique_filename=None):
        
        try:
            self.is_generating_warehouse = True
            self.clear_all()

            
            if hasattr(self, 'johnsons_graph'):
                del self.johnsons_graph
                print("Cleared cached Johnson's graph.")
            if hasattr(self, 'last_grid_state'):
                del self.last_grid_state
                print("Cleared last grid state.")

            
            if unique_filename:
                warehouse_data = generate_warehouse_data(
                    num_aisles=self.num_aisles,
                    max_shelves_per_aisle=self.max_shelves_per_aisle,
                    save_to_csv=True,
                    filename=unique_filename
                )
            else:
                warehouse_data = generate_warehouse_data(
                    num_aisles=self.num_aisles,
                    max_shelves_per_aisle=self.max_shelves_per_aisle,
                    save_to_csv=True
                )


            self.item_dropdown.blockSignals(True)
            self.item_dropdown.clear()
            self.item_dropdown.addItem("Select Item")

            self.item_nodes = []  

            aisles = warehouse_data['Aisle_Number'].nunique()
            aisle_positions = []

            vertical_positions = set()
            horizontal_positions = set()

            
            max_x, max_y = 12, 12  

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
                    x = 2 + i * (self.spacing + 2)  
                    vertical_positions.add(x)
                    aisle_positions.append(('vertical', x))

                for i in range(num_horizontal_aisles):
                    y = 2 + i * (self.spacing + 2)
                    
                    while y in vertical_positions or y + 1 in vertical_positions:
                        y += 1
                    horizontal_positions.add(y)
                    aisle_positions.append(('horizontal', y))

                max_x = max(max(vertical_positions, default=0), 2 + self.max_shelves_per_aisle) + 2
                max_y = max(max(horizontal_positions, default=0), 2 + self.max_shelves_per_aisle) + 2


            max_x = min(max_x, 50)  
            max_y = min(max_y, 50)

            
            self.grid_size = max(int(max_x), int(max_y)) + 2
            self.init_grid()  

            for i, row in warehouse_data.iterrows():
                try:
                    aisle_num = int(row['Aisle_Number'].split('_')[1])
                    shelf_num = int(row['Shelf_Number'].split('_')[1])
                    shelf_loc = row['Shelf_Location']
                    item = row['Item'] if row['Item'] is not None else "Empty"

                    
                    if aisle_num - 1 >= len(aisle_positions):
                        continue

                    orientation_type, pos = aisle_positions[aisle_num - 1]

                    if orientation_type == 'vertical':
                        x = pos
                        y = 2 + (shelf_num - 1)  
                        aisle_color = QColor(150, 150, 250)  
                    else:  
                        x = 2 + (shelf_num - 1)
                        y = pos
                        aisle_color = QColor(150, 0, 250)  

                    
                    if x >= self.grid_size or y >= self.grid_size:
                        print(f"Skipping node at ({x}, {y}) - out of bounds.")
                        continue  

                    node = self.grid[y][x]
                    node.set_as_aisle(aisle_color)  
                    
                    node.name = f"Aisle_{aisle_num}_Shelf_{shelf_num}_{shelf_loc}"

                    
                    if item != "Empty":
                        self.item_dropdown.addItem(
                            f"{item} (Aisle {aisle_num}, Shelf {shelf_num}, Location {shelf_loc})")

                        
                        node_info = {
                            'node': node,
                            'item': item,
                            'x': x,
                            'y': y
                        }
                        self.item_nodes.append(node_info)

                        
                        item_number = item.split('_')[-1] if item != "Empty" else "Empty"

                        
                        node.set_item_label(item_number)
                except Exception as e:
                    print(f"Error processing row {i}: {e}")

            
            self.item_dropdown.blockSignals(False)

            
            self.adjust_zoom()
            print(f"Item nodes available: {len(self.item_nodes)}")

        except Exception as e:
            print(f"Error in generate_warehouse_layout: {e}")
        finally:
            
            self.is_generating_warehouse = False

    def set_mode_start(self):
        
        self.current_mode = 'start'

    def set_mode_end(self):
        
        self.current_mode = 'end'

    def set_mode_barrier(self):
        
        self.current_mode = 'barrier'

    def set_start_node(self, node):
        
        
        if node not in [n for row in self.grid for n in row]:
            QMessageBox.warning(self, "Invalid Node", "The selected start node is no longer valid.")
            return

        if not hasattr(self, 'start_node'):
            self.start_node = None
        if self.end_node == node:
            return

        
        if self.start_node:
            self.start_node.is_start = False
            self.start_node.reset()

        
        self.start_node = node
        self.start_node.set_as_start()

    def set_end_node(self, node):
        
        
        if not hasattr(self, 'end_node'):
            self.end_node = None
        if self.start_node == node:
            return

        
        if self.end_node:
            self.end_node.is_end = False
            self.end_node.reset()

        
        self.end_node = node
        self.end_node.set_as_end()

    def set_as_start(self):
        
        if self.parent_window.end_node == self:
            return  
        if self.parent_window.start_node:
            self.parent_window.start_node.is_start = False
            self.parent_window.start_node.reset()
        self.is_start = True
        self.is_obstacle = False
        self.setBrush(QBrush(QColor(0, 255, 0)))  
        self.parent_window.start_node = self

    def set_as_end(self):
        
        if self.parent_window.start_node == self:
            return  
        if self.parent_window.end_node:
            self.parent_window.end_node.is_end = False
            self.parent_window.end_node.reset()
        self.is_end = True
        self.is_obstacle = False
        self.setBrush(QBrush(QColor(255, 0, 0)))  
        self.parent_window.end_node = self

    def set_end_node_from_dropdown(self):
        
        selected_item = self.item_dropdown.currentText()

        
        print("\nDropdown Debug Info:")
        print(f"Current dropdown text: '{selected_item}'")
        print(f"Dropdown item count: {self.item_dropdown.count()}")
        print("All dropdown items:")
        for i in range(self.item_dropdown.count()):
            print(f"  Item {i}: '{self.item_dropdown.itemText(i)}'")

        if not selected_item:
            print("Warning: Selected item is empty")
            return

        if selected_item == "Select Item":
            print("Info: Default 'Select Item' option chosen")
            return

        try:
            print(f"\nParsing Process for: '{selected_item}'")

            
            if " (Aisle " not in selected_item:
                print(f"Error: Missing expected delimiter ' (Aisle ' in string: '{selected_item}'")
                print("Expected format: 'Item_XXXX (Aisle N, Shelf M, Location L)'")
                return

            
            try:
                item_part, location_part = selected_item.split(" (Aisle ", 1)
                print(f"Split result:")
                print(f"  Item part: '{item_part}'")
                print(f"  Location part: '{location_part}'")
            except ValueError as e:
                print(f"Error splitting item and location: {e}")
                print(f"Full string being split: '{selected_item}'")
                return

            
            location_part = location_part.rstrip(')')
            print(f"Location part after removing parenthesis: '{location_part}'")

            try:
                
                aisle_shelf_loc = location_part.split(", ")
                print(f"Location components after split: {aisle_shelf_loc}")

                if len(aisle_shelf_loc) != 3:
                    print(f"Error: Expected 3 location components, got {len(aisle_shelf_loc)}")
                    print(f"Components: {aisle_shelf_loc}")
                    return

                
                try:
                    aisle_num = int(aisle_shelf_loc[0])
                    print(f"Parsed aisle number: {aisle_num}")
                except ValueError:
                    print(f"Error: Could not convert aisle number '{aisle_shelf_loc[0]}' to integer")
                    return

                
                shelf_part = aisle_shelf_loc[1].split(" ")
                print(f"Shelf part split: {shelf_part}")

                if len(shelf_part) != 2 or shelf_part[0] != "Shelf":
                    print(f"Error: Invalid shelf format: '{aisle_shelf_loc[1]}'")
                    print("Expected format: 'Shelf X' where X is a number")
                    return

                try:
                    shelf_num = int(shelf_part[1])
                    print(f"Parsed shelf number: {shelf_num}")
                except ValueError:
                    print(f"Error: Could not convert shelf number '{shelf_part[1]}' to integer")
                    return

                
                loc_part = aisle_shelf_loc[2].split(" ")
                print(f"Location part split: {loc_part}")

                if len(loc_part) != 2 or loc_part[0] != "Location":
                    print(f"Error: Invalid location format: '{aisle_shelf_loc[2]}'")
                    print("Expected format: 'Location X'")
                    return

                shelf_loc = loc_part[1]
                print(f"Parsed shelf location: '{shelf_loc}'")

            except (ValueError, IndexError) as e:
                print(f"Error parsing location components: {e}")
                print(f"Location part being parsed: '{location_part}'")
                return

            
            orientation = self.layout_dropdown.currentText()
            print(f"\nLayout Configuration:")
            print(f"Selected orientation: '{orientation}'")
            print(f"Number of aisles: {self.num_aisles}")
            print(f"Spacing: {self.spacing}")

            
            if orientation == "Vertical Aisles":
                orientation_type = 'vertical'
            elif orientation == "Horizontal Aisles":
                orientation_type = 'horizontal'
            elif orientation == "Mixed Aisles":
                orientation_type = 'mixed'
            else:
                orientation_type = 'vertical'  
                print(f"Warning: Unknown orientation '{orientation}', defaulting to vertical")

            print(f"Using orientation type: {orientation_type}")

            
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

            print(f"\nCalculated Positions:")
            print(f"Aisle positions: {aisle_positions}")

            
            if aisle_num - 1 >= len(aisle_positions):
                print(f"Error: Aisle number {aisle_num} exceeds available positions ({len(aisle_positions)})")
                return

            
            orientation_type_aisle, pos = aisle_positions[aisle_num - 1]
            print(f"Using position data: type={orientation_type_aisle}, pos={pos}")

            if orientation_type_aisle == 'vertical':
                x = pos
                y = 2 + (shelf_num - 1)
            else:
                x = 2 + (shelf_num - 1)
                y = pos

            print(f"Calculated coordinates: x={x}, y={y}")

            
            if x >= self.grid_size or y >= self.grid_size:
                print(f"Error: Calculated position ({x}, {y}) is out of bounds. Grid size is {self.grid_size}")
                return

            
            node = self.grid[y][x]
            if node:
                print(f"Found node at ({x}, {y}). Setting as end node.")
                self.set_end_node(node)
            else:
                print(f"Error: No node found at position ({x}, {y})")

        except Exception as e:
            print("\nDetailed Error Information:")
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception message: {str(e)}")
            print(f"Selected item text: '{selected_item}'")
            import traceback
            print("Full traceback:")
            traceback.print_exc()
    def has_negative_weights(self):
        
        for row in self.grid:
            for node in row:
                if node.edge_weight < 0:
                    return True
        return False

    def handle_search(self):
        
        
        has_negative = self.has_negative_weights()
        selected_algorithm = self.algorithm_dropdown.currentText()

        
        current_weights_state = {
            (x, y): self.grid[y][x].edge_weight
            for x in range(self.grid_size)
            for y in range(self.grid_size)
        }

        
        if hasattr(self, 'last_weights_state') and self.last_weights_state != current_weights_state:
            if hasattr(self, 'johnsons_graph'):
                delattr(self, 'johnsons_graph')
                print("Cleared cached Johnson's graph due to edge weight changes.")

        self.last_weights_state = current_weights_state

        
        if has_negative and not self.algorithm_capabilities[selected_algorithm]["handles_negative"]:
            compatible_algorithms = [
                name for name, caps in self.algorithm_capabilities.items()
                if caps["handles_negative"]
            ]
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

        
        self.reset_grid()

        
        start = (
            int(self.start_node.pos().x() // self.node_size),
            int(self.start_node.pos().y() // self.node_size)
        )
        end = (
            int(self.end_node.pos().x() // self.node_size),
            int(self.end_node.pos().y() // self.node_size)
        )

        
        diagonal_neighbors = self.diagonal_checkbox.isChecked()

        
        try:
            if selected_algorithm == "Dijkstra's":
                path, self.nodes_searched = self.run_dijkstra(
                    start, end, diagonal_neighbors, visualize=True
                )
            elif selected_algorithm == "Bellman-Ford":
                path, self.nodes_searched = self.run_bellman_ford(
                    start, end, diagonal_neighbors, visualize=True
                )
            elif selected_algorithm == "A* (Manhattan Distance)":
                path, self.nodes_searched = self.run_astar(
                    start, end, diagonal_neighbors, visualize=True,
                    heuristic_type="Manhattan Distance"
                )
            elif selected_algorithm == "A* (Euclidean Distance)":
                path, self.nodes_searched = self.run_astar(
                    start, end, diagonal_neighbors, visualize=True,
                    heuristic_type="Euclidean Distance"
                )
            elif selected_algorithm == "A* (Modified Euclidean 1.2x Y Priority)":
                path, self.nodes_searched = self.run_astar(
                    start, end, diagonal_neighbors, visualize=True,
                    heuristic_type="Modified Euclidean"
                )
            elif selected_algorithm == "SPFA":
                path, self.nodes_searched = self.run_spfa(
                    start, end, diagonal_neighbors, visualize=True
                )
            elif selected_algorithm == "Johnson's":
                path, (mandatory_visits, pathfinding_visits) = self.run_johnsons(
                    start, end, diagonal_neighbors, visualize=True
                )
                
                self.nodes_searched = mandatory_visits + pathfinding_visits
                
                self.counter_label.setText(
                    f"Mandatory: {mandatory_visits}, Pathfinding: {pathfinding_visits}"
                )
                return  
            elif selected_algorithm == "Johnson's with A*":
                path, (mandatory_visits, pathfinding_visits) = self.run_johnsons_astar(
                    start, end, diagonal_neighbors, visualize=True
                )
                
                self.nodes_searched = mandatory_visits + pathfinding_visits
                
                self.counter_label.setText(
                    f"Mandatory: {mandatory_visits}, Pathfinding: {pathfinding_visits}"
                )
                return  
            else:
                self.counter_label.setText("Invalid algorithm selection.")
                return

            
            if path:
                self.search_path = path
                self.visualize_path_step_by_step()

                
                total_path_length = len(path)
                self.counter_label.setText(
                    f"Nodes Searched: {self.nodes_searched}, Path Length: {total_path_length}"
                )
            else:
                self.counter_label.setText("No path found.")

        except Exception as e:
            print(f"Error in handle_search: {e}")
            self.counter_label.setText(f"Error: {str(e)}")
            QMessageBox.warning(
                self,
                "Error",
                f"An error occurred while searching for path:\n{str(e)}"
            )

    def calculate_path_length(self, path, diagonal_neighbors=False):
        
        if not path or len(path) < 2:
            return 0

        total_length = 0
        for i in range(len(path) - 1):
            current = path[i]
            next_point = path[i + 1]

            
            dx = abs(next_point[0] - current[0])
            dy = abs(next_point[1] - current[1])

            
            current_node = self.grid[current[1]][current[0]]
            next_node = self.grid[next_point[1]][next_point[0]]

            
            avg_weight = (current_node.edge_weight + next_node.edge_weight) / 2

            if diagonal_neighbors and dx == 1 and dy == 1:
                
                step_length = sqrt(2) * avg_weight
            else:
                
                step_length = (dx + dy) * avg_weight

            total_length += step_length

        return total_length

    def get_neighbors_for_reweighting(self, node, diagonal_neighbors=False):
        
        (x, y) = node

        
        four_neighbors = [
            ((-1, 0), 1.0),  
            ((1, 0), 1.0),  
            ((0, -1), 1.0),  
            ((0, 1), 1.0)  
        ]

        eight_neighbors = [
            ((-1, -1), sqrt(2)),  
            ((1, -1), sqrt(2)),  
            ((-1, 1), sqrt(2)),  
            ((1, 1), sqrt(2))  
        ]

        neighbors = []

        
        for (dx, dy), base_cost in four_neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                neighbor_node = self.grid[ny][nx]
                if not neighbor_node.is_obstacle:
                    current_node = self.grid[y][x]
                    
                    avg_weight = (current_node.edge_weight + neighbor_node.edge_weight) / 2
                    total_cost = base_cost * avg_weight
                    neighbors.append(((nx, ny), total_cost))

        
        if diagonal_neighbors:
            def is_diagonal_valid(curr_x, curr_y, new_x, new_y):
                dx = new_x - curr_x
                dy = new_y - curr_y
                if not (0 <= curr_x + dx < self.grid_size and
                        0 <= curr_y + dy < self.grid_size):
                    return False
                return (not self.grid[curr_y][curr_x + dx].is_obstacle and
                        not self.grid[curr_y + dy][curr_x].is_obstacle)

            for (dx, dy), base_cost in eight_neighbors:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    neighbor_node = self.grid[ny][nx]
                    if not neighbor_node.is_obstacle and is_diagonal_valid(x, y, nx, ny):
                        current_node = self.grid[y][x]
                        
                        avg_weight = (current_node.edge_weight + neighbor_node.edge_weight) / 2
                        total_cost = base_cost * avg_weight
                        neighbors.append(((nx, ny), total_cost))

        return neighbors

    def run_johnsons(self, start, end, diagonal_neighbors=False, visualize=True):
        
        mandatory_visits = 0
        pathfinding_visits = 0

        
        current_nodes = [(x, y) for y in range(self.grid_size) for x in range(self.grid_size)
                         if not self.grid[y][x].is_obstacle]

        
        if self.grid[start[1]][start[0]].is_obstacle or self.grid[end[1]][end[0]].is_obstacle:
            if visualize:
                self.counter_label.setStyleSheet("color: red;")
                self.counter_label.setText("Invalid start or end position")
            return None, (mandatory_visits, pathfinding_visits)

        
        if not hasattr(self, 'johnsons_graph') or not hasattr(self, 'last_grid_state'):
            try:
                if visualize:
                    self.counter_label.setStyleSheet("color: green;")
                    self.counter_label.setText("Creating reweighted graph using Bellman-Ford")
                    QApplication.processEvents()

                
                self.last_grid_state = {(x, y): self.grid[y][x].is_obstacle
                                        for x in range(self.grid_size)
                                        for y in range(self.grid_size)}

                
                virtual_node = (-1, -1)
                modified_graph = {virtual_node: []}

                
                for node in current_nodes:
                    modified_graph[node] = []
                    mandatory_visits += 1
                    if visualize and node != start and node != end:
                        self.grid[node[1]][node[0]].set_visited()
                        self.counter_label.setText(f"Building graph: {mandatory_visits} nodes processed")
                        QApplication.processEvents()

                
                for node in current_nodes:
                    modified_graph[virtual_node].append((node, 0))
                    for neighbor, weight in self.get_neighbors_for_reweighting(node, diagonal_neighbors):
                        if neighbor in current_nodes:
                            modified_graph[node].append((neighbor, weight))

                
                h_values = {node: float('inf') for node in current_nodes}
                h_values[virtual_node] = 0

                
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

                
                for u in modified_graph:
                    for v, weight in modified_graph[u]:
                        if h_values[u] != float('inf') and h_values[u] + weight < h_values[v]:
                            if visualize:
                                self.counter_label.setStyleSheet("color: red;")
                                self.counter_label.setText("Negative cycle detected - cannot proceed")
                                QApplication.processEvents()
                            return None, (mandatory_visits, pathfinding_visits)

                
                self.johnsons_graph = True  

            except Exception as e:
                if visualize:
                    self.counter_label.setStyleSheet("color: red;")
                    self.counter_label.setText(f"Error in Johnson's preprocessing: {str(e)}")
                    QApplication.processEvents()
                return None, (mandatory_visits, pathfinding_visits)

        
        if visualize:
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    if (x, y) != start and (x, y) != end:
                        self.grid[y][x].reset()

        
        distances = {(x, y): float('inf')
                     for x in range(self.grid_size)
                     for y in range(self.grid_size)}
        distances[start] = 0
        predecessors = {(x, y): None
                        for x in range(self.grid_size)
                        for y in range(self.grid_size)}
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

            
            for neighbor, weight in self.get_neighbors(current, diagonal_neighbors):
                if neighbor not in visited:
                    can_traverse = (
                            not self.grid[neighbor[1]][neighbor[0]].is_aisle  
                            or neighbor == end  
                            or current == start  
                    )

                    if can_traverse:
                        distance = distances[current] + weight
                        if distance < distances[neighbor]:
                            distances[neighbor] = distance
                            predecessors[neighbor] = current
                            heapq.heappush(pq, (distance, neighbor))

        
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
        
        mandatory_visits = 0
        pathfinding_visits = 0

        
        current_nodes = [(x, y) for y in range(self.grid_size) for x in range(self.grid_size)
                         if not self.grid[y][x].is_obstacle]

        
        if self.grid[start[1]][start[0]].is_obstacle or self.grid[end[1]][end[0]].is_obstacle:
            if visualize:
                self.counter_label.setStyleSheet("color: red;")
                self.counter_label.setText("Invalid start or end position")
            return None, (mandatory_visits, pathfinding_visits)

        
        if not hasattr(self, 'johnsons_graph') or not hasattr(self, 'last_grid_state'):
            try:
                if visualize:
                    self.counter_label.setStyleSheet("color: green;")
                    self.counter_label.setText("Creating reweighted graph using Bellman-Ford")
                    QApplication.processEvents()

                
                self.last_grid_state = {(x, y): self.grid[y][x].is_obstacle
                                        for x in range(self.grid_size)
                                        for y in range(self.grid_size)}

                
                virtual_node = (-1, -1)
                modified_graph = {virtual_node: []}

                
                for node in current_nodes:
                    modified_graph[node] = []
                    mandatory_visits += 1
                    if visualize and node != start and node != end:
                        self.grid[node[1]][node[0]].set_visited()
                        self.counter_label.setText(f"Building graph: {mandatory_visits} nodes processed")
                        QApplication.processEvents()

                
                for node in current_nodes:
                    modified_graph[virtual_node].append((node, 0))
                    for neighbor, weight in self.get_neighbors_for_reweighting(node, diagonal_neighbors):
                        if neighbor in current_nodes:
                            modified_graph[node].append((neighbor, weight))

                
                h_values = {node: float('inf') for node in current_nodes}
                h_values[virtual_node] = 0

                
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

                
                for u in modified_graph:
                    for v, weight in modified_graph[u]:
                        if h_values[u] != float('inf') and h_values[u] + weight < h_values[v]:
                            if visualize:
                                self.counter_label.setStyleSheet("color: red;")
                                self.counter_label.setText("Negative cycle detected - cannot proceed")
                                QApplication.processEvents()
                            return None, (mandatory_visits, pathfinding_visits)

                
                self.johnsons_graph = True

            except Exception as e:
                if visualize:
                    self.counter_label.setStyleSheet("color: red;")
                    self.counter_label.setText(f"Error in Johnson's preprocessing: {str(e)}")
                    QApplication.processEvents()
                return None, (mandatory_visits, pathfinding_visits)

        
        if visualize:
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    if (x, y) != start and (x, y) != end:
                        self.grid[y][x].reset()

        
        g_score = {(x, y): float('inf')
                   for x in range(self.grid_size)
                   for y in range(self.grid_size)}
        g_score[start] = 0
        f_score = {(x, y): float('inf')
                   for x in range(self.grid_size)
                   for y in range(self.grid_size)}
        f_score[start] = self.heuristic(start, end, "Manhattan Distance")

        open_set = [(f_score[start], start)]
        predecessors = {(x, y): None
                        for x in range(self.grid_size)
                        for y in range(self.grid_size)}
        visited = set()

        while open_set:
            current_f, current = heapq.heappop(open_set)

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

            
            for neighbor, weight in self.get_neighbors(current, diagonal_neighbors):
                if neighbor not in visited:
                    can_traverse = (
                            not self.grid[neighbor[1]][neighbor[0]].is_aisle  
                            or neighbor == end  
                            or current == start  
                    )

                    if can_traverse:
                        tentative_g_score = g_score[current] + weight
                        if tentative_g_score < g_score[neighbor]:
                            predecessors[neighbor] = current
                            g_score[neighbor] = tentative_g_score
                            f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, end, "Manhattan Distance")
                            heapq.heappush(open_set, (f_score[neighbor], neighbor))

        
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
        
        distances = {node: float('inf') for node in graph}
        distances[source] = 0

        
        for _ in range(len(graph) - 1):
            for u in graph:
                for v, weight in graph[u]:
                    if distances[u] + weight < distances[v]:
                        distances[v] = distances[u] + weight

        
        for u in graph:
            for v, weight in graph[u]:
                if distances[u] + weight < distances[v]:
                    return None  

        return distances

    def dijkstra_for_johnsons(self, graph, source):
        
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

    def run_astar(self, start, end, diagonal_neighbors=False, visualize=True, heuristic_type="Manhattan Distance"):
        
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}

        
        h = self.heuristic(start, end, heuristic_type)
        f_score = {start: h}
        self.nodes_searched = 0

        while open_set:
            current_f, current = heapq.heappop(open_set)

            if current == end:
                path = self.reconstruct_path(came_from, current)
                return path, self.nodes_searched

            if visualize:
                self.grid[current[1]][current[0]].set_visited()
                self.counter_label.setText(f"Nodes Searched: {self.nodes_searched}")
                QApplication.processEvents()

            self.nodes_searched += 1

            for neighbor, weight in self.get_neighbors(current, diagonal_neighbors):
                tentative_g_score = g_score[current] + weight

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, end, heuristic_type)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None, self.nodes_searched

    def run_spfa(self, start, end, diagonal_neighbors=False, visualize=True):
        
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

            if visualize:
                self.grid[current[1]][current[0]].set_visited()
                self.counter_label.setText(f"Nodes Searched: {self.nodes_searched}")
                QApplication.processEvents()

            self.nodes_searched += 1

            for neighbor_coords, weight in self.get_neighbors(current, diagonal_neighbors):
                if distance[current] + weight < distance[neighbor_coords]:
                    distance[neighbor_coords] = distance[current] + weight
                    predecessor[neighbor_coords] = current

                    if not in_queue[neighbor_coords]:
                        queue.append(neighbor_coords)
                        in_queue[neighbor_coords] = True

            if current == end:
                break

        
        if distance[end] == float('inf'):
            return None, self.nodes_searched

        path = []
        current = end
        while current is not None:
            path.append(current)
            current = predecessor[current]
        path.reverse()

        return path, self.nodes_searched
    def run_bellman_ford(self, start, end, diagonal_neighbors=False, visualize=True):
        
        nodes = [(x, y) for y in range(self.grid_size) for x in range(self.grid_size)]
        edges = []

        
        distance = {node: float('inf') for node in nodes}
        distance[start] = 0
        predecessor = {node: None for node in nodes}

        
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                current_node = (x, y)
                for neighbor_coords, weight in self.get_neighbors(current_node, diagonal_neighbors):
                    edges.append((current_node, neighbor_coords, weight))

        self.nodes_searched = 0
        visited_nodes = set()

        
        for _ in range(len(nodes) - 1):
            for u, v, weight in edges:
                if distance[u] + weight < distance[v]:
                    distance[v] = distance[u] + weight
                    predecessor[v] = u

                    if v not in visited_nodes:
                        visited_nodes.add(v)
                        self.nodes_searched += 1

                    if visualize:
                        self.grid[v[1]][v[0]].set_visited()
                        self.counter_label.setText(f"Nodes Searched: {self.nodes_searched}")
                        QApplication.processEvents()

        
        for u, v, weight in edges:
            if distance[u] + weight < distance[v]:
                print("Graph contains a negative-weight cycle")
                return None, self.nodes_searched

        
        if distance[end] == float('inf'):
            return None, self.nodes_searched

        path = []
        current = end
        while current is not None:
            path.append(current)
            current = predecessor[current]
        path.reverse()

        return path if path[0] == start else None, self.nodes_searched
    def run_dijkstra(self, start, end, diagonal_neighbors=False, visualize=True):
        
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
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

            for neighbor, weight in self.get_neighbors(current, diagonal_neighbors):
                tentative_g_score = g_score[current] + weight

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    heapq.heappush(open_set, (g_score[neighbor], neighbor))

        return None, self.nodes_searched
    def is_valid_position(self, x, y):
        
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def is_obstacle_func(self, x, y):
        
        if not self.is_valid_position(x, y):
            return True
        
        if self.grid[y][x].is_aisle and self.grid[y][x] != self.end_node:
            return True
        return self.grid[y][x].is_obstacle

    def is_traversable(self, x, y):
        
        if not self.is_valid_position(x, y):
            return False
        node = self.grid[y][x]

        
        if node.is_aisle and node != self.end_node:
            return False

        return not node.is_obstacle

    def distance(self, node_a, node_b):
        
        (x1, y1) = node_a
        (x2, y2) = node_b
        return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def heuristic(self, node, goal, heuristic_type="Manhattan"):
        
        (x1, y1) = node
        (x2, y2) = goal

        if heuristic_type == "Manhattan Distance":
            return abs(x1 - x2) + abs(y1 - y2)
        elif heuristic_type == "Euclidean Distance":
            return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        elif heuristic_type == "Modified Euclidean":
            return sqrt((x1 - x2) ** 2 + (1.2 * (y1 - y2)) ** 2)
        else:
            
            return abs(x1 - x2) + abs(y1 - y2)

    def get_neighbors(self, node, diagonal_neighbors=False):
        
        (x, y) = node

        
        four_neighbors = [
            ((-1, 0), 1.0),  
            ((1, 0), 1.0),  
            ((0, -1), 1.0),  
            ((0, 1), 1.0)  
        ]

        eight_neighbors = [
            ((-1, -1), sqrt(2)),  
            ((1, -1), sqrt(2)),  
            ((-1, 1), sqrt(2)),  
            ((1, 1), sqrt(2))  
        ]

        neighbors = []

        
        for (dx, dy), base_cost in four_neighbors:
            nx, ny = x + dx, y + dy
            if self.is_valid_position(nx, ny) and self.is_traversable(nx, ny):
                neighbor_coords = (nx, ny)
                neighbor_node = self.grid[ny][nx]
                current_node = self.grid[y][x]
                avg_weight = (current_node.edge_weight + neighbor_node.edge_weight) / 2
                total_cost = base_cost * avg_weight
                neighbors.append((neighbor_coords, total_cost))

        
        if diagonal_neighbors:
            def is_diagonal_valid(curr_x, curr_y, new_x, new_y):
                dx = new_x - curr_x
                dy = new_y - curr_y
                return (self.is_traversable(curr_x + dx, curr_y) and
                        self.is_traversable(curr_x, curr_y + dy))

            for (dx, dy), base_cost in eight_neighbors:
                nx, ny = x + dx, y + dy
                if self.is_valid_position(nx, ny) and self.is_traversable(nx, ny):
                    if is_diagonal_valid(x, y, nx, ny):
                        neighbor_coords = (nx, ny)
                        neighbor_node = self.grid[ny][nx]
                        current_node = self.grid[y][x]
                        avg_weight = (current_node.edge_weight + neighbor_node.edge_weight) / 2
                        total_cost = base_cost * avg_weight
                        neighbors.append((neighbor_coords, total_cost))

        return neighbors

    def reconstruct_path(self, came_from, current):
        
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(current)
        path.reverse()
        return path

    def visualize_next_path(self):
        
        if self.current_target_index >= len(self.all_target_nodes_copy):
            self.all_paths_timer.stop()
            QMessageBox.information(self, "Show All Paths", "Completed visualizing all paths.")
            self.set_ui_enabled(True)
            print("Completed visualizing all paths")
            return

        
        node = self.all_target_nodes_copy[self.current_target_index]
        print(f"Visualizing path {self.current_target_index + 1} of {len(self.all_target_nodes_copy)}")
        print(f"Visualizing path to node: {node.name}")

        
        self.end_node = node

        
        self.reset_grid()
        print("Grid reset before visualizing the current path")

        
        start = (int(self.start_node.pos().x() // self.node_size),
                 int(self.start_node.pos().y() // self.node_size))
        end = (int(self.end_node.pos().x() // self.node_size),
               int(self.end_node.pos().y() // self.node_size))
        print(f"Start coordinates: {start}, End coordinates: {end}")

        
        selected_algorithm = self.algorithm_dropdown.currentText()
        print(f"Running algorithm: {selected_algorithm}")

        try:
            
            if selected_algorithm == "Dijkstra's":
                path, _ = self.run_dijkstra(start, end, diagonal_neighbors=False, visualize=False)
            elif selected_algorithm == "Bellman-Ford":
                path, _ = self.run_bellman_ford(start, end, diagonal_neighbors=False, visualize=False)
            elif selected_algorithm.startswith("A*"):
                
                if selected_algorithm == "A* (Manhattan Distance)":
                    heuristic_type = "Manhattan"
                elif selected_algorithm == "A* (Euclidean Distance)":
                    heuristic_type = "Euclidean"
                elif selected_algorithm == "A* (Modified Euclidean 1.2x Y Priority)":
                    heuristic_type = "Modified Euclidean"
                else:
                    heuristic_type = "Manhattan"  

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

            
            if path:
                for node_coords in path:
                    if node_coords != start and node_coords != end:
                        self.grid[node_coords[1]][node_coords[0]].set_path()
                print(f"Path to {node.name} visualized successfully.")
            else:
                print(f"No path found to {node.name}.")

            
            self.counter_label.setText(
                f"Visualized paths: {self.current_target_index + 1}/{len(self.all_target_nodes_copy)}")
            QApplication.processEvents()  

        except Exception as e:
            print(f"Exception during path visualization: {e}")
            QMessageBox.warning(self, "Error", f"An error occurred while visualizing the path:\n{e}")
            self.all_paths_timer.stop()
            self.set_ui_enabled(True)
            return

        
        self.current_target_index += 1

    def visualize_path_step_by_step(self):
        
        self.step_index = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_step)
        self.timer.start(10)  

        
        total_path_length = len(self.search_path)

        
        if self.algorithm_dropdown.currentText() == "Johnson's":
            
            current_text = self.counter_label.text()
            self.counter_label.setText(f"{current_text}, Path Length: {total_path_length}")
        else:
            
            self.counter_label.setText(f"Nodes Searched: {self.nodes_searched}, Path Length: {total_path_length}")

    def update_step(self):
        
        if self.step_index < len(self.search_path):
            x, y = self.search_path[self.step_index]
            node = self.grid[y][x]
            if not node.is_aisle and not node.is_start and not node.is_end:
                node.set_path()
            self.step_index += 1
        else:
            self.timer.stop()  

    def save_scenario(self):
        
        
        scenario_name, ok = QInputDialog.getText(self, "Save Scenario", "Enter scenario name:")
        if ok and scenario_name:
            
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
                        'color': node.brush().color().name(),
                        'name': node.name if hasattr(node, 'name') else None,
                        'item_label': node.item_label.toPlainText() if hasattr(node, 'item_label') else None
                    }
                    scenario_data['nodes'].append(node_data)

            
            json_file = os.path.join(self.scenario_dir, f"{scenario_name}.json")
            with open(json_file, 'w') as f:
                json.dump(scenario_data, f)

            
            if hasattr(self, 'item_nodes'):
                warehouse_data = []
                for node_info in self.item_nodes:
                    node = node_info['node']
                    aisle_shelf_loc = node.name.split('_')  
                    if len(aisle_shelf_loc) >= 5:
                        warehouse_data.append({
                            'Aisle_Number': f"{aisle_shelf_loc[0]}_{aisle_shelf_loc[1]}",
                            'Shelf_Number': f"{aisle_shelf_loc[2]}_{aisle_shelf_loc[3]}",
                            'Shelf_Location': aisle_shelf_loc[4],
                            'Item': node_info['item'],
                            'Quantity': random.randint(1, 100),  
                            'Is_Shelf_Empty': False
                        })

                
                csv_file = os.path.join(self.scenario_dir, f"{scenario_name}_warehouse.csv")
                df_warehouse = pd.DataFrame(warehouse_data)
                df_warehouse.to_csv(csv_file, index=False)

            
            self.load_scenarios()
            QMessageBox.information(self, "Scenario Saved", f"Scenario '{scenario_name}' saved successfully.")

    def get_node_position(self, node):
        
        if node:
            x = int(node.pos().x() // self.node_size)
            y = int(node.pos().y() // self.node_size)
            return {'x': x, 'y': y}
        return None

    def load_scenario(self):
        
        scenario_name = self.load_dropdown.currentText()
        if scenario_name == "Select Scenario":
            return

        json_file = os.path.join(self.scenario_dir, f"{scenario_name}.json")
        csv_file = os.path.join(self.scenario_dir, f"{scenario_name}_warehouse.csv")

        if not os.path.exists(json_file):
            QMessageBox.warning(self, "Error", f"Scenario file '{scenario_name}' not found.")
            return

        
        with open(json_file, 'r') as f:
            scenario_data = json.load(f)

        
        self.grid_size = scenario_data['grid_size']
        self.spacing = scenario_data['spacing']
        self.num_aisles = scenario_data['num_aisles']
        self.max_shelves_per_aisle = scenario_data['max_shelves_per_aisle']

        
        self.spacing_dropdown.setCurrentText(str(self.spacing))
        self.aisle_spinbox.setValue(self.num_aisles)
        self.shelf_spinbox.setValue(self.max_shelves_per_aisle)
        self.layout_dropdown.setCurrentText(scenario_data['layout_type'])

        
        self.item_dropdown.clear()
        self.item_dropdown.addItem("Select Item")
        self.item_nodes = []

        
        self.init_grid()

        
        for node_data in scenario_data['nodes']:
            x = node_data['x']
            y = node_data['y']
            node = self.grid[y][x]
            node.is_obstacle = node_data['is_obstacle']
            node.is_aisle = node_data['is_aisle']
            node.name = node_data.get('name', '')
            color = QColor(node_data['color'])
            node.setBrush(QBrush(color))

            
            if node_data.get('item_label'):
                node.set_item_label(node_data['item_label'])

        
        if os.path.exists(csv_file):
            warehouse_df = pd.read_csv(csv_file)
            for _, row in warehouse_df.iterrows():
                item = row['Item']
                if item and not pd.isna(item):
                    aisle_num = int(row['Aisle_Number'].split('_')[1])
                    shelf_num = int(row['Shelf_Number'].split('_')[1])
                    shelf_loc = row['Shelf_Location']

                    
                    self.item_dropdown.addItem(
                        f"{item} (Aisle {aisle_num}, Shelf {shelf_num}, Location {shelf_loc})"
                    )

                    
                    for node in [n for row in self.grid for n in row]:
                        if (node.name == f"Aisle_{aisle_num}_Shelf_{shelf_num}_{shelf_loc}"):
                            node_info = {
                                'node': node,
                                'item': item,
                                'x': int(node.pos().x() // self.node_size),
                                'y': int(node.pos().y() // self.node_size)
                            }
                            self.item_nodes.append(node_info)
                            break

        
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

        self.adjust_zoom()
        QMessageBox.information(self, "Scenario Loaded", f"Scenario '{scenario_name}' loaded successfully.")

    def load_scenarios(self):
        
        self.load_dropdown.blockSignals(True)  

        self.load_dropdown.clear()
        self.load_dropdown.addItem("Select Scenario")  

        scenario_files = [f for f in os.listdir(self.scenario_dir) if f.endswith('.json')]
        scenario_names = [os.path.splitext(f)[0] for f in scenario_files]

        self.load_dropdown.addItems(scenario_names)

        self.load_dropdown.blockSignals(False)  

    def process_benchmark_data(self, benchmark_data):
        
        
        accumulated_metrics = {}

        for run in benchmark_data['runs']:  
            algorithm = run['algorithm']  
            if algorithm not in accumulated_metrics:
                accumulated_metrics[algorithm] = {
                    'total_path_length': 0,
                    'total_nodes_searched': 0,
                    'total_time_taken': 0,
                    'valid_runs': 0
                }
            
            if run['avg_path_length'] is not None:
                accumulated_metrics[algorithm]['total_path_length'] += run['avg_path_length']
                accumulated_metrics[algorithm]['total_nodes_searched'] += run['avg_nodes_searched']
                accumulated_metrics[algorithm]['total_time_taken'] += run['avg_time_taken']
                accumulated_metrics[algorithm]['valid_runs'] += 1

        
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
        
        results_str = f"Benchmark Results ({movement_type}):\n"

        for algorithm, metrics in averaged_metrics.items():
            results_str += f"\nAlgorithm: {algorithm}\n"
            if metrics['avg_path_length'] is not None:
                results_str += f"  - Average Path Length: {metrics['avg_path_length']:.2f}\n"
                results_str += f"  - Average Nodes Searched: {metrics['avg_nodes_searched']:.2f}\n"
                results_str += f"  - Average Time Taken: {metrics['avg_time_taken']:.4f} seconds\n"

                
                if 'avg_mandatory_visits' in metrics and metrics['avg_mandatory_visits'] is not None:
                    results_str += f"  - Average Mandatory Visits: {metrics['avg_mandatory_visits']:.2f}\n"
                if 'avg_pathfinding_visits' in metrics and metrics['avg_pathfinding_visits'] is not None:
                    results_str += f"  - Average Pathfinding Visits: {metrics['avg_pathfinding_visits']:.2f}\n"
            else:
                results_str += "  - No valid paths found.\n"

        
        QMessageBox.information(self, f"Benchmark Results - {movement_type}", results_str)
        print(results_str)  

    def save_benchmark_results(self, metrics_per_algorithm):
        
        
        filename = os.path.join(self.scenario_dir, "benchmark_results.json")
        with open(filename, 'w') as f:
            json.dump(metrics_per_algorithm, f, indent=4)

        QMessageBox.information(self, "Benchmark Saved", f"Benchmark results saved to {filename}")

    def plot_benchmark_results(self, averaged_metrics, num_runs, movement_type):
        
        
        plots_dir = os.path.join(self.scenario_dir, "benchmark_plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        graph_size = f"{self.grid_size}x{self.grid_size}"
        layout_type = self.layout_dropdown.currentText().replace(" ", "_")

        
        algorithms = list(averaged_metrics.keys())
        avg_path_length = []
        avg_nodes_searched = []
        avg_time_taken = []

        for alg in algorithms:
            metrics = averaged_metrics[alg]
            avg_path_length.append(metrics['avg_path_length'] if metrics['avg_path_length'] is not None else 0)
            avg_nodes_searched.append(metrics['avg_nodes_searched'] if metrics['avg_nodes_searched'] is not None else 0)
            avg_time_taken.append(metrics['avg_time_taken'] if metrics['avg_time_taken'] is not None else 0)

        
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

        
        for config in plot_configs:
            plt.figure(figsize=(12, 6))
            bars = plt.bar(algorithms, config['data'], color=config['color'])
            plt.xlabel('Algorithm', fontsize=12)
            plt.ylabel(config['ylabel'], fontsize=12)
            plt.title(config['title'], fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(fontsize=10)
            plt.tight_layout()

            
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

            
            plot_path = os.path.join(plots_dir, config['filename'])
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

        
        QMessageBox.information(
            self,
            "Benchmark Plots Saved",
            f"Benchmark plots for {movement_type} have been saved in:\n{plots_dir}"
        )
        print(f"Benchmark plots for {movement_type} saved in {plots_dir}")

    def run_random_benchmarks(self):
        
        
        num_runs = self.benchmark_spinbox.value()
        if num_runs < 1:
            QMessageBox.warning(self, "Invalid Number of Runs", "Please select at least one benchmark run.")
            return

        has_negative = self.has_negative_weights()

        
        if has_negative:
            algorithms_to_test = [algo for algo, caps in self.algorithm_capabilities.items()
                                  if caps["handles_negative"]]
            if not algorithms_to_test:
                QMessageBox.warning(
                    self,
                    "No Compatible Algorithms",
                    "No algorithms available that can handle negative weights."
                )
                return
            QMessageBox.information(
                self,
                "Negative Weights Detected",
                f"Only running algorithms that support negative weights:\n{', '.join(algorithms_to_test)}"
            )
        else:
            algorithms_to_test = list(self.algorithm_capabilities.keys())

        
        if self.all_nodes_checkbox.isChecked():
            target_nodes = [
                node for row in self.grid for node in row
                if not node.is_obstacle and node != self.start_node
            ]
        else:
            target_nodes = [node_info['node'] for node_info in self.item_nodes]

        if not target_nodes:
            QMessageBox.warning(self, "Error", "No target nodes available for benchmarking.")
            return

        use_fixed_start = self.use_start_as_benchmark_start_checkbox.isChecked()

        
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

        
        self.set_ui_enabled(False)

        
        for run in range(1, num_runs + 1):
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            warehouse_filename = os.path.join(
                self.scenario_dir,
                f"benchmark_run_{run}_{timestamp}_warehouse.csv"
            )

            
            selected_layout = self.layout_dropdown.currentText()
            if selected_layout == "Vertical Aisles":
                orientation = 'vertical'
            elif selected_layout == "Horizontal Aisles":
                orientation = 'horizontal'
            elif selected_layout == "Mixed Aisles":
                orientation = 'mixed'
            else:
                orientation = 'vertical'  

            
            self.generate_warehouse_layout(orientation=orientation, unique_filename=warehouse_filename)

            
            if use_fixed_start:
                if not self.start_node:
                    QMessageBox.warning(self, "Error", "Start node is not set.")
                    self.set_ui_enabled(True)
                    return
                start_node = self.start_node
            else:
                traversable_nodes = [
                    node for row in self.grid for node in row
                    if not node.is_obstacle and not node.is_aisle
                ]
                if not traversable_nodes:
                    QMessageBox.warning(self, "Error", "No traversable nodes available.")
                    self.set_ui_enabled(True)
                    return
                start_node = random.choice(traversable_nodes)

            
            if self.all_nodes_checkbox.isChecked():
                
                current_target_nodes = [
                    node for row in self.grid for node in row
                    if not node.is_obstacle and node != start_node
                ]
            else:
                
                current_target_nodes = [node_info['node'] for node_info in self.item_nodes]

            if not current_target_nodes:
                QMessageBox.warning(self, "Error", "No target nodes available for benchmarking.")
                self.set_ui_enabled(True)
                return

            for movement_type in ['orthogonal', 'diagonal']:
                diagonal_enabled = (movement_type == 'diagonal')
                movement_mode = 'Orthogonal Movement' if not diagonal_enabled else 'Diagonal Movement'

                
                run_metrics = {}

                
                for end_node in current_target_nodes:
                    
                    self.set_start_node(start_node)
                    self.set_end_node(end_node)

                    self.counter_label.setText(
                        f"Benchmark Run {run}/{num_runs} - {movement_mode}"
                    )
                    QApplication.processEvents()

                    
                    single_run_metrics = self.benchmark_single_run(start_node, end_node, diagonal_enabled,
                                                                   algorithms_to_test)

                    
                    for algorithm, metrics in single_run_metrics.items():
                        if algorithm not in run_metrics:
                            run_metrics[algorithm] = {
                                'path_length': 0,
                                'nodes_searched': 0,
                                'time_taken': 0,
                                'valid_runs': 0
                            }
                        if metrics['path_length'] is not None:
                            run_metrics[algorithm]['path_length'] += metrics['path_length']
                            run_metrics[algorithm]['nodes_searched'] += metrics['nodes_searched']
                            run_metrics[algorithm]['time_taken'] += metrics['time_taken']
                            run_metrics[algorithm]['valid_runs'] += 1

                
                for algorithm, metrics in run_metrics.items():
                    if metrics['valid_runs'] > 0:
                        avg_path_length = metrics['path_length'] / metrics['valid_runs']
                        avg_nodes_searched = metrics['nodes_searched'] / metrics['valid_runs']
                        avg_time_taken = metrics['time_taken'] / metrics['valid_runs']
                    else:
                        avg_path_length = None
                        avg_nodes_searched = None
                        avg_time_taken = None

                    benchmark_data[movement_type]['runs'].append({
                        'run_number': run,
                        'algorithm': algorithm,
                        'avg_path_length': avg_path_length,
                        'avg_nodes_searched': avg_nodes_searched,
                        'avg_time_taken': avg_time_taken
                    })

        
        output_filename = f"random_benchmarks_{num_runs}_runs_{benchmark_data['orthogonal']['timestamp']}.json"
        result_dir = os.path.join(self.scenario_dir, "result_plots")
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        output_filepath = os.path.join(result_dir, output_filename)

        try:
            with open(output_filepath, 'w') as f:
                json.dump(benchmark_data, f, indent=4)
            QMessageBox.information(self, "Benchmark Saved", f"Results saved to {output_filepath}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save benchmark results:\n{e}")

        
        for movement_type in ['orthogonal', 'diagonal']:
            averaged_metrics = self.process_benchmark_data(benchmark_data[movement_type])
            self.display_benchmark_results(averaged_metrics, movement_type)
            self.plot_benchmark_results(
                averaged_metrics,
                num_runs,
                f"{movement_type.capitalize()} Movement"
            )

        
        self.set_ui_enabled(True)

        QMessageBox.information(
            self,
            "Benchmarks Completed",
            f"Completed {num_runs} runs with both orthogonal and diagonal movement."
        )

    def benchmark_single_run(self, start_node, end_node, diagonal_enabled=False, algorithms_to_test=None):
        
        self.reset_grid()

        if not start_node or not end_node:
            print("Invalid start or end node.")
            return {}

        start_coords = (
            int(start_node.pos().x() // self.node_size),
            int(start_node.pos().y() // self.node_size)
        )
        end_coords = (
            int(end_node.pos().x() // self.node_size),
            int(end_node.pos().y() // self.node_size)
        )

        metrics_per_algorithm = {}

        for algorithm in algorithms_to_test:
            start_time = time.time()

            try:
                if algorithm == "Dijkstra's":
                    path, nodes_searched = self.run_dijkstra(start_coords, end_coords, diagonal_enabled,
                                                             visualize=False)
                elif algorithm.startswith("A*"):
                    heuristic_type = algorithm.split("(")[1].split(")")[0].strip()
                    path, nodes_searched = self.run_astar(start_coords, end_coords, diagonal_enabled, visualize=False,
                                                          heuristic_type=heuristic_type)
                elif algorithm == "Bellman-Ford":
                    path, nodes_searched = self.run_bellman_ford(start_coords, end_coords, diagonal_enabled,
                                                                 visualize=False)
                elif algorithm == "SPFA":
                    path, nodes_searched = self.run_spfa(start_coords, end_coords, diagonal_enabled, visualize=False)
                elif algorithm == "Johnson's":
                    path, nodes_searched = self.run_johnsons(start_coords, end_coords, diagonal_enabled,
                                                             visualize=False)
                elif algorithm == "Johnson's with A*":
                    path, nodes_searched = self.run_johnsons_astar(start_coords, end_coords, diagonal_enabled,
                                                                   visualize=False)
                else:
                    print(f"Unknown algorithm: {algorithm}")
                    continue

                time_taken = time.time() - start_time


                if isinstance(nodes_searched, tuple):
                    nodes_searched = sum(nodes_searched)

                if path:
                    path_length = self.calculate_path_length(path, diagonal_enabled)
                    metrics_per_algorithm[algorithm] = {
                        'path_length': path_length,
                        'nodes_searched': nodes_searched,
                        'time_taken': time_taken
                    }
                else:
                    metrics_per_algorithm[algorithm] = {
                        'path_length': None,
                        'nodes_searched': nodes_searched,
                        'time_taken': time_taken
                    }

            except Exception as e:
                print(f"Error running algorithm {algorithm}: {e}")
                metrics_per_algorithm[algorithm] = {
                    'path_length': None,
                    'nodes_searched': None,
                    'time_taken': None,
                    'error': str(e)
                }

        return metrics_per_algorithm

    def display_benchmark_results(self, averaged_metrics, movement_type=""):
        
        results_str = f"Benchmark Results ({movement_type}):\n"

        for algorithm, metrics in averaged_metrics.items():
            results_str += f"\nAlgorithm: {algorithm}\n"
            if metrics['avg_path_length'] is not None:
                results_str += f"  - Average Path Length: {metrics['avg_path_length']:.2f}\n"
                results_str += f"  - Average Nodes Searched: {metrics['avg_nodes_searched']:.2f}\n"
                results_str += f"  - Average Time Taken: {metrics['avg_time_taken']:.4f} seconds\n"


                if 'avg_mandatory_visits' in metrics and metrics['avg_mandatory_visits'] is not None:
                    results_str += f"  - Average Mandatory Visits: {metrics['avg_mandatory_visits']:.2f}\n"
                if 'avg_pathfinding_visits' in metrics and metrics['avg_pathfinding_visits'] is not None:
                    results_str += f"  - Average Pathfinding Visits: {metrics['avg_pathfinding_visits']:.2f}\n"
            else:
                results_str += "  - No valid paths found.\n"


        QMessageBox.information(self, f"Benchmark Results - {movement_type}", results_str)
        print(results_str)

    def handle_show_all_paths(self):
        print("handle_show_all_paths called")
        if not self.start_node:
            print("Start node not set")
            QMessageBox.warning(self, "Error", "Start node not set.")
            return

        if self.all_nodes_checkbox.isChecked():
            self.all_target_nodes = [
                node for row in self.grid for node in row
                if not node.is_obstacle and node != self.start_node
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


        self.set_ui_enabled(False)
        print("UI elements disabled for Show All Paths operation")

        
        self.all_target_nodes_copy = self.all_target_nodes.copy()
        self.current_target_index = 0
        print(f"Starting visualization of {len(self.all_target_nodes_copy)} paths")

        
        self.all_paths_timer = QTimer(self)
        self.all_paths_timer.timeout.connect(self.visualize_next_path)
        self.all_paths_timer.start(10)  

        
        self.reset_grid()
        print("Grid reset for path visualization")


    def set_ui_enabled(self, enabled):
        
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
        self.show_all_paths_button.setEnabled(enabled)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WarehouseVisualizer()
    sys.exit(app.exec())
