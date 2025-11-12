import napari
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pathlib import Path
import cv2
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QCheckBox,
    QSpinBox,
    QScrollArea,
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QKeyEvent


class LazyVideoLoader:
    """Lazy video loader using OpenCV - only opens file, doesn't load frames."""
    def __init__(self, video_path):
        self.video_path = str(video_path)
        self._cap = None
        self._shape = None
        self._dtype = None
        self._num_frames = None
        
    def _ensure_open(self):
        """Open video file and get metadata (doesn't load frames)."""
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.video_path)
            if not self._cap.isOpened():
                raise ValueError(f"Could not open video: {self.video_path}")
            
            # Get properties without loading frames
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._num_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            
            # Read ONE frame to determine shape/dtype, then reset
            ret, sample_frame = self._cap.read()
            if not ret:
                raise ValueError("Could not read sample frame")
            
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
            
            # Determine shape
            if len(sample_frame.shape) == 3:
                # Color: (H, W, C)
                self._shape = (self._num_frames, height, width, sample_frame.shape[2])
            else:
                # Grayscale: (H, W)
                self._shape = (self._num_frames, height, width)
            
            self._dtype = sample_frame.dtype
            del sample_frame  # Free memory immediately
            
    @property
    def shape(self):
        self._ensure_open()
        return self._shape
    
    @property
    def dtype(self):
        self._ensure_open()
        return self._dtype
    
    def __len__(self):
        self._ensure_open()
        return self._num_frames
    
    def get_frame(self, frame_idx):
        """Load a single frame by index."""
        self._ensure_open()
        if frame_idx < 0 or frame_idx >= self._num_frames:
            raise IndexError(f"Frame {frame_idx} out of range [0, {self._num_frames})")
        
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._cap.read()
        if not ret:
            raise ValueError(f"Could not read frame {frame_idx}")
        return frame
    
    def close(self):
        """Close video file."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
    
    def __del__(self):
        self.close()


class VideoLoaderWidget(QWidget):
    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.video_loader = None
        self.video_layer = None
        self.video_path = None
        self.frame_slider = None
        self.frame_label = None
        self.frame_changed_callback = None
        self.current_frame_idx = -1
        self.num_frames = 0
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        self.load_button = QPushButton("Load Video...")
        self.load_button.clicked.connect(self.load_video)
        layout.addWidget(self.load_button)
        
        self.info_label = QLabel("Drag & drop video file or click button to load")
        layout.addWidget(self.info_label)
        
        # Enable keyboard focus for arrow key navigation
        self.setFocusPolicy(Qt.StrongFocus)
        
    def load_video(self):
        """Load video using file dialog."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video file",
            "",
            "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*.*)",
        )
        if not path:
            return
        self.load_video_file(path)
    
    def load_video_file(self, path):
        """Load video with true lazy loading - only current frame in memory."""
        path_str = str(path)
        
        if not Path(path_str).exists():
            self.info_label.setText(f"File not found: {Path(path_str).name}")
            return
        
        self.info_label.setText("Opening video...")
        print(f"Opening video (lazy): {path_str}")
        
        try:
            # Close previous video
            if self.video_loader is not None:
                self.video_loader.close()
            
            # Create lazy loader (opens file, reads metadata, but NO frames loaded)
            self.video_loader = LazyVideoLoader(path_str)
            self.video_path = path_str
            
            # Get properties (fast, no frame loading)
            video_shape = self.video_loader.shape
            video_dtype = self.video_loader.dtype
            num_frames = len(self.video_loader)
            
            print(f"Video: {video_shape}, dtype={video_dtype}, frames={num_frames}")
            
            # Assume grayscale video (user specified)
            # Handle both grayscale and color, but treat as grayscale
            if len(video_shape) == 4:
                # Color video - convert to grayscale
                height, width = video_shape[1], video_shape[2]
                channels = video_shape[3]
                print(f"Note: Color video detected ({channels} channels), will display as grayscale")
            else:
                # Grayscale
                height, width = video_shape[1], video_shape[2]
                channels = 1
            
            # Load ONLY the first frame to create the layer
            first_frame = self.video_loader.get_frame(0)
            
            # Convert to grayscale if needed
            if len(first_frame.shape) == 3:
                # Color frame - convert to grayscale
                first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
            
            # Verify frame is grayscale
            if len(first_frame.shape) != 2:
                raise ValueError(f"Expected grayscale frame (H, W), got shape: {first_frame.shape}")
            
            print(f"First frame shape: {first_frame.shape}, dtype: {first_frame.dtype}, min={first_frame.min()}, max={first_frame.max()}")
            
            # Create single-frame grayscale array: (1, H, W)
            initial_data = first_frame[np.newaxis, ...]
            
            # Remove old layer
            try:
                if "video" in self.viewer.layers:
                    self.viewer.layers.remove("video")
            except Exception:
                pass
            
            # Store video metadata
            self.num_frames = num_frames
            self.video_dtype = video_dtype
            self.height = height
            self.width = width
            
            # Calculate memory per frame (grayscale)
            bytes_per_frame = height * width * np.dtype(video_dtype).itemsize
            full_array_size = num_frames * bytes_per_frame / (1024**3)  # GB
            
            print(f"Video size if fully loaded: {full_array_size:.2f} GB")
            if full_array_size > 1.0:
                print(f"WARNING: Full video would use {full_array_size:.2f} GB - using lazy loading")
            
            # Single frame grayscale array: (1, H, W)
            video_array = initial_data.copy()
            
            # Add layer with SINGLE frame only
            # For grayscale, we add as 2D image (H, W) - Napari handles this better
            # Remove time dimension for initial display
            video_array_2d = video_array[0]  # Shape: (H, W)
            
            add_kwargs = {'name': 'video'}
            # Set contrast limits based on actual data
            data_min = float(video_array_2d.min())
            data_max = float(video_array_2d.max())
            if data_max > data_min:
                add_kwargs['contrast_limits'] = (data_min, data_max)
            elif video_dtype == np.uint8:
                add_kwargs['contrast_limits'] = (0, 255)
            
            # Add the image layer as 2D (H, W) - Napari displays this better
            layer_result = self.viewer.add_image(video_array_2d, **add_kwargs)
            
            # Store the 2D shape for later updates
            self.video_shape_2d = video_array_2d.shape
            
            # Napari's add_image can return either a single layer or a list
            # Get the actual layer from the viewer's layers collection to be safe
            if "video" in self.viewer.layers:
                self.video_layer = self.viewer.layers["video"]
            elif isinstance(layer_result, list):
                # If it returned a list, use the first one
                self.video_layer = layer_result[0] if layer_result else None
            else:
                self.video_layer = layer_result
            
            if self.video_layer is None:
                raise ValueError("Failed to create video layer")
            
            # Verify it's actually a layer object
            if not hasattr(self.video_layer, 'data'):
                raise ValueError(f"Video layer is not a valid layer: {type(self.video_layer)}")
            
            # Configure layer
            self.video_layer.visible = True
            self.video_layer.opacity = 1.0
            data_min = float(video_array_2d.min())
            data_max = float(video_array_2d.max())
            if data_max > data_min:
                self.video_layer.contrast_limits = (data_min, data_max)
            
            self.viewer.reset_view()
            self.video_layer.refresh()
            
            # Simple navigation: slider + play button
            nav_layout = QVBoxLayout()
            
            # Frame slider - compact and easy to use
            slider_row = QHBoxLayout()
            slider_row.addWidget(QLabel("Frame:"))
            self.frame_slider = QSlider()
            self.frame_slider.setOrientation(Qt.Horizontal)  # Horizontal
            self.frame_slider.setMinimum(0)
            self.frame_slider.setMaximum(max(0, num_frames - 1))
            self.frame_slider.setValue(0)
            self.frame_slider.setEnabled(True)
            self.frame_slider.setMaximumHeight(20)  # Compact height for easier dragging
            self.frame_slider.setMinimumHeight(20)
            self.frame_slider.setToolTip(f"Drag to navigate frames (0-{num_frames-1}). Use arrow keys when focused.")
            self.frame_slider.valueChanged.connect(self.on_slider_change)
            # Enable keyboard focus for arrow keys
            self.frame_slider.setFocusPolicy(Qt.StrongFocus)
            slider_row.addWidget(self.frame_slider, stretch=1)  # Make slider expandable
            self.frame_label = QLabel("0")
            self.frame_label.setMinimumWidth(60)  # Fixed width for frame number
            slider_row.addWidget(self.frame_label)
            nav_layout.addLayout(slider_row)
            
            # Play button
            self.play_button = QPushButton("Play")
            self.play_button.clicked.connect(self.toggle_play)
            nav_layout.addWidget(self.play_button)
            
            self.layout().addLayout(nav_layout)
            
            # Install event filter for keyboard navigation on the viewer
            self._setup_keyboard_navigation()
            
            # Store state
            self.num_frames = num_frames
            self.current_frame_idx = 0
            self.is_playing = False
            self.play_timer = None
            del first_frame
            
            self.info_label.setText(f"Loaded: {Path(path_str).name} ({num_frames} frames)")
            
        except Exception as e:
            error_msg = f"Error: {type(e).__name__}: {str(e)}"
            self.info_label.setText(error_msg)
            print(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            if self.video_loader:
                self.video_loader.close()
                self.video_loader = None
    
    def on_slider_change(self, value):
        """Load frame when slider changes."""
        if self.video_loader is None or "video" not in self.viewer.layers:
            return
        
        requested_frame = int(value)
        
        # Always update label even if frame is the same
        if self.frame_label:
            self.frame_label.setText(str(requested_frame))
        
        # Skip frame loading if it's the same frame (optimization)
        if requested_frame == self.current_frame_idx:
            # Still notify callback in case plot needs to update
            if self.frame_changed_callback:
                self.frame_changed_callback()
            return
        
        try:
            # Load frame from disk
            frame = self.video_loader.get_frame(requested_frame)
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Update layer
            video_layer = self.viewer.layers["video"]
            if video_layer.data.shape == frame.shape:
                np.copyto(video_layer.data, frame)
            else:
                video_layer.data = frame
            video_layer.refresh()
            
            # Update UI
            self.current_frame_idx = requested_frame
            
            # Notify plot widget - always call this to ensure plot updates
            if self.frame_changed_callback:
                self.frame_changed_callback()
                
        except Exception as e:
            print(f"Error loading frame {value}: {e}")
    
    def toggle_play(self):
        """Toggle video playback."""
        if self.video_loader is None or self.num_frames == 0:
            return
        
        if self.is_playing:
            self.is_playing = False
            self.play_button.setText("Play")
            if self.play_timer:
                self.play_timer.stop()
        else:
            self.is_playing = True
            self.play_button.setText("Pause")
            from qtpy.QtCore import QTimer
            if self.play_timer is None:
                self.play_timer = QTimer()
                self.play_timer.timeout.connect(self.advance_frame)
            self.play_timer.start(33)  # ~30 fps
    
    def advance_frame(self):
        """Advance to next frame during playback."""
        if not self.is_playing or self.video_loader is None or not self.frame_slider:
            return
        
        current = self.frame_slider.value()
        if current < self.num_frames - 1:
            self.frame_slider.setValue(current + 1)
        else:
            self.toggle_play()
    
    def _setup_keyboard_navigation(self):
        """Setup keyboard navigation for arrow keys."""
        from qtpy.QtCore import QObject
        
        class KeyboardFilter(QObject):
            """Event filter for keyboard navigation."""
            def __init__(self, video_widget):
                super().__init__()
                self.video_widget = video_widget
            
            def eventFilter(self, obj, event):
                """Filter key press events."""
                from qtpy.QtGui import QKeyEvent
                if isinstance(event, QKeyEvent) and event.type() == QKeyEvent.KeyPress:
                    if self.video_widget.frame_slider is None or self.video_widget.video_loader is None:
                        return False
                    
                    key = event.key()
                    current = self.video_widget.frame_slider.value()
                    
                    if key == Qt.Key_Left:
                        # Move backward one frame
                        new_frame = max(0, current - 1)
                        self.video_widget.frame_slider.setValue(new_frame)
                        return True
                    elif key == Qt.Key_Right:
                        # Move forward one frame
                        new_frame = min(self.video_widget.num_frames - 1, current + 1)
                        self.video_widget.frame_slider.setValue(new_frame)
                        return True
                    elif key == Qt.Key_Space:
                        # Toggle play/pause
                        self.video_widget.toggle_play()
                        return True
                
                return False
        
        # Install event filter on the viewer's canvas
        try:
            qt_viewer = self.viewer.window.qt_viewer
            self.keyboard_filter = KeyboardFilter(self)
            qt_viewer.canvas.installEventFilter(self.keyboard_filter)
            # Also make canvas focusable
            qt_viewer.canvas.setFocusPolicy(Qt.StrongFocus)
        except Exception as e:
            print(f"Could not setup keyboard navigation: {e}")


class ExcelPlotWidget(QWidget):
    def __init__(self, viewer, video_widget=None, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.video_widget = video_widget  # Reference to video widget
        self.df = None
        self.frame_range = 100  # Default ±100 frames
        self.show_all = False  # Whether to show entire dataset
        self.selected_features = []  # List of selected feature names
        self.plot_axes = {}  # Dict: feature_name -> (ax, frame_vline, plot_range)
        self.plot_slider = None  # Slider for plot navigation

        # --- Layout ---
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Button: load Excel
        self.load_button = QPushButton("Load Excel file...")
        self.load_button.clicked.connect(self.load_excel)
        main_layout.addWidget(self.load_button)

        # Info label
        self.info_label = QLabel("No file loaded.")
        main_layout.addWidget(self.info_label)

        # Feature selection: collapsible section
        feature_header = QHBoxLayout()
        feature_label = QLabel("Select features to plot:")
        feature_header.addWidget(feature_label)
        feature_header.addStretch()
        self.toggle_features_button = QPushButton("Hide")
        self.toggle_features_button.setMaximumWidth(60)
        self.toggle_features_button.clicked.connect(self.toggle_features_visibility)
        self.toggle_features_button.setEnabled(False)
        feature_header.addWidget(self.toggle_features_button)
        main_layout.addLayout(feature_header)
        
        self.feature_list = QListWidget()
        self.feature_list.setEnabled(False)
        self.feature_list.itemChanged.connect(self.on_feature_selection_changed)
        self.feature_list.setVisible(True)  # Start visible
        main_layout.addWidget(self.feature_list)

        # Range controls
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Frame range (±):"))
        self.range_spinbox = QSpinBox()
        self.range_spinbox.setMinimum(1)
        self.range_spinbox.setMaximum(10000)
        self.range_spinbox.setValue(100)
        self.range_spinbox.setEnabled(False)
        self.range_spinbox.valueChanged.connect(self.on_range_changed)
        range_layout.addWidget(self.range_spinbox)
        
        self.show_all_button = QPushButton("Show All")
        self.show_all_button.setEnabled(False)
        self.show_all_button.clicked.connect(self.toggle_show_all)
        range_layout.addWidget(self.show_all_button)
        main_layout.addLayout(range_layout)

        # Scrollable area for plots
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(400)
        
        self.plots_widget = QWidget()
        self.plots_layout = QVBoxLayout()
        self.plots_widget.setLayout(self.plots_layout)
        scroll.setWidget(self.plots_widget)
        main_layout.addWidget(scroll)
        
        # Matplotlib figure (will be created dynamically)
        self.fig = None
        self.canvas = None
        
        # Frame slider for plot navigation
        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Frame:"))
        self.plot_slider = QSlider()
        self.plot_slider.setOrientation(Qt.Horizontal)
        self.plot_slider.setMinimum(0)
        self.plot_slider.setMaximum(0)  # Will be updated when data is loaded
        self.plot_slider.setValue(0)
        self.plot_slider.setEnabled(False)
        self.plot_slider.setMaximumHeight(20)
        self.plot_slider.setMinimumHeight(20)
        self.plot_slider.setToolTip("Drag to navigate frames. Synced with video.")
        self.plot_slider.valueChanged.connect(self.on_plot_slider_change)
        self.plot_slider.setFocusPolicy(Qt.StrongFocus)
        slider_row.addWidget(self.plot_slider, stretch=1)
        self.plot_frame_label = QLabel("0")
        self.plot_frame_label.setMinimumWidth(60)
        slider_row.addWidget(self.plot_frame_label)
        main_layout.addLayout(slider_row)

    def load_excel(self):
        """Open a dialog, load Excel into a DataFrame, populate feature combo."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Excel file",
            "",
            "Excel files (*.xlsx *.xls);;All files (*.*)",
        )
        if not path:
            return

        # Load Excel
        self.df = pd.read_excel(path)

        # Use row index as "frame" (0, 1, 2, ...) for now
        self.df = self.df.reset_index(drop=True)

        # Only numeric columns for plotting
        numeric_cols = self.df.select_dtypes(include="number").columns.tolist()

        if not numeric_cols:
            self.info_label.setText("Loaded file, but no numeric columns found.")
            self.feature_list.clear()
            self.feature_list.setEnabled(False)
            self.toggle_features_button.setEnabled(False)
            self.range_spinbox.setEnabled(False)
            self.show_all_button.setEnabled(False)
            self._clear_plots()
            return

        self.info_label.setText(f"Loaded: {path}")
        
        # Populate feature list with checkboxes
        self.feature_list.clear()
        for col in numeric_cols:
            item = QListWidgetItem(col)
            item.setCheckState(Qt.Unchecked)
            self.feature_list.addItem(item)
        self.feature_list.setEnabled(True)
        self.toggle_features_button.setEnabled(True)
        
        # Enable range controls
        self.range_spinbox.setEnabled(True)
        self.show_all_button.setEnabled(True)
        
        # Update plot slider range
        if self.plot_slider:
            max_frame = len(self.df) - 1
            self.plot_slider.setMaximum(max(0, max_frame))
            self.plot_slider.setEnabled(True)
            # Sync with video slider if available
            if self.video_widget and self.video_widget.frame_slider:
                current_video_frame = self.video_widget.frame_slider.value()
                current_video_frame = max(0, min(current_video_frame, max_frame))
                self.plot_slider.setValue(current_video_frame)
        
        # Clear previous plots
        self._clear_plots()

    def get_current_frame(self):
        """Get current frame index from video widget if available."""
        # Get frame from video widget's slider
        if (self.video_widget and 
            hasattr(self.video_widget, 'frame_slider') and 
            self.video_widget.frame_slider is not None):
            return self.video_widget.frame_slider.value()
        return 0
    
    def toggle_features_visibility(self):
        """Toggle visibility of the feature selection list."""
        is_visible = self.feature_list.isVisible()
        self.feature_list.setVisible(not is_visible)
        self.toggle_features_button.setText("Show" if is_visible else "Hide")
    
    def _clear_plots(self):
        """Clear all plots and remove from layout."""
        # Remove all widgets from plots layout
        while self.plots_layout.count():
            item = self.plots_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.plot_axes = {}
        self.fig = None
        self.canvas = None
    
    def on_feature_selection_changed(self):
        """Handle feature selection changes."""
        if self.df is None:
            return
        
        # Get selected features
        selected = []
        for i in range(self.feature_list.count()):
            item = self.feature_list.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.text())
        
        self.selected_features = selected
        self.update_all_plots()
    
    def on_range_changed(self, value):
        """Handle frame range change."""
        self.frame_range = value
        if not self.show_all:
            self.update_all_plots()
    
    def toggle_show_all(self):
        """Toggle between show all and zoomed view."""
        self.show_all = not self.show_all
        if self.show_all:
            self.show_all_button.setText("Show Zoomed")
            self.range_spinbox.setEnabled(False)
        else:
            self.show_all_button.setText("Show All")
            self.range_spinbox.setEnabled(True)
        self.update_all_plots()
    
    def update_all_plots(self):
        """Update all selected feature plots."""
        if self.df is None or not self.selected_features:
            self._clear_plots()
            return
        
        frame = self.get_current_frame()
        frame = max(0, min(frame, len(self.df) - 1))
        
        # Determine x-axis range
        if self.show_all:
            x_min = 0
            x_max = len(self.df) - 1
        else:
            x_min = max(0, frame - self.frame_range)
            x_max = min(len(self.df) - 1, frame + self.frame_range)
        
        # Create figure if needed or if number of features changed
        num_features = len(self.selected_features)
        if self.fig is None or len(self.plot_axes) != num_features:
            self._clear_plots()
            self.fig = Figure(figsize=(8, 3 * num_features))
            self.canvas = FigureCanvas(self.fig)
            self.canvas.mpl_connect('button_press_event', self.on_plot_click)
            
            # Create subplots for each feature
            for i, feature_name in enumerate(self.selected_features):
                ax = self.fig.add_subplot(num_features, 1, i + 1)
                self.plot_axes[feature_name] = {
                    'ax': ax,
                    'vline': None,
                    'plot_range': None,
                    'index': i
                }
            
            self.plots_layout.addWidget(self.canvas)
        
        # Update each plot
        for feature_name in self.selected_features:
            if feature_name not in self.plot_axes:
                continue
            
            ax = self.plot_axes[feature_name]['ax']
            ax.clear()
            plot_idx = self.plot_axes[feature_name]['index']
            
            # Get data range
            if self.show_all:
                plot_x_min = x_min
                plot_x_max = x_max
            else:
                plot_x_min = x_min
                plot_x_max = x_max
            
            # Plot data
            x = self.df.index[plot_x_min:plot_x_max+1].to_numpy()
            y = self.df[feature_name].iloc[plot_x_min:plot_x_max+1].to_numpy()
            
            ax.plot(x, y, 'b-', linewidth=1.5)
            vline = ax.axvline(frame, color='r', linestyle="--", linewidth=2)
            
            ax.set_title(feature_name)
            ax.set_xlabel("Frame" if plot_idx == num_features - 1 else "")
            ax.set_ylabel(feature_name)
            ax.grid(True, alpha=0.3)
            
            # Set x-axis limits
            padding = (plot_x_max - plot_x_min) * 0.05 if plot_x_max > plot_x_min else 1
            ax.set_xlim(plot_x_min - padding, plot_x_max + padding)
            
            # Auto-scale y-axis
            valid_y = y[np.isfinite(y)]
            if len(valid_y) > 0:
                y_min = valid_y.min()
                y_max = valid_y.max()
                if np.isfinite(y_min) and np.isfinite(y_max):
                    y_padding = (y_max - y_min) * 0.1 if y_max != y_min else abs(y_max) * 0.1 if y_max != 0 else 1
                    ax.set_ylim(y_min - y_padding, y_max + y_padding)
            
            # Store references
            self.plot_axes[feature_name]['vline'] = vline
            self.plot_axes[feature_name]['plot_range'] = (plot_x_min, plot_x_max)
        
        self.fig.tight_layout()
        self.canvas.draw_idle()
    
    def on_plot_click(self, event):
        """Click plot to navigate video."""
        if self.df is None or event.xdata is None:
            return
        
        # Check if click is in any of our axes
        clicked_ax = None
        for feature_name, plot_data in self.plot_axes.items():
            if event.inaxes == plot_data['ax']:
                clicked_ax = plot_data['ax']
                break
        
        if clicked_ax is None:
            return
        
        clicked_frame = max(0, min(int(round(event.xdata)), len(self.df) - 1))
        # Update both sliders
        if self.plot_slider:
            self.plot_slider.setValue(clicked_frame)
        if self.video_widget and self.video_widget.frame_slider:
            self.video_widget.frame_slider.setValue(clicked_frame)
    
    def on_plot_slider_change(self, value):
        """Handle plot slider changes - sync with video slider."""
        if self.df is None:
            return
        
        frame = int(value)
        frame = max(0, min(frame, len(self.df) - 1))
        
        # Update label
        if self.plot_frame_label:
            self.plot_frame_label.setText(str(frame))
        
        # Sync with video slider
        if self.video_widget and self.video_widget.frame_slider:
            # Block signals to avoid recursive updates
            self.video_widget.frame_slider.blockSignals(True)
            self.video_widget.frame_slider.setValue(frame)
            self.video_widget.frame_slider.blockSignals(False)
            # Manually trigger update
            self.video_widget.on_slider_change(frame)
        
        # Update plot
        self.on_frame_change()

    def on_frame_change(self, event=None):
        """Update plot when video frame changes."""
        if self.df is None:
            return
        frame = self.get_current_frame()
        frame = max(0, min(frame, len(self.df) - 1))
        
        # Sync plot slider with current frame
        if self.plot_slider and self.plot_slider.value() != frame:
            self.plot_slider.blockSignals(True)
            self.plot_slider.setValue(frame)
            self.plot_slider.blockSignals(False)
            if self.plot_frame_label:
                self.plot_frame_label.setText(str(frame))
        
        if not self.selected_features:
            return
        
        # Check if we need to replot (frame outside current range or show_all mode)
        need_replot = False
        if self.show_all:
            # In show_all mode, just update vertical lines
            need_replot = False
        else:
            # Check if frame is outside any plot's range
            for feature_name in self.selected_features:
                if feature_name not in self.plot_axes:
                    need_replot = True
                    break
                plot_range = self.plot_axes[feature_name]['plot_range']
                if plot_range is None:
                    need_replot = True
                    break
                if frame < plot_range[0] or frame > plot_range[1]:
                    need_replot = True
                    break
        
        if need_replot:
            # Full replot with new frame center
            self.update_all_plots()
        else:
            # Just update the vertical line positions (much faster)
            if self.canvas and self.fig:
                for feature_name in self.selected_features:
                    if feature_name in self.plot_axes:
                        vline = self.plot_axes[feature_name]['vline']
                        if vline is not None:
                            vline.set_xdata([frame, frame])
                self.canvas.draw_idle()


def setup_drag_drop(viewer, video_widget):
    """Setup drag and drop handling for video files."""
    from qtpy.QtGui import QDragEnterEvent, QDropEvent
    import warnings
    
    # Suppress deprecation warning for now
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qt_viewer = viewer.window.qt_viewer
    
    # Store original methods if they exist
    original_drop = getattr(qt_viewer, 'dropEvent', None)
    original_drag_enter = getattr(qt_viewer, 'dragEnterEvent', None)
    
    def drag_enter_event(event: QDragEnterEvent):
        """Accept video file drags."""
        mime_data = event.mimeData()
        if mime_data.hasUrls():
            for url in mime_data.urls():
                path = url.toLocalFile()
                if path and Path(path).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                    event.acceptProposedAction()
                    return
        # Fall back to original handler
        if original_drag_enter:
            original_drag_enter(event)
        else:
            event.ignore()
    
    def drop_event(event: QDropEvent):
        """Handle video file drops."""
        mime_data = event.mimeData()
        video_handled = False
        if mime_data.hasUrls():
            for url in mime_data.urls():
                path = url.toLocalFile()
                if path and Path(path).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                    try:
                        video_widget.load_video_file(path)
                        event.acceptProposedAction()
                        video_handled = True
                        break
                    except Exception as e:
                        print(f"Error in custom video loader: {e}")
                        # Try Napari's native loading as fallback
                        if original_drop:
                            original_drop(event)
                            return
        
        if not video_handled:
            # Fall back to Napari's default handling for other files
            if original_drop:
                original_drop(event)
            else:
                event.ignore()
    
    # Install event handlers
    qt_viewer.dragEnterEvent = drag_enter_event
    qt_viewer.dropEvent = drop_event


def main():
    viewer = napari.Viewer()

    # Add video loader widget
    video_widget = VideoLoaderWidget(viewer)
    viewer.window.add_dock_widget(video_widget, area="left", name="Video Loader")
    
    # Add Excel plot widget with reference to video widget
    excel_widget = ExcelPlotWidget(viewer, video_widget=video_widget)
    viewer.window.add_dock_widget(excel_widget, area="right", name="Feature Plot")
    
    # Connect video widget frame changes to Excel plot updates
    # When video frame changes, update the plot's frame indicator
    def update_plot_on_frame_change():
        if excel_widget.df is not None:
            excel_widget.on_frame_change()
    
    # Store callback in video widget
    video_widget.frame_changed_callback = update_plot_on_frame_change
    
    # Enable drag and drop for videos
    setup_drag_drop(viewer, video_widget)

    napari.run()


if __name__ == "__main__":
    main()
