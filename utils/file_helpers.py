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
    QVBoxLayout,
    QWidget,
)


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
            
            # Determine if color
            is_color = len(video_shape) == 4
            if is_color:
                height, width, channels = video_shape[1], video_shape[2], video_shape[3]
            else:
                height, width = video_shape[1], video_shape[2]
                channels = 1
            
            # Load ONLY the first frame to create the layer
            # This is the ONLY frame that will be in memory initially
            first_frame = self.video_loader.get_frame(0)
            
            # Create a single-frame array for Napari
            # We'll update this array when the slider moves
            if is_color:
                # For color, we create a single time point: (1, H, W, C)
                initial_data = first_frame[np.newaxis, ...]  # Add time dimension
            else:
                # Grayscale: (1, H, W)
                initial_data = first_frame[np.newaxis, ...]
            
            # Remove old layer
            try:
                if "video" in self.viewer.layers:
                    self.viewer.layers.remove("video")
            except Exception:
                pass
            
            # Store video metadata
            self.num_frames = num_frames
            self.is_color = is_color
            self.video_dtype = video_dtype
            self.height = height
            self.width = width
            self.channels = channels
            
            # MEMORY-EFFICIENT APPROACH: Don't create full array!
            # Instead, create a minimal array and use a custom update mechanism
            # We'll create a small buffer (e.g., 3 frames) and update it dynamically
            
            # Calculate memory per frame
            if is_color:
                bytes_per_frame = height * width * channels * np.dtype(video_dtype).itemsize
                full_array_size = num_frames * bytes_per_frame / (1024**3)  # GB
            else:
                bytes_per_frame = height * width * np.dtype(video_dtype).itemsize
                full_array_size = num_frames * bytes_per_frame / (1024**3)  # GB
            
            print(f"Video size if fully loaded: {full_array_size:.2f} GB")
            if full_array_size > 1.0:
                print(f"WARNING: Full video would use {full_array_size:.2f} GB - using lazy loading")
            
            # CRITICAL: Don't create full array! Only create single-frame array
            # We'll update it dynamically when user navigates frames
            # This prevents allocating 60GB of memory
            
            if is_color:
                # Single frame array: (1, H, W, C) - minimal memory
                video_array = first_frame[np.newaxis, ...].copy()
            else:
                # Single frame: (1, H, W)
                video_array = first_frame[np.newaxis, ...].copy()
            
            # Add layer with SINGLE frame only
            # Note: Don't use channel_axis - it returns a list of layers
            # Instead, add as RGB/RGBA image directly
            add_kwargs = {'name': 'video'}
            add_kwargs['contrast_limits'] = [0, 255] if video_dtype == np.uint8 else None
            
            # Add the image layer
            layer_result = self.viewer.add_image(video_array, **add_kwargs)
            
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
            
            print(f"Video layer created: {type(self.video_layer)}, data shape: {self.video_layer.data.shape}")
            
            # Add frame navigation controls to the widget
            from qtpy.QtWidgets import QSlider, QHBoxLayout, QPushButton
            nav_layout = QHBoxLayout()
            nav_layout.addWidget(QLabel("Frame:"))
            
            # Create slider
            self.frame_slider = QSlider()
            self.frame_slider.setOrientation(0)  # Horizontal slider
            self.frame_slider.setMinimum(0)
            self.frame_slider.setMaximum(max(0, num_frames - 1))
            self.frame_slider.setValue(0)
            self.frame_slider.setEnabled(True)
            # Use sliderReleased for less frequent updates, or valueChanged for real-time
            self.frame_slider.sliderMoved.connect(self.on_slider_change)
            self.frame_slider.valueChanged.connect(self.on_slider_change)
            nav_layout.addWidget(self.frame_slider)
            
            self.frame_label = QLabel(f"0 / {num_frames - 1}")
            nav_layout.addWidget(self.frame_label)
            
            # Add play/pause buttons
            self.play_button = QPushButton("Play")
            self.play_button.clicked.connect(self.toggle_play)
            nav_layout.addWidget(self.play_button)
            
            self.layout().addLayout(nav_layout)
            
            # Store state
            self.num_frames = num_frames
            self.current_frame_idx = 0
            self.is_playing = False
            self.play_timer = None
            del first_frame
            
            self.info_label.setText(f"Loaded: {Path(path_str).name} ({num_frames} frames)")
            print(f"SUCCESS: Video opened. Only current frame in memory.")
            print(f"Frame slider range: 0 to {num_frames - 1}")
            
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
        """Load frame when slider changes - TRUE LAZY LOADING."""
        if self.video_loader is None:
            return
        
        # Always get the layer from viewer's layers collection (most reliable)
        if "video" not in self.viewer.layers:
            print("Warning: Video layer not found in viewer")
            return
        
        # Get the layer directly from viewer (this ensures we have the correct reference)
        video_layer = self.viewer.layers["video"]
        
        # Safety check
        if not hasattr(video_layer, 'data'):
            print(f"Warning: Video layer has no data attribute: {type(video_layer)}")
            return
        
        try:
            requested_frame = int(value)
            
            # Skip if already showing this frame (avoid redundant loads)
            if requested_frame == self.current_frame_idx:
                return
            
            # Load ONLY this frame from disk
            frame = self.video_loader.get_frame(requested_frame)
            
            # Update the SINGLE frame in our array (replacing the old one)
            # This is the key - we only have ONE frame in memory at a time
            # The data array has shape (1, H, W) or (1, H, W, C)
            
            # Prepare frame with time dimension: (1, H, W) or (1, H, W, C)
            if len(frame.shape) == 2:
                # Grayscale: add time dimension
                frame_with_time = frame[np.newaxis, ...]
            elif len(frame.shape) == 3:
                # Color: add time dimension
                frame_with_time = frame[np.newaxis, ...]
            else:
                print(f"Warning: Unexpected frame shape: {frame.shape}")
                return
            
            # Update the layer data directly
            # Napari will automatically refresh when we assign to .data
            video_layer.data = frame_with_time
            
            # Also update our stored reference
            self.video_layer = video_layer
            
            # Update label
            if self.frame_label:
                self.frame_label.setText(f"{requested_frame} / {self.num_frames - 1}")
            
            # Store current frame
            self.current_frame_idx = requested_frame
            
            # Free the frame (it's copied into video_layer.data)
            del frame
            
            # Notify other widgets (e.g., Excel plot) of frame change
            if self.frame_changed_callback:
                try:
                    self.frame_changed_callback()
                except Exception as e:
                    print(f"Error in frame change callback: {e}")
            
        except Exception as e:
            print(f"Error loading frame {value}: {e}")
            import traceback
            traceback.print_exc()
    
    def toggle_play(self):
        """Toggle video playback."""
        if self.video_loader is None or self.num_frames == 0:
            return
        
        if self.is_playing:
            # Stop playback
            self.is_playing = False
            self.play_button.setText("Play")
            if self.play_timer:
                self.play_timer.stop()
                self.play_timer = None
        else:
            # Start playback
            self.is_playing = True
            self.play_button.setText("Pause")
            from qtpy.QtCore import QTimer
            self.play_timer = QTimer()
            self.play_timer.timeout.connect(self.advance_frame)
            self.play_timer.start(33)  # ~30 fps
    
    def advance_frame(self):
        """Advance to next frame during playback."""
        if not self.is_playing or self.video_loader is None:
            return
        
        current = self.frame_slider.value()
        if current < self.num_frames - 1:
            self.frame_slider.setValue(current + 1)
        else:
            # Reached end, stop playback
            self.toggle_play()


class ExcelPlotWidget(QWidget):
    def __init__(self, viewer, video_widget=None, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.video_widget = video_widget  # Reference to video widget
        self.df = None
        self.frame_vline = None

        # --- Layout ---
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Button: load Excel
        self.load_button = QPushButton("Load Excel file...")
        self.load_button.clicked.connect(self.load_excel)
        layout.addWidget(self.load_button)

        # Info label
        self.info_label = QLabel("No file loaded.")
        layout.addWidget(self.info_label)

        # Combo: choose feature (column)
        layout.addWidget(QLabel("Feature to plot:"))
        self.feature_combo = QComboBox()
        self.feature_combo.currentTextChanged.connect(self.update_plot)
        self.feature_combo.setEnabled(False)
        layout.addWidget(self.feature_combo)

        # Matplotlib figure
        self.fig = Figure(figsize=(4, 3))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        layout.addWidget(self.canvas)

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
            self.feature_combo.clear()
            self.feature_combo.setEnabled(False)
            self.ax.clear()
            self.canvas.draw_idle()
            return

        self.info_label.setText(f"Loaded: {path}")
        self.feature_combo.clear()
        self.feature_combo.addItems(numeric_cols)
        self.feature_combo.setEnabled(True)

        # Plot first feature by default
        self.update_plot(self.feature_combo.currentText())

    def get_current_frame(self):
        """Get current frame index from video widget if available."""
        # Get frame from video widget's slider
        if self.video_widget and hasattr(self.video_widget, 'frame_slider'):
            return self.video_widget.frame_slider.value()
        return 0

    def update_plot(self, feature_name: str):
        """Plot the selected feature vs row index, add/update vertical frame line."""
        if self.df is None or not feature_name:
            return

        self.ax.clear()

        x = self.df.index.to_numpy()  # treat row index as "frame"
        y = self.df[feature_name].to_numpy()

        self.ax.plot(x, y, 'b-', linewidth=1.5)
        self.ax.set_title(feature_name)
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel(feature_name)
        self.ax.grid(True, alpha=0.3)

        # Add a vertical line at current frame
        frame = self.get_current_frame()
        # Clamp frame to valid range
        frame = max(0, min(frame, len(self.df) - 1))
        self.frame_vline = self.ax.axvline(frame, color='r', linestyle="--", linewidth=2, label='Current frame')
        
        # Set x-axis limits to show full data range
        self.ax.set_xlim(-0.5, len(self.df) - 0.5)

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def on_frame_change(self, event=None):
        """Move vertical line when the viewer's current frame changes."""
        if self.df is None or self.frame_vline is None:
            return

        frame = self.get_current_frame()
        # Clamp frame to valid range
        frame = max(0, min(frame, len(self.df) - 1))
        self.frame_vline.set_xdata([frame, frame])
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
        if excel_widget.df is not None and excel_widget.frame_vline is not None:
            excel_widget.on_frame_change()
    
    # Store callback in video widget
    video_widget.frame_changed_callback = update_plot_on_frame_change
    
    # Enable drag and drop for videos
    setup_drag_drop(viewer, video_widget)

    napari.run()


if __name__ == "__main__":
    main()
