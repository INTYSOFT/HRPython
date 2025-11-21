"""Definición de la ventana principal basada en PyQt6."""

from __future__ import annotations

import json
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import fitz  # PyMuPDF
import pandas as pd
from PyQt6.QtCore import QEvent, QPoint, QSize, Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QPixmap, QColor, QImage, QWheelEvent, QMouseEvent, QFont
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QFileDialog,
    QComboBox,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QProgressBar,
    QSpinBox,
    QToolButton,
)

from models import AlumnoHoja, Respuesta
from omr_processor import OMRConfig, procesar_pdf


class _ProcessWorker(QObject):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, pdf_path: Path, cache_dir: Path, config: OMRConfig) -> None:
        super().__init__()
        self.pdf_path = pdf_path
        self.cache_dir = cache_dir
        self.config = config

    def run(self) -> None:
        try:
            def _progress_wrapper(current: int, total: int) -> None:
                if total <= 0:
                    self.progress.emit(0)
                    return
                porcentaje = int(max(0, min(100, (current * 100) / total)))
                self.progress.emit(porcentaje)

            resultados = procesar_pdf(
                self.pdf_path,
                self.cache_dir,
                self.config,
                progress_callback=_progress_wrapper,
            )
        except Exception as exc:  # pragma: no cover - mostrado en UI
            self.error.emit(str(exc))
            return
        self.finished.emit(resultados)


class MainWindow(QMainWindow):
    """Ventana principal con estilo moderno y paneles divididos."""

    API_BASE = "http://192.168.1.50:5000"
    ALL_SECTIONS_KEY = "__all__"
    PREVIEW_DPI = 300
    MIN_ZOOM = 0.1
    MAX_ZOOM = 3.0
    ZOOM_STEP = 0.05

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Lector OMR")
        self.resize(1400, 800)
        self.pdf_path: Path | None = None
        self.cache_dir = Path(tempfile.mkdtemp(prefix="omr_cache_"))
        self.resultados: List[AlumnoHoja] = []
        self.resultados_por_dni: Dict[str, AlumnoHoja] = {}
        self.resultados_no_asignados: List[AlumnoHoja] = []
        self._sincronizando_no_encontrados = False
        self.config = OMRConfig()
        self.evaluaciones: List[dict] = []
        self.evaluacion_detalle: List[dict] = []
        self._default_status_style = self.statusBar().styleSheet()
        self._pdf_doc: fitz.Document | None = None
        self._current_page = 1
        self._zoom_factor = 1.0
        self._syncing_selection = False
        self._is_panning = False
        self._pan_start: QPoint | None = None

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("Progreso %p%")
        self._progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress_bar.setVisible(False)
        self._progress_bar.setStyleSheet(
            """
            QProgressBar {
                background-color: #fee2e2;
                color: #991b1b;
                border: 1px solid #fecdd3;
                border-radius: 10px;
                padding: 4px;
            }
            QProgressBar::chunk {
                background-color: #f87171;
                border-radius: 8px;
            }
            """
        )
        self.statusBar().addPermanentWidget(self._progress_bar, 1)
        self._process_thread: QThread | None = None
        self._process_worker: _ProcessWorker | None = None

        self._build_ui()
        self._reset_pdf_state()
        self._load_evaluaciones()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        # ---------- WIDGET CENTRAL Y LAYOUT RAÍZ ----------
        central = QWidget(self)
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        # top = 0 para que todo quede pegado al borde superior de la ventana
        main_layout.setContentsMargins(0, 0, 0, 8)
        main_layout.setSpacing(4)

        # ==================================================
        #      ZONA SUPERIOR: TÍTULO + CONTROLES
        # ==================================================
        top_container = QWidget()
        top_layout = QVBoxLayout(top_container)
        # solo un poco de margen lateral y abajo, nada arriba
        top_layout.setContentsMargins(8, 0, 8, 4)
        top_layout.setSpacing(2)

        # ------- fila 1: encabezado (logo + título) -------
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)

        self.logo_badge = QLabel("OMR")
        self.logo_badge.setObjectName("logoBadge")
        self.logo_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.title_label = QLabel("SCAN HOJA DE RESPUESTAS")
        self.title_label.setObjectName("titleLabel")

        header.addWidget(self.logo_badge)
        header.addWidget(self.title_label)
        header.addStretch(1)

        # ------- fila 2: barra de acciones -------
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(6)

        self.combo_evaluaciones = QComboBox()
        self.combo_evaluaciones.setPlaceholderText("Seleccione evaluación")
        self.combo_evaluaciones.setMinimumWidth(180)

        self.combo_secciones = QComboBox()
        self.combo_secciones.setPlaceholderText("Seleccione sección")
        self.combo_secciones.setMinimumWidth(180)

        self.btn_load = QPushButton("Cargar PDF")
        self.btn_process = QPushButton("Procesar")
        self.btn_export = QPushButton("Exportar resultados")
        self.btn_register = QPushButton("Registrar respuestas")

        self.lbl_file = QLabel("Ningún archivo seleccionado")
        self.lbl_file.setObjectName("fileLabel")

        toolbar.addWidget(self.combo_evaluaciones)
        toolbar.addWidget(self.combo_secciones)
        toolbar.addWidget(self.btn_load)
        toolbar.addWidget(self.btn_process)
        toolbar.addWidget(self.btn_export)
        toolbar.addWidget(self.btn_register)
        toolbar.addStretch(1)
        toolbar.addWidget(self.lbl_file)

        # agregar al contenedor superior
        top_layout.addLayout(header)
        top_layout.addLayout(toolbar)

        # contenedor superior primero en el layout principal
        main_layout.addWidget(top_container, 0)

        # ==================================================
        #      ZONA CENTRAL: SPLITTER (TABLAS + VISOR)
        # ==================================================
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)

        # ------------ Panel izquierdo (tablas) ------------
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 0, 4, 0)
        left_layout.setSpacing(6)

        self.table_students = QTableWidget(0, 5)
        self.table_students.setHorizontalHeaderLabels(
            ["Página", "DNI", "Alumno", "Ciclo", "Sección"]
        )
        self.table_students.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.table_students.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection
        )
        self.table_students.verticalHeader().setVisible(False)
        self.table_students.setAlternatingRowColors(True)
        left_layout.addWidget(self.table_students, stretch=3)

        self.table_not_found = QTableWidget(0, 2)
        self.table_not_found.setHorizontalHeaderLabels(["Página", "DNI"])
        self.table_not_found.verticalHeader().setVisible(False)
        self.table_not_found.setAlternatingRowColors(True)
        self.table_not_found.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.table_not_found.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection
        )

        self.table_answers = QTableWidget(0, 3)
        self.table_answers.setHorizontalHeaderLabels(["Pregunta", "Respuesta", "Estado"])
        self.table_answers.verticalHeader().setVisible(False)
        self.table_answers.setAlternatingRowColors(True)

        lower_tables = QHBoxLayout()
        lower_tables.setContentsMargins(0, 0, 0, 0)
        lower_tables.setSpacing(6)
        lower_tables.addWidget(self.table_not_found, stretch=1)
        lower_tables.addWidget(self.table_answers, stretch=2)

        left_layout.addLayout(lower_tables, stretch=4)

        splitter.addWidget(left_panel)       


        # ------------ Panel derecho (visor PDF) -----------
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 0, 8, 0)
        right_layout.setSpacing(4)

        icon_size_small = QSize(10, 10)

        # botones de zoom
        self.btn_zoom_out = QToolButton()
        self.btn_zoom_out.setText("−")
        self.btn_zoom_out.setToolTip("Disminuir zoom")
        self.btn_zoom_out.setObjectName("zoomToolButton")
        self.btn_zoom_out.setAutoRaise(True)
        self.btn_zoom_out.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)

        font = self.btn_zoom_out.font()
        font.setPointSize(10)
        font.setBold(True)
        self.btn_zoom_out.setFont(font)

        self.btn_zoom_in = QToolButton()
        self.btn_zoom_in.setText("+")
        self.btn_zoom_in.setToolTip("Aumentar zoom")
        self.btn_zoom_in.setObjectName("zoomToolButton")
        self.btn_zoom_in.setAutoRaise(True)
        self.btn_zoom_in.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.btn_zoom_in.setFont(font)

        # navegación de páginas
        self.btn_prev_page = QToolButton()
        self.btn_prev_page.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowUp)
        )
        self.btn_prev_page.setIconSize(icon_size_small)
        self.btn_prev_page.setToolTip("Página anterior")
        self.btn_prev_page.setObjectName("navToolButton")
        self.btn_prev_page.setAutoRaise(True)

        self.btn_next_page = QToolButton()
        self.btn_next_page.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowDown)
        )
        self.btn_next_page.setIconSize(icon_size_small)
        self.btn_next_page.setToolTip("Página siguiente")
        self.btn_next_page.setObjectName("navToolButton")
        self.btn_next_page.setAutoRaise(True)

        self.btn_reset_zoom = QToolButton()
        self.btn_reset_zoom.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload)
        )
        self.btn_reset_zoom.setIconSize(icon_size_small)
        self.btn_reset_zoom.setObjectName("resetZoomToolButton")
        self.btn_reset_zoom.setToolTip("Ajustar al visor")
        self.btn_reset_zoom.setAutoRaise(True)

        self.page_selector = QSpinBox()
        self.page_selector.setRange(1, 1)
        self.page_selector.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.page_selector.setMinimumWidth(60)
        self.page_selector.setObjectName("pageSelector")
        self.page_selector.setVisible(False)

        self.lbl_page_info = QLabel("Página 0 / 0")
        self.lbl_current_page = QLabel("0")
        self.lbl_current_page.setObjectName("pageBadge")
        self.lbl_total_pages = QLabel("0")
        self.lbl_total_pages.setObjectName("pageTotal")
        self.lbl_zoom_info = QLabel("100 %")

        # visor de imagen / pdf
        self.image_label = QLabel("Seleccione un alumno")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(350)
        self.image_label.setMouseTracking(True)

        image_scroll = QScrollArea()
        image_scroll.setWidget(self.image_label)
        image_scroll.setWidgetResizable(True)
        self.image_scroll = image_scroll
        self.image_scroll.viewport().installEventFilter(self)
        self.image_label.installEventFilter(self)

        # barra lateral de navegación/zoom
        nav_panel = QWidget()
        nav_panel.setObjectName("navPanel")
        nav_layout = QVBoxLayout(nav_panel)
        nav_layout.setContentsMargins(8, 8, 8, 8)
        nav_layout.setSpacing(6)
        nav_panel.setFixedWidth(44)
        nav_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

        nav_layout.addWidget(self.lbl_current_page, alignment=Qt.AlignmentFlag.AlignHCenter)
        nav_layout.addWidget(self.lbl_total_pages, alignment=Qt.AlignmentFlag.AlignHCenter)
        nav_layout.addSpacing(4)
        nav_layout.addWidget(self.btn_prev_page, alignment=Qt.AlignmentFlag.AlignHCenter)
        nav_layout.addWidget(self.btn_next_page, alignment=Qt.AlignmentFlag.AlignHCenter)
        nav_layout.addSpacing(6)
        nav_layout.addWidget(self.btn_reset_zoom, alignment=Qt.AlignmentFlag.AlignHCenter)
        nav_layout.addWidget(self.btn_zoom_in, alignment=Qt.AlignmentFlag.AlignHCenter)
        nav_layout.addWidget(self.btn_zoom_out, alignment=Qt.AlignmentFlag.AlignHCenter)
        nav_layout.addStretch(1)

        viewer_layout = QHBoxLayout()
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.setSpacing(6)
        viewer_layout.addWidget(image_scroll, stretch=1)
        viewer_layout.addWidget(nav_panel, stretch=0)

        info_bar = QHBoxLayout()
        info_bar.addWidget(self.page_selector)
        info_bar.addWidget(self.lbl_page_info)
        info_bar.addStretch(1)
        info_bar.addWidget(self.lbl_zoom_info)

        right_layout.addLayout(viewer_layout, stretch=3)
        right_layout.addLayout(info_bar)

        splitter.addWidget(right_panel)
        

        # splitter ocupa todo el espacio restante
        # 40% (izquierda) y 60% (derecha)
        splitter.setStretchFactor(0, 15)   # índice 0 -> left_panel
        splitter.setStretchFactor(1, 85)   # índice 1 -> right_panel

        main_layout.addWidget(splitter, 1)
        

        # estilos y señales se aplican fuera
        self._apply_styles()
        self._connect_signals()


    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                background-color: #f8fafc;
                font-family: 'Segoe UI', 'Open Sans', sans-serif;
                font-size: 9pt;
                color: #1f2933;
            }

            QLabel#titleLabel {
                font-size: 16pt;
                font-weight: 700;
                color: #0f172a;
                letter-spacing: 0.5px;
            }

            QLabel#logoBadge {
                min-width: 44px;
                min-height: 44px;
                max-width: 44px;
                max-height: 44px;
                border-radius: 22px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #c7d2fe, stop:1 #b8e0d2);
                color: #0f172a;
                font-weight: 800;
                font-size: 12pt;
                border: 1px solid #dbeafe;
            }

            /* Botones principales más compactos */
            QPushButton {
                background-color: #b8e0d2;
                padding: 4px 10px;             /* antes 10px 18px */
                border-radius: 6px;            /* antes 10px */
                border: 1px solid #a1d2c5;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #a5d6c9;
            }
            QPushButton:pressed {
                background-color: #8ac3b4;
            }

            /* Tablas más limpias */
            QTableWidget {
                background-color: #ffffff;
                border: 1px solid #dfe7ef;
                border-radius: 6px;            /* antes 12px */
                gridline-color: #d0d9e2;
                selection-background-color: #e4d7ff;
                selection-color: #111827;
                alternate-background-color: #f4f7fb;
            }
            QHeaderView::section {
                background-color: #e5edf5;
                border: none;
                padding: 4px 6px;              /* antes 8px */
                font-weight: 600;
                font-size: 8pt;
                color: #0f172a;
            }

            QLabel#fileLabel {
                color: #4b5563;
                font-size: 8pt;
            }

            QScrollArea {
                border: none;
            }

            /* Combos más pequeños */
            QComboBox {
                padding: 4px 8px;              /* antes 8px 12px */
                border-radius: 6px;            /* antes 10px */
                border: 1px solid #cfd8e3;
                background-color: #ffffff;
                min-height: 24px;
            }
            QComboBox:focus {
                border-color: #4ade80;
            }
            QComboBox QListView {
                background-color: #ffffff;
                padding: 2px 0;
            }

            QSplitter::handle {
                background-color: #e1e7ef;
            }

            /* Panel lateral de navegación */
            QWidget#navPanel {
                background-color: #e3f2fd;
                border-radius: 12px;
            }
            QToolButton#navToolButton,
            QToolButton#zoomToolButton,
            QToolButton#resetZoomToolButton {
                background-color: transparent;
                border: none;
                color: #1f2937;
                padding: 4px;
            }
            QToolButton#navToolButton:hover,
            QToolButton#zoomToolButton:hover,
            QToolButton#resetZoomToolButton:hover {
                background-color: rgba(31, 41, 55, 0.08);
                border-radius: 8px;
            }

            QLabel#pageBadge {
                background-color: #a5b4fc;
                color: #0f172a;
                border-radius: 6px;
                padding: 3px 6px;
                min-width: 26px;
                qproperty-alignment: AlignCenter;
                font-weight: 600;
                font-size: 8pt;
            }
            QLabel#pageTotal {
                color: #1f2937;
                qproperty-alignment: AlignCenter;
                font-size: 8pt;
            }

            QSpinBox#pageSelector {
                background-color: #e3f2fd;
                border: 1px solid #c7d2fe;
                border-radius: 6px;
                color: #111827;
                padding: 3px 6px;
                min-height: 22px;
                font-size: 8pt;
            }
            """
        )
        self.lbl_file.setObjectName("fileLabel")


    def _connect_signals(self) -> None:
        self.btn_load.clicked.connect(self._on_load_pdf)
        self.btn_process.clicked.connect(self._on_process)
        self.btn_export.clicked.connect(self._on_export)
        self.btn_register.clicked.connect(self._on_register_responses)
        self.table_students.itemSelectionChanged.connect(self._on_student_selected)
        self.table_not_found.itemSelectionChanged.connect(
            self._on_not_found_selected
        )
        self.combo_evaluaciones.currentIndexChanged.connect(
            self._on_evaluacion_changed
        )
        self.combo_secciones.currentIndexChanged.connect(self._on_seccion_changed)
        self.table_not_found.itemChanged.connect(self._on_not_found_item_changed)
        self.btn_prev_page.clicked.connect(self._on_prev_page)
        self.btn_next_page.clicked.connect(self._on_next_page)
        self.btn_zoom_in.clicked.connect(self._on_zoom_in)
        self.btn_zoom_out.clicked.connect(self._on_zoom_out)
        self.btn_reset_zoom.clicked.connect(self._on_reset_zoom)
        self.page_selector.valueChanged.connect(self._on_page_selected)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # pragma: no cover - interacción UI
        if obj is self.image_scroll.viewport() and event.type() == QEvent.Type.Wheel:
            return self._handle_wheel_zoom(event) or super().eventFilter(obj, event)

        if obj is self.image_label and isinstance(event, QMouseEvent):
            if event.type() == QEvent.Type.MouseButtonPress:
                if (
                    event.button() == Qt.MouseButton.LeftButton
                    and self._can_pan_view()
                ):
                    self._is_panning = True
                    self._pan_start = event.position().toPoint()
                    self.image_label.setCursor(Qt.CursorShape.ClosedHandCursor)
                    return True
            elif event.type() == QEvent.Type.MouseMove and self._is_panning:
                if self._pan_start is None:
                    self._pan_start = event.position().toPoint()
                    return True
                delta = event.position().toPoint() - self._pan_start
                h_bar = self.image_scroll.horizontalScrollBar()
                v_bar = self.image_scroll.verticalScrollBar()
                h_bar.setValue(h_bar.value() - delta.x())
                v_bar.setValue(v_bar.value() - delta.y())
                self._pan_start = event.position().toPoint()
                return True
            elif event.type() == QEvent.Type.MouseButtonRelease:
                if self._is_panning and event.button() == Qt.MouseButton.LeftButton:
                    self._is_panning = False
                    self._pan_start = None
                    self.image_label.unsetCursor()
                    return True
        return super().eventFilter(obj, event)

    def _can_pan_view(self) -> bool:
        """Indica si hay espacio de desplazamiento disponible para arrastrar."""

        h_bar = self.image_scroll.horizontalScrollBar()
        v_bar = self.image_scroll.verticalScrollBar()
        return (h_bar.maximum() > 0) or (v_bar.maximum() > 0)

    # --------------------------------------------------------------- acciones
    def _on_load_pdf(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(self, "Seleccionar PDF", "", "PDF (*.pdf)")
        if not file_name:
            return
        self.pdf_path = Path(file_name)
        try:
            self._pdf_doc = fitz.open(str(self.pdf_path))
        except Exception as exc:  # pragma: no cover - errores de UI
            QMessageBox.critical(self, "Error PDF", f"No se pudo abrir el PDF: {exc}")
            self._pdf_doc = None
            self._toggle_pdf_controls(False)
            self._reset_pdf_state()
            return
        self.lbl_file.setText(self.pdf_path.name)
        self.statusBar().showMessage(f"Archivo cargado: {self.pdf_path}", 5000)
        self._reset_pdf_state()
        self._initialize_pdf_navigation()

    def _on_process(self) -> None:
        if not self.pdf_path:
            QMessageBox.warning(self, "Sin archivo", "Primero seleccione un PDF.")
            return
        self._iniciar_procesamiento_async()

    def _iniciar_procesamiento_async(self) -> None:
        self._preparar_estado_procesamiento()

        self._process_thread = QThread(self)
        self._process_worker = _ProcessWorker(
            self.pdf_path, self.cache_dir, self.config
        )
        self._process_worker.moveToThread(self._process_thread)

        self._process_thread.started.connect(self._process_worker.run)
        self._process_worker.finished.connect(self._process_thread.quit)
        self._process_worker.error.connect(self._process_thread.quit)
        self._process_worker.progress.connect(self._update_progress)
        self._process_worker.finished.connect(self._on_process_completed)
        self._process_worker.error.connect(self._on_process_failed)
        self._process_thread.finished.connect(self._limpiar_hilo_proceso)

        self._process_thread.start()

    def _preparar_estado_procesamiento(self) -> None:
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self.statusBar().setStyleSheet(
            "background-color: #fee2e2; color: #991b1b; font-weight: 600;"
        )
        self.statusBar().showMessage("Procesando... 0%", 0)
        self._toggle_controls(False)

    def _toggle_controls(self, enabled: bool) -> None:
        for widget in (
            self.btn_load,
            self.btn_process,
            self.btn_export,
            self.btn_register,
            self.combo_evaluaciones,
            self.combo_secciones,
        ):
            widget.setEnabled(enabled)

    def _toggle_pdf_controls(self, enabled: bool) -> None:
        for widget in (
            self.btn_prev_page,
            self.btn_next_page,
            self.page_selector,
            self.btn_zoom_in,
            self.btn_zoom_out,
            self.btn_reset_zoom,
        ):
            widget.setEnabled(enabled)

    def _reset_pdf_state(self) -> None:
        self._current_page = 1
        self._zoom_factor = 1.0
        self.lbl_zoom_info.setText("100 %")
        self.lbl_page_info.setText("Página 0 / 0")
        self.lbl_current_page.setText("0")
        self.lbl_total_pages.setText("0")
        self.page_selector.blockSignals(True)
        self.page_selector.setRange(1, 1)
        self.page_selector.setValue(1)
        self.page_selector.blockSignals(False)
        self._toggle_pdf_controls(self._pdf_doc is not None)

    def _initialize_pdf_navigation(self) -> None:
        if not self._pdf_doc:
            self._mostrar_imagen(None)
            return
        total_pages = self._pdf_doc.page_count
        if total_pages <= 0:
            self._mostrar_imagen(None)
            self._toggle_pdf_controls(False)
            return
        self.page_selector.blockSignals(True)
        self.page_selector.setRange(1, total_pages)
        self.page_selector.setValue(1)
        self.page_selector.blockSignals(False)
        self._toggle_pdf_controls(True)
        self._update_page_label(1, total_pages)
        self._zoom_factor = self._calculate_fit_zoom(1)
        self._update_zoom_label()
        self._mostrar_pagina_pdf(1)

    def _on_process_completed(self, resultados: List[AlumnoHoja]) -> None:
        self.resultados = resultados
        self._integrar_resultados_en_tablas()
        self._progress_bar.setValue(100)
        self.statusBar().showMessage("Procesamiento finalizado", 5000)
        self._finalizar_estado_procesamiento()

    def _on_process_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Error procesando", message)
        self.statusBar().clearMessage()
        self._finalizar_estado_procesamiento()

    def _update_progress(self, value: int) -> None:
        clamped_value = max(0, min(100, value))
        self._progress_bar.setValue(clamped_value)
        if not self._progress_bar.isVisible():
            self._progress_bar.setVisible(True)
        self.statusBar().showMessage(f"Procesando... {clamped_value}%", 0)

    def _finalizar_estado_procesamiento(self) -> None:
        self._toggle_controls(True)
        self.statusBar().setStyleSheet(self._default_status_style)
        self._progress_bar.setVisible(False)

    def _limpiar_hilo_proceso(self) -> None:
        if self._process_worker:
            self._process_worker.deleteLater()
            self._process_worker = None
        if self._process_thread:
            self._process_thread.deleteLater()
            self._process_thread = None

    def _on_export(self) -> None:
        if not self.resultados:
            QMessageBox.information(self, "Sin datos", "Procese un PDF antes de exportar.")
            return

        save_name, _ = QFileDialog.getSaveFileName(
            self, "Guardar resultados", "resultados.csv", "CSV (*.csv)"
        )
        if not save_name:
            return

        registros = [registro for alumno in self.resultados for registro in alumno.to_records()]
        df = pd.DataFrame(registros)
        df.to_csv(save_name, index=False)
        QMessageBox.information(self, "Exportado", f"Archivo guardado en {save_name}")

    def _on_student_selected(self) -> None:
        if self._syncing_selection:
            return
        selected = self.table_students.selectedIndexes()
        if not selected:
            return
        row = selected[0].row()
        self._mostrar_detalle_por_indice(row)

    def _on_not_found_selected(self) -> None:
        if self._syncing_selection:
            return
        selected = self.table_not_found.selectedIndexes()
        if not selected:
            return
        row = selected[0].row()
        pagina_item = self.table_not_found.item(row, 0)
        if not pagina_item:
            return
        resultado: AlumnoHoja | None = pagina_item.data(Qt.ItemDataRole.UserRole)
        self._mostrar_detalle_alumno(resultado)

    # -------------------------------------------------------- evaluaciones API
    def _load_evaluaciones(self, estado_id: int = 2) -> None:
        """Obtiene las evaluaciones desde el API y llena el desplegable."""

        self.combo_evaluaciones.clear()
        self.combo_evaluaciones.addItem("Cargando evaluaciones...", None)

        url = f"{self.API_BASE.rstrip('/')}/api/EvaluacionProgramadums/estado/{estado_id}"
        try:
            with urlopen(
                Request(url, headers={"Accept": "application/json"}), timeout=10
            ) as response:
                raw_data = response.read()
            payload = json.loads(raw_data)
        except (HTTPError, URLError) as exc:  # pragma: no cover - interacción remota
            self._handle_evaluacion_error(
                f"No se pudo conectar al servicio de evaluaciones ({exc})."
            )
            return
        except json.JSONDecodeError as exc:  # pragma: no cover - validación de datos
            self._handle_evaluacion_error(
                f"Respuesta inválida del servicio de evaluaciones ({exc})."
            )
            return

        if not isinstance(payload, list):
            self._handle_evaluacion_error("Formato inesperado al leer evaluaciones.")
            return

        self.evaluaciones = self._normalize_evaluaciones(payload)
        if not self.evaluaciones:
            self._handle_evaluacion_error("No se encontraron evaluaciones disponibles.")
            return

        self.combo_evaluaciones.clear()
        for evaluacion in self.evaluaciones:
            display = f"{evaluacion['nombre']} - {evaluacion['fecha_inicio']}"
            self.combo_evaluaciones.addItem(display, evaluacion)
        self.statusBar().showMessage("Evaluaciones cargadas", 4000)
        self._reset_secciones()

    def _handle_evaluacion_error(self, message: str) -> None:
        self.combo_evaluaciones.clear()
        self.combo_evaluaciones.addItem("No se pudieron cargar las evaluaciones", None)
        self.statusBar().showMessage(message, 5000)
        self._reset_resultados()

    def _handle_detalle_error(self, message: str) -> None:
        QMessageBox.warning(self, "Consulta de evaluación", message)
        self.statusBar().showMessage(message, 5000)

    def _normalize_evaluaciones(self, payload: List[dict]) -> List[dict]:
        normalizadas: List[dict] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            nombre = item.get("nombre") or item.get("Nombre")
            if not nombre:
                continue
            evaluacion_id = (
                item.get("evaluacionId")
                or item.get("EvaluacionId")
                or item.get("evaluacion_id")
            )
            fecha_val = (
                item.get("fechaInicio")
                or item.get("fecha_inicio")
                or item.get("FechaInicio")
            )
            fecha_texto = self._format_fecha(fecha_val)
            normalizadas.append(
                {
                    "id": item.get("id") or item.get("Id"),
                    "nombre": nombre,
                    "fecha_inicio": fecha_texto,
                    "evaluacionId": evaluacion_id,
                    "version": item.get("version") or item.get("Version"),
                }
            )
        return normalizadas

    def _format_fecha(self, fecha: object) -> str:
        if fecha is None:
            return "Sin fecha"
        if isinstance(fecha, datetime):
            return fecha.date().strftime("%d/%m/%Y")
        if isinstance(fecha, date):
            return fecha.strftime("%d/%m/%Y")
        if isinstance(fecha, str):
            try:
                return date.fromisoformat(fecha).strftime("%d/%m/%Y")
            except ValueError:
                return fecha
        return str(fecha)

    def _on_evaluacion_changed(self, index: int) -> None:
        if index < 0:
            return
        data = self.combo_evaluaciones.currentData()
        if not data or not data.get("id"):
            self._reset_secciones()
            self._llenar_tabla_evaluacion([])
            self._reset_resultados()
            return
        self._consultar_evaluacion_programada(int(data["id"]))

    def _consultar_evaluacion_programada(self, evaluacion_programada_id: int) -> None:
        """Consulta el API por alumnos asociados a la evaluación seleccionada."""

        url = f"{self.API_BASE.rstrip('/')}/consulta/evaluaciones-programadas/{evaluacion_programada_id}"
        #mostrar mensaje url
        self.statusBar().showMessage(f"Consultando: {url}", 0)
        


        try:
            with urlopen(
                Request(url, headers={"Accept": "application/json"}), timeout=10
            ) as response:
                raw_data = response.read()
            payload = json.loads(raw_data)
        except (HTTPError, URLError) as exc:  # pragma: no cover - interacción remota
            self._handle_detalle_error(
                f"No se pudo consultar la evaluación seleccionada ({exc})."
            )
            return
        except json.JSONDecodeError as exc:  # pragma: no cover - validación remota
            self._handle_detalle_error(
                f"Respuesta inválida del servicio de consulta ({exc})."
            )
            return

        if not isinstance(payload, list):
            self._handle_detalle_error("Formato inesperado al leer los alumnos.")
            return

        self.evaluacion_detalle = self._normalize_evaluacion_detalle(payload)
        if not self.evaluacion_detalle:
            self._handle_detalle_error("No se encontraron alumnos para la evaluación.")
            self._reset_secciones()
            self._llenar_tabla_evaluacion([])
            return

        self._reset_resultados()

        self._poblar_secciones()
        self._llenar_tabla_evaluacion([])
        self.statusBar().showMessage("Evaluación cargada correctamente", 5000)

    def _normalize_evaluacion_detalle(self, payload: List[dict]) -> List[dict]:
        normalizadas: List[dict] = []
        for idx, item in enumerate(payload, start=1):
            if not isinstance(item, dict):
                continue

            # Soportar ambas variantes de nombre
            apellidos = item.get("AlumnoApellidos") or item.get("alumnoApellidos") or ""
            nombres = item.get("AlumnoNombres") or item.get("alumnoNombres") or ""
            nombre_completo = f"{apellidos} {nombres}".strip()

            normalizadas.append(
                {
                    "orden": idx,
                    "pagina": "",
                    "dni": item.get("AlumnoDni") or item.get("alumnoDni") or "",
                    "alumno": nombre_completo,
                    "ciclo": item.get("Ciclo") or item.get("ciclo") or "",
                    "seccion": item.get("Seccion") or item.get("seccion") or "",
                    "evaluacionId": item.get("evaluacionId")
                    or item.get("EvaluacionId")
                    or item.get("evaluacion_id"),
                    "evaluacionProgramadaId": item.get("evaluacionProgramadaId")
                    or item.get("EvaluacionProgramadaId"),
                    "seccionId": item.get("SeccionId")
                    or item.get("seccionId")
                    or item.get("seccion_id"),
                    "alumnoId": item.get("AlumnoId")
                    or item.get("alumnoId")
                    or item.get("alumno_id"),
                    "version": item.get("Version")
                    or item.get("version")
                    or item.get("version_respuesta"),
                }
            )
        return normalizadas


    def _poblar_secciones(self) -> None:
        secciones: list[str] = []
        vistos: set[str] = set()
        for item in self.evaluacion_detalle:
            nombre = (item.get("seccion") or "").strip()
            if nombre and nombre not in vistos:
                vistos.add(nombre)
                secciones.append(nombre)

        secciones.sort(key=str.casefold)

        self.combo_secciones.blockSignals(True)
        self.combo_secciones.clear()
        self.combo_secciones.addItem("Seleccione sección", None)
        
        #if secciones:
        #   self.combo_secciones.addItem("Todas las secciones", self.ALL_SECTIONS_KEY)
            
        for sec in secciones:
            self.combo_secciones.addItem(sec, sec)
        self.combo_secciones.setCurrentIndex(0)
        self.combo_secciones.blockSignals(False)

    def _reset_secciones(self) -> None:
        self.combo_secciones.blockSignals(True)
        self.combo_secciones.clear()
        self.combo_secciones.addItem("Sin secciones", None)
        self.combo_secciones.blockSignals(False)
        self._llenar_tabla_evaluacion([])

    def _on_seccion_changed(self, index: int) -> None:
        if index < 0:
            return
        filtro = self.combo_secciones.currentData()
        if filtro is None:
            self._llenar_tabla_evaluacion([])
            return
        self._refrescar_tabla_filtrada()

    # ------------------------------------------------------------- helpers UI
    def _llenar_tabla_evaluacion(self, alumnos: List[dict]) -> None:
        self.table_students.setRowCount(0)
        for alumno in alumnos:
            row = self.table_students.rowCount()
            self.table_students.insertRow(row)
            self.table_students.setItem(
                row, 0, QTableWidgetItem(str(alumno.get("pagina", "")))
            )
            self.table_students.setItem(row, 1, QTableWidgetItem(alumno.get("dni", "")))
            self.table_students.setItem(
                row, 2, QTableWidgetItem(alumno.get("alumno", ""))
            )
            self.table_students.setItem(row, 3, QTableWidgetItem(alumno.get("ciclo", "")))
            self.table_students.setItem(row, 4, QTableWidgetItem(alumno.get("seccion", "")))
        if alumnos:
            self.table_students.selectRow(0)

    def _refrescar_tabla_filtrada(self) -> None:
        filtro = self.combo_secciones.currentData()
        datos = self.evaluacion_detalle
        if filtro is None:
            datos = []
        elif filtro != self.ALL_SECTIONS_KEY:
            datos = [item for item in self.evaluacion_detalle if item.get("seccion") == filtro]
        self._llenar_tabla_evaluacion(datos)

    def _reset_resultados(self) -> None:
        self.resultados = []
        self.resultados_por_dni.clear()
        self.resultados_no_asignados = []
        self._sincronizando_no_encontrados = True
        self.table_not_found.setRowCount(0)
        self._sincronizando_no_encontrados = False
        self.table_answers.setRowCount(0)
        self.image_label.clear()
        self.image_label.setText("Seleccione un alumno")

    def _integrar_resultados_en_tablas(self) -> None:
        alumnos_por_dni = {
            (item.get("dni") or "").strip(): item
            for item in self.evaluacion_detalle
            if item.get("dni")
        }

        for item in self.evaluacion_detalle:
            item["pagina"] = item.get("pagina") or ""

        self.resultados_por_dni.clear()
        self.resultados_no_asignados = []

        for resultado in self.resultados:
            dni_normalizado = resultado.dni.strip()
            if dni_normalizado and dni_normalizado in alumnos_por_dni:
                alumnos_por_dni[dni_normalizado]["pagina"] = resultado.pagina
                self.resultados_por_dni[dni_normalizado] = resultado
            else:
                self.resultados_no_asignados.append(resultado)

        self._poblar_no_encontrados()
        self._refrescar_tabla_filtrada()

    def _poblar_no_encontrados(self) -> None:
        self._sincronizando_no_encontrados = True
        self.table_not_found.setRowCount(0)
        for resultado in self.resultados_no_asignados:
            row = self.table_not_found.rowCount()
            self.table_not_found.insertRow(row)
            pagina_item = QTableWidgetItem(str(resultado.pagina))
            pagina_item.setData(Qt.ItemDataRole.UserRole, resultado)
            pagina_item.setFlags(pagina_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            dni_item = QTableWidgetItem(resultado.dni)
            dni_item.setData(Qt.ItemDataRole.UserRole, resultado)
            self.table_not_found.setItem(row, 0, pagina_item)
            self.table_not_found.setItem(row, 1, dni_item)
        self._sincronizando_no_encontrados = False

    def _on_not_found_item_changed(self, item: QTableWidgetItem) -> None:
        if self._sincronizando_no_encontrados:
            return
        if item.column() != 1:
            return

        fila = item.row()
        pagina_item = self.table_not_found.item(fila, 0)
        if not pagina_item:
            return

        resultado: AlumnoHoja | None = pagina_item.data(Qt.ItemDataRole.UserRole)
        if resultado is None:
            return

        nuevo_dni = item.text().strip()
        if not nuevo_dni:
            return

        match = next(
            (al for al in self.evaluacion_detalle if al.get("dni") == nuevo_dni), None
        )
        if not match:
            return

        resultado.dni = nuevo_dni
        match["pagina"] = resultado.pagina
        self.resultados_por_dni[nuevo_dni] = resultado

        self._sincronizando_no_encontrados = True
        self.table_not_found.removeRow(fila)
        self._sincronizando_no_encontrados = False

        try:
            self.resultados_no_asignados.remove(resultado)
        except ValueError:
            pass

        self._refrescar_tabla_filtrada()

    # ----------------------------------------------------- registro respuestas
    def _on_register_responses(self) -> None:
        evaluacion_data = self.combo_evaluaciones.currentData()
        if not evaluacion_data or not evaluacion_data.get("id"):
            QMessageBox.warning(
                self,
                "Evaluación requerida",
                "Seleccione una evaluación antes de registrar respuestas.",
            )
            return

        if self.table_students.rowCount() == 0:
            QMessageBox.warning(
                self,
                "Sin alumnos",
                "No hay alumnos cargados para registrar respuestas.",
            )
            return

        filas_validas = self._validar_tabla_alumnos()
        if not filas_validas:
            QMessageBox.warning(
                self,
                "Datos incompletos",
                "Complete página, DNI y nombre en todas las filas resaltadas.",
            )
            return

        payload, advertencias, alumnos_incluidos = self._construir_payload_respuestas(
            int(evaluacion_data["id"]), evaluacion_data
        )

        if not payload:
            resumen = "No se encontraron respuestas escaneadas para registrar."
            if advertencias:
                resumen = f"{resumen}\n\n" + "\n".join(advertencias)
            QMessageBox.warning(self, "Sin respuestas", resumen)
            return

        self.btn_register.setEnabled(False)
        try:
            existen_previas = self._existen_respuestas_previas(
                int(evaluacion_data["id"])
            )
            if existen_previas is None:
                return
            if existen_previas:
                decision = QMessageBox.question(
                    self,
                    "Reemplazar respuestas",
                    "Ya existen respuestas registradas para esta evaluación.\n"
                    "¿Desea reemplazarlas con las respuestas escaneadas?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if decision != QMessageBox.StandardButton.Yes:
                    return
                if not self._eliminar_respuestas_previas(int(evaluacion_data["id"])):
                    return

            registrados = self._enviar_respuestas(payload)
            if registrados == 0:
                QMessageBox.warning(
                    self,
                    "Registro incompleto",
                    "No se pudo registrar ninguna respuesta.",
                )
                return

            resumen = (
                f"Se registraron respuestas para {len(alumnos_incluidos)} alumno(s)."
            )
            if advertencias:
                resumen = f"{resumen}\n\nAdvertencias:\n- " + "\n- ".join(
                    advertencias
                )
            QMessageBox.information(self, "Registro finalizado", resumen)
        finally:
            self.btn_register.setEnabled(True)

    def _validar_tabla_alumnos(self) -> bool:
        valido = True
        color_error = QColor("#fecdd3")
        color_base = self.table_students.palette().base().color()
        for row in range(self.table_students.rowCount()):
            pagina_item = self.table_students.item(row, 0)
            dni_item = self.table_students.item(row, 1)
            nombre_item = self.table_students.item(row, 2)

            fila_valida = True

            for col in range(self.table_students.columnCount()):
                item = self.table_students.item(row, col)
                if item:
                    item.setBackground(color_base)

            if not pagina_item or not pagina_item.text().strip():
                fila_valida = False
            if not dni_item or not dni_item.text().strip():
                fila_valida = False
            if not nombre_item or not nombre_item.text().strip():
                fila_valida = False

            if not fila_valida:
                valido = False
                for col in range(self.table_students.columnCount()):
                    item = self.table_students.item(row, col)
                    if item:
                        item.setBackground(color_error)
        return valido

    def _construir_payload_respuestas(
        self, evaluacion_programada_id: int, evaluacion_data: dict
    ) -> Tuple[List[dict], List[str], Set[str]]:
        payload: List[dict] = []
        advertencias: List[str] = []
        alumnos_incluidos: Set[str] = set()
        fecha_envio = datetime.now().isoformat()

        for row in range(self.table_students.rowCount()):
            dni = (self.table_students.item(row, 1) or QTableWidgetItem(""))
            nombre = (self.table_students.item(row, 2) or QTableWidgetItem(""))
            pagina_item = self.table_students.item(row, 0)
            dni_valor = dni.text().strip()
            nombre_valor = nombre.text().strip()
            pagina_valor = pagina_item.text().strip() if pagina_item else ""

            if not dni_valor:
                advertencias.append(
                    f"La fila {row + 1} no tiene DNI asignado en la evaluación."
                )
                continue

            detalle_alumno = next(
                (item for item in self.evaluacion_detalle if item.get("dni") == dni_valor),
                None,
            )
            if detalle_alumno is None:
                advertencias.append(
                    f"El DNI {dni_valor} no pertenece a la evaluación seleccionada."
                )
                continue

            hoja = self.resultados_por_dni.get(dni_valor)
            if hoja is None:
                advertencias.append(
                    f"No se encontraron respuestas escaneadas para el DNI {dni_valor}."
                )
                continue

            if not hoja.respuestas:
                advertencias.append(
                    f"La hoja del DNI {dni_valor} no contiene respuestas detectadas."
                )
                continue

            if pagina_valor and str(hoja.pagina) != pagina_valor:
                advertencias.append(
                    f"La página indicada ({pagina_valor}) para {dni_valor} no coincide con la hoja escaneada ({hoja.pagina})."
                )

            for respuesta in hoja.respuestas:
                payload.append(
                    {
                        "evaluacionId": (
                            detalle_alumno.get("evaluacionId")
                            or detalle_alumno.get("evaluacion_id")
                            or evaluacion_data.get("evaluacionId")
                            or evaluacion_data.get("evaluacion_id")
                        ),
                        "evaluacionProgramadaId": evaluacion_programada_id,
                        "seccionId": detalle_alumno.get("seccionId")
                        or detalle_alumno.get("seccion_id"),
                        "alumnoId": detalle_alumno.get("alumnoId")
                        or detalle_alumno.get("alumno_id"),
                        "dniAlumno": dni_valor,
                        "version": detalle_alumno.get("version")
                        or detalle_alumno.get("Version")
                        or evaluacion_data.get("version")
                        or 1,
                        "preguntaOrden": respuesta.pregunta,
                        "respuesta": None
                        if respuesta.alternativa.strip() == "-"
                        else respuesta.alternativa,
                        "fuente": "omr",
                        "fechaRegistro": fecha_envio,
                        "activo": True,
                    }
                )
            alumnos_incluidos.add(dni_valor)

        return payload, advertencias, alumnos_incluidos

    def _existen_respuestas_previas(self, evaluacion_programada_id: int) -> Optional[bool]:
        url = f"{self.API_BASE.rstrip('/')}/api/EvaluacionRespuestums/ByEvaluacionProgramada/{evaluacion_programada_id}"
        try:
            with urlopen(Request(url, headers={"Accept": "application/json"}), timeout=10) as response:
                response.read()
            return True
        except HTTPError as exc:  # pragma: no cover - interacción remota
            if exc.code == 404:
                return False
            QMessageBox.critical(
                self,
                "Error de consulta",
                f"No se pudo verificar respuestas previas ({exc}).",
            )
        except URLError as exc:  # pragma: no cover - interacción remota
            QMessageBox.critical(
                self,
                "Error de conexión",
                f"No se pudo verificar respuestas previas ({exc}).",
            )
        return None

    def _eliminar_respuestas_previas(self, evaluacion_programada_id: int) -> bool:
        url = f"{self.API_BASE.rstrip('/')}/api/EvaluacionRespuestums/ByEvaluacionProgramada/{evaluacion_programada_id}"
        try:
            with urlopen(Request(url, method="DELETE"), timeout=10) as response:
                response.read()
            return True
        except (HTTPError, URLError) as exc:  # pragma: no cover - interacción remota
            QMessageBox.critical(
                self,
                "Error eliminando",
                f"No se pudieron eliminar respuestas previas ({exc}).",
            )
            return False

    def _enviar_respuestas(self, payload: List[dict]) -> int:
        url = f"{self.API_BASE.rstrip('/')}/api/EvaluacionRespuestums"
        registrados = 0
        for item in payload:
            data = json.dumps(item).encode("utf-8")
            try:
                with urlopen(
                    Request(
                        url,
                        data=data,
                        headers={"Content-Type": "application/json"},
                    ),
                    timeout=10,
                ) as response:
                    response.read()
                registrados += 1
            except (HTTPError, URLError) as exc:  # pragma: no cover - interacción remota
                QMessageBox.critical(
                    self,
                    "Error de registro",
                    f"No se pudo registrar una respuesta (pregunta {item.get('preguntaOrden')}): {exc}",
                )
                break
        return registrados

    def _mostrar_detalle_por_indice(self, index: int) -> None:
        if index < 0 or index >= self.table_students.rowCount():
            return
        dni_item = self.table_students.item(index, 1)
        if not dni_item:
            return
        dni = dni_item.text().strip()
        alumno = self.resultados_por_dni.get(dni)
        self._mostrar_detalle_alumno(alumno)

    def _mostrar_detalle_alumno(self, alumno: AlumnoHoja | None) -> None:
        if not alumno:
            self._mostrar_imagen(None)
            self._llenar_tabla_respuestas([])
            return
        try:
            pagina = int(alumno.pagina)
        except (ValueError, TypeError):
            pagina = 1

        self._mostrar_pagina_pdf(pagina)
        self._llenar_tabla_respuestas(alumno.respuestas)

    def _mostrar_pagina_pdf(self, pagina: int) -> None:
        """Renderiza la página del PDF original y la muestra en image_label."""
        if not self._pdf_doc:
            self.image_label.clear()
            self.image_label.setText("PDF no cargado")
            self._toggle_pdf_controls(False)
            return

        if pagina < 1 or pagina > self._pdf_doc.page_count:
            self.image_label.clear()
            self.image_label.setText(f"Página {pagina} fuera de rango")
            return

        self._current_page = pagina
        self.page_selector.blockSignals(True)
        self.page_selector.setValue(pagina)
        self.page_selector.blockSignals(False)
        self._toggle_pdf_controls(True)

        self._update_page_label(pagina, self._pdf_doc.page_count)
        self._update_zoom_label()

        page = self._pdf_doc[pagina - 1]

        zoom = (self.PREVIEW_DPI / 72) * self._zoom_factor
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        if pix.alpha:
            fmt = QImage.Format.Format_RGBA8888
        else:
            fmt = QImage.Format.Format_RGB888

        qimg = QImage(pix.samples, pix.width, pix.height, pix.stride, fmt).copy()
        pixmap = QPixmap.fromImage(qimg)

        self.image_label.setPixmap(pixmap)
        self.image_label.setFixedSize(pixmap.size())
        self.image_label.adjustSize()
        self.image_label.update()
        self._resaltar_pagina_en_tablas(pagina)

    def _mostrar_imagen(self, path: Path | None) -> None:
        if not path or not path.exists():
            self.image_label.clear()
            self.image_label.setText("Imagen no disponible")
            return
        pix = QPixmap(str(path))
        if pix.isNull():
            self.image_label.setText("No se pudo cargar la imagen")
            return
        target_size = self.image_label.size()
        if target_size.isEmpty():
            self.image_label.setPixmap(pix)
            return
        self.image_label.setPixmap(
            pix.scaled(
                target_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _update_page_label(self, current: int, total: int) -> None:
        self.lbl_page_info.setText(f"Página {current} / {total}")
        self.lbl_current_page.setText(str(current))
        self.lbl_total_pages.setText(str(total))

    def _update_zoom_label(self) -> None:
        self.lbl_zoom_info.setText(f"{int(self._zoom_factor * 100)} %")

    def _adjust_zoom(self, delta: float) -> None:
        new_zoom = min(self.MAX_ZOOM, max(self.MIN_ZOOM, self._zoom_factor + delta))
        if abs(new_zoom - self._zoom_factor) < 1e-3:
            return
        self._zoom_factor = new_zoom
        self._update_zoom_label()
        self._render_current_page()

    def _handle_wheel_zoom(self, event: QEvent) -> bool:
        if not self._pdf_doc:
            return False
        wheel_event = event if isinstance(event, QWheelEvent) else None
        if not wheel_event:
            return False
        delta_y = wheel_event.angleDelta().y()
        if delta_y == 0:
            return False
        step_multiplier = 1 if delta_y > 0 else -1
        self._adjust_zoom(self.ZOOM_STEP * step_multiplier)
        wheel_event.accept()
        return True

    def _calculate_fit_zoom(self, pagina: int | None = None) -> float:
        if not self._pdf_doc:
            return 1.0
        if pagina is None:
            pagina = self._current_page
        pagina = max(1, min(self._pdf_doc.page_count, pagina))
        viewport = self.image_scroll.viewport()
        available_width = max(1, viewport.width() - 24)
        available_height = max(1, viewport.height() - 24)
        page = self._pdf_doc[pagina - 1]
        base_scale = self.PREVIEW_DPI / 72
        base_width = page.rect.width * base_scale
        base_height = page.rect.height * base_scale
        width_ratio = available_width / base_width if base_width else 1.0
        height_ratio = available_height / base_height if base_height else 1.0
        fit_zoom = max(self.MIN_ZOOM, min(width_ratio, height_ratio))
        return min(self.MAX_ZOOM, fit_zoom)

    def _render_current_page(self) -> None:
        if not self._pdf_doc:
            return
        self._mostrar_pagina_pdf(self._current_page)

    def _on_prev_page(self) -> None:
        if not self._pdf_doc:
            return
        target = max(1, self._current_page - 1)
        if target == self._current_page:
            return
        self._mostrar_pagina_pdf(target)

    def _on_next_page(self) -> None:
        if not self._pdf_doc:
            return
        target = min(self._pdf_doc.page_count, self._current_page + 1)
        if target == self._current_page:
            return
        self._mostrar_pagina_pdf(target)

    def _on_page_selected(self, value: int) -> None:
        if not self._pdf_doc:
            return
        if value < 1 or value > self._pdf_doc.page_count:
            return
        if value == self._current_page:
            return
        self._mostrar_pagina_pdf(value)

    def _on_zoom_in(self) -> None:
        self._adjust_zoom(self.ZOOM_STEP)

    def _on_zoom_out(self) -> None:
        self._adjust_zoom(-self.ZOOM_STEP)

    def _on_reset_zoom(self) -> None:
        target_zoom = self._calculate_fit_zoom()
        if abs(self._zoom_factor - target_zoom) < 1e-3:
            return
        self._zoom_factor = target_zoom
        self._update_zoom_label()
        self._render_current_page()

    def resizeEvent(self, event) -> None:  # pragma: no cover - actualización visual
        super().resizeEvent(event)
        pixmap = self.image_label.pixmap()
        if not pixmap or pixmap.isNull():
            return
        if self._pdf_doc:
            # Mantener el zoom actual de la vista previa sin forzar escalado.
            self.image_label.setFixedSize(pixmap.size())
            return
        target_size = self.image_label.size()
        if target_size.isEmpty():
            return
        self.image_label.setPixmap(
            pixmap.scaled(
                target_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _resaltar_pagina_en_tablas(self, pagina: int) -> None:
        if self._syncing_selection:
            return

        def _buscar_en_tabla(tabla: QTableWidget) -> int | None:
            for row in range(tabla.rowCount()):
                item = tabla.item(row, 0)
                if not item:
                    continue
                try:
                    item_page = int(item.text())
                except ValueError:
                    continue
                if item_page == pagina:
                    return row
            return None

        self._syncing_selection = True
        try:
            self.table_students.blockSignals(True)
            self.table_not_found.blockSignals(True)
            student_row = _buscar_en_tabla(self.table_students)
            self.table_students.clearSelection()
            self.table_not_found.clearSelection()
            resultado: AlumnoHoja | None = None
            if student_row is not None:
                self.table_students.selectRow(student_row)
                self.table_students.scrollToItem(
                    self.table_students.item(student_row, 0)
                )
                resultado = self._resultado_por_fila(self.table_students, student_row)
            else:
                nf_row = _buscar_en_tabla(self.table_not_found)
                if nf_row is not None:
                    self.table_not_found.selectRow(nf_row)
                    self.table_not_found.scrollToItem(
                        self.table_not_found.item(nf_row, 0)
                    )
                    resultado = self._resultado_por_fila(
                        self.table_not_found, nf_row
                    )
            self._sincronizar_respuestas_por_pagina(pagina, resultado)
        finally:
            self.table_students.blockSignals(False)
            self.table_not_found.blockSignals(False)
            self._syncing_selection = False

    def _resultado_por_fila(
        self, tabla: QTableWidget, fila: int
    ) -> AlumnoHoja | None:
        if tabla is self.table_students:
            dni_item = tabla.item(fila, 1)
            dni = dni_item.text().strip() if dni_item else ""
            return self.resultados_por_dni.get(dni)
        if tabla is self.table_not_found:
            pagina_item = tabla.item(fila, 0)
            if pagina_item:
                resultado = pagina_item.data(Qt.ItemDataRole.UserRole)
                if isinstance(resultado, AlumnoHoja):
                    return resultado
        return None

    def _sincronizar_respuestas_por_pagina(
        self, pagina: int, resultado: AlumnoHoja | None = None
    ) -> None:
        objetivo = resultado or self._buscar_resultado_por_pagina(pagina)
        if objetivo:
            self._llenar_tabla_respuestas(objetivo.respuestas)
        else:
            self._llenar_tabla_respuestas([])

    def _buscar_resultado_por_pagina(self, pagina: int) -> AlumnoHoja | None:
        for resultado in self.resultados:
            try:
                pagina_resultado = int(resultado.pagina)
            except (TypeError, ValueError):
                continue
            if pagina_resultado == pagina:
                return resultado
        return None

    def _llenar_tabla_respuestas(self, respuestas: List[Respuesta]) -> None:
        self.table_answers.setRowCount(0)
        for resp in respuestas:
            row = self.table_answers.rowCount()
            self.table_answers.insertRow(row)
            self.table_answers.setItem(row, 0, QTableWidgetItem(str(resp.pregunta)))
            self.table_answers.setItem(row, 1, QTableWidgetItem(resp.alternativa))
            self.table_answers.setItem(row, 2, QTableWidgetItem(resp.estado))


__all__ = ["MainWindow"]

