"""Definición de la ventana principal basada en PyQt6."""

from __future__ import annotations

import json
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QComboBox,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QProgressBar,
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
            resultados = procesar_pdf(
                self.pdf_path,
                self.cache_dir,
                self.config,
                progress_callback=self.progress.emit,
            )
        except Exception as exc:  # pragma: no cover - mostrado en UI
            self.error.emit(str(exc))
            return
        self.finished.emit(resultados)


class MainWindow(QMainWindow):
    """Ventana principal con estilo moderno y paneles divididos."""

    API_BASE = "http://192.168.1.50:5000"
    ALL_SECTIONS_KEY = "__all__"

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
        self._load_evaluaciones()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)

        # Barra de acciones
        toolbar = QHBoxLayout()
        self.combo_evaluaciones = QComboBox()
        self.combo_evaluaciones.setPlaceholderText("Seleccione evaluación")
        self.combo_evaluaciones.setMinimumWidth(280)
        self.combo_secciones = QComboBox()
        self.combo_secciones.setPlaceholderText("Seleccione sección")
        self.combo_secciones.setMinimumWidth(200)
        self.btn_load = QPushButton("Cargar PDF")
        self.btn_process = QPushButton("Procesar")
        self.btn_export = QPushButton("Exportar resultados")
        self.lbl_file = QLabel("Ningún archivo seleccionado")

        toolbar.addWidget(self.combo_evaluaciones)
        toolbar.addWidget(self.combo_secciones)
        toolbar.addWidget(self.btn_load)
        toolbar.addWidget(self.btn_process)
        toolbar.addWidget(self.btn_export)
        toolbar.addStretch(1)
        toolbar.addWidget(self.lbl_file)

        layout.addLayout(toolbar)

        splitter = QSplitter()
        splitter.setOrientation(Qt.Orientation.Horizontal)

        # Panel izquierdo (alumnos y no encontrados)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        self.table_students = QTableWidget(0, 5)
        self.table_students.setHorizontalHeaderLabels(
            ["Página", "DNI", "Alumno", "Ciclo", "Sección"]
        )
        self.table_students.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_students.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
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
        left_layout.addWidget(self.table_not_found, stretch=1)

        splitter.addWidget(left_panel)

        # Panel derecho (imagen + respuestas)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.image_label = QLabel("Seleccione un alumno")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(350)
        image_scroll = QScrollArea()
        image_scroll.setWidget(self.image_label)
        image_scroll.setWidgetResizable(True)
        right_layout.addWidget(image_scroll, stretch=3)

        self.table_answers = QTableWidget(0, 3)
        self.table_answers.setHorizontalHeaderLabels(["Pregunta", "Respuesta", "Estado"])
        self.table_answers.verticalHeader().setVisible(False)
        self.table_answers.setAlternatingRowColors(True)
        right_layout.addWidget(self.table_answers, stretch=2)

        splitter.addWidget(right_panel)
        layout.addWidget(splitter)

        self.setCentralWidget(central)

        self._apply_styles()
        self._connect_signals()

    def _apply_styles(self) -> None:
        """Aplica una hoja de estilos pastel."""

        self.setStyleSheet(
            """
            QWidget {
                background-color: #f8fafc;
                font-family: 'Segoe UI', 'Open Sans', sans-serif;
                color: #1f1f1f;
            }
            QPushButton {
                background-color: #b8e0d2;
                padding: 10px 18px;
                border-radius: 10px;
                border: 1px solid #a1d2c5;
                font-weight: 600;
                letter-spacing: 0.2px;
            }
            QPushButton:hover {
                background-color: #a5d6c9;
            }
            QPushButton:pressed {
                background-color: #8ac3b4;
            }
            QTableWidget {
                background-color: #ffffff;
                border: 1px solid #dfe7ef;
                border-radius: 12px;
                gridline-color: #d0d9e2;
                selection-background-color: #e4d7ff;
                selection-color: #1f1f1f;
                alternate-background-color: #f4f7fb;
            }
            QHeaderView::section {
                background-color: #d5e8f3;
                border: none;
                padding: 8px;
                font-weight: 600;
                color: #0f172a;
            }
            QLabel#fileLabel {
                color: #4b5563;
            }
            QScrollArea {
                border: none;
            }
            QComboBox {
                padding: 8px 12px;
                border-radius: 10px;
                border: 1px solid #cfd8e3;
                background-color: #ffffff;
                min-width: 260px;
            }
            QComboBox:focus {
                border-color: #a5d6c9;
                box-shadow: 0 0 0 3px rgba(165, 214, 201, 0.35);
            }
            QComboBox QListView {
                background-color: #ffffff;
                border-radius: 10px;
                padding: 4px 0;
            }
            QSplitter::handle {
                background-color: #e1e7ef;
            }
        """
        )
        self.lbl_file.setObjectName("fileLabel")

    def _connect_signals(self) -> None:
        self.btn_load.clicked.connect(self._on_load_pdf)
        self.btn_process.clicked.connect(self._on_process)
        self.btn_export.clicked.connect(self._on_export)
        self.table_students.itemSelectionChanged.connect(self._on_student_selected)
        self.table_not_found.itemSelectionChanged.connect(
            self._on_not_found_selected
        )
        self.combo_evaluaciones.currentIndexChanged.connect(
            self._on_evaluacion_changed
        )
        self.combo_secciones.currentIndexChanged.connect(self._on_seccion_changed)
        self.table_not_found.itemChanged.connect(self._on_not_found_item_changed)

    # --------------------------------------------------------------- acciones
    def _on_load_pdf(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(self, "Seleccionar PDF", "", "PDF (*.pdf)")
        if not file_name:
            return
        self.pdf_path = Path(file_name)
        self.lbl_file.setText(self.pdf_path.name)
        self.statusBar().showMessage(f"Archivo cargado: {self.pdf_path}", 5000)

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
        self.statusBar().showMessage("Procesando...", 0)
        self._toggle_controls(False)

    def _toggle_controls(self, enabled: bool) -> None:
        for widget in (
            self.btn_load,
            self.btn_process,
            self.btn_export,
            self.combo_evaluaciones,
            self.combo_secciones,
        ):
            widget.setEnabled(enabled)

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
        self._progress_bar.setValue(max(0, min(100, value)))

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
        selected = self.table_students.selectedIndexes()
        if not selected:
            return
        row = selected[0].row()
        self._mostrar_detalle_por_indice(row)

    def _on_not_found_selected(self) -> None:
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
        self._mostrar_imagen(alumno.imagen_path)
        self._llenar_tabla_respuestas(alumno.respuestas)

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

    def resizeEvent(self, event) -> None:  # pragma: no cover - actualización visual
        super().resizeEvent(event)
        pixmap = self.image_label.pixmap()
        if not pixmap or pixmap.isNull():
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

    def _llenar_tabla_respuestas(self, respuestas: List[Respuesta]) -> None:
        self.table_answers.setRowCount(0)
        for resp in respuestas:
            row = self.table_answers.rowCount()
            self.table_answers.insertRow(row)
            self.table_answers.setItem(row, 0, QTableWidgetItem(str(resp.pregunta)))
            self.table_answers.setItem(row, 1, QTableWidgetItem(resp.alternativa))
            self.table_answers.setItem(row, 2, QTableWidgetItem(resp.estado))


__all__ = ["MainWindow"]

