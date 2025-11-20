"""Definición de la ventana principal basada en PyQt6."""

from __future__ import annotations

import json
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from PyQt6.QtCore import Qt
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
)

from models import AlumnoHoja, Respuesta
from omr_processor import OMRConfig, procesar_pdf


class MainWindow(QMainWindow):
    """Ventana principal con estilo moderno y paneles divididos."""

    API_BASE = "http://192.168.1.50:5000"

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Lector OMR")
        self.resize(1400, 800)
        self.pdf_path: Path | None = None
        self.cache_dir = Path(tempfile.mkdtemp(prefix="omr_cache_"))
        self.resultados: List[AlumnoHoja] = []
        self.config = OMRConfig()
        self.evaluaciones: List[dict] = []
        self.evaluacion_detalle: List[dict] = []

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
        self.combo_secciones.setPlaceholderText("Todas las secciones")
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

        # Tabla de alumnos
        self.table_students = QTableWidget(0, 6)
        self.table_students.setHorizontalHeaderLabels(
            ["#", "Página", "DNI", "Alumno", "Ciclo", "Sección"]
        )
        self.table_students.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_students.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table_students.verticalHeader().setVisible(False)
        self.table_students.setAlternatingRowColors(True)
        splitter.addWidget(self.table_students)

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
                background-color: #f6f6f6;
                font-family: 'Segoe UI', 'Open Sans', sans-serif;
                color: #1f1f1f;
            }
            QPushButton {
                background-color: #A9D6E5;
                padding: 8px 16px;
                border-radius: 8px;
                border: none;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #90c7d8;
            }
            QPushButton:pressed {
                background-color: #78b8cb;
            }
            QTableWidget {
                background-color: #ffffff;
                border: 1px solid #dcdcdc;
                border-radius: 10px;
                gridline-color: #c8c8c8;
                selection-background-color: #E2D4FF;
                alternate-background-color: #F2FBF7;
            }
            QHeaderView::section {
                background-color: #CDE8D7;
                border: none;
                padding: 6px;
                font-weight: bold;
            }
            QLabel#fileLabel {
                color: #555555;
            }
            QScrollArea {
                border: none;
            }
            QComboBox {
                padding: 6px 10px;
                border-radius: 8px;
                border: 1px solid #c8c8c8;
                background-color: #ffffff;
                min-width: 260px;
            }
            QComboBox:focus {
                border-color: #90c7d8;
            }
            QComboBox QListView {
                background-color: #ffffff;
                border-radius: 8px;
            }
            QSplitter::handle {
                background-color: #d8e2dc;
            }
            """
        )
        self.lbl_file.setObjectName("fileLabel")

    def _connect_signals(self) -> None:
        self.btn_load.clicked.connect(self._on_load_pdf)
        self.btn_process.clicked.connect(self._on_process)
        self.btn_export.clicked.connect(self._on_export)
        self.table_students.itemSelectionChanged.connect(self._on_student_selected)
        self.combo_evaluaciones.currentIndexChanged.connect(
            self._on_evaluacion_changed
        )
        self.combo_secciones.currentIndexChanged.connect(self._on_seccion_changed)

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
        self.statusBar().showMessage("Procesando...", 0)
        try:
            self.resultados = procesar_pdf(self.pdf_path, self.cache_dir, self.config)
        except Exception as exc:  # pragma: no cover - mostrado al usuario
            QMessageBox.critical(self, "Error procesando", str(exc))
            self.statusBar().clearMessage()
            return
        self._llenar_tabla_alumnos()
        self.statusBar().showMessage("Procesamiento finalizado", 5000)

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
        self._mostrar_detalle(row)

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

        self._poblar_secciones()
        self._llenar_tabla_evaluacion(self.evaluacion_detalle)
        self.statusBar().showMessage("Evaluación cargada correctamente", 5000)

    def _normalize_evaluacion_detalle(self, payload: List[dict]) -> List[dict]:
        normalizadas: List[dict] = []
        for idx, item in enumerate(payload, start=1):
            if not isinstance(item, dict):
                continue
            apellidos = item.get("AlumnoApellidos") or ""
            nombres = item.get("AlumnoNombres") or ""
            nombre_completo = f"{apellidos} {nombres}".strip()
            normalizadas.append(
                {
                    "orden": idx,
                    "pagina": item.get("pagina")
                    or item.get("Pagina")
                    or item.get("EvaluacionId")
                    or idx,
                    "dni": item.get("AlumnoDni") or "",
                    "alumno": nombre_completo,
                    "ciclo": item.get("Ciclo") or "",
                    "seccion": item.get("Seccion") or "",
                }
            )
        return normalizadas

    def _poblar_secciones(self) -> None:
        secciones = []
        vistos = set()
        for item in self.evaluacion_detalle:
            nombre = item.get("seccion") or ""
            if nombre and nombre not in vistos:
                vistos.add(nombre)
                secciones.append(nombre)

        self.combo_secciones.blockSignals(True)
        self.combo_secciones.clear()
        self.combo_secciones.addItem("Todas las secciones", None)
        for sec in secciones:
            self.combo_secciones.addItem(sec, sec)
        self.combo_secciones.setCurrentIndex(0)
        self.combo_secciones.blockSignals(False)

    def _reset_secciones(self) -> None:
        self.combo_secciones.blockSignals(True)
        self.combo_secciones.clear()
        self.combo_secciones.addItem("Sin secciones", None)
        self.combo_secciones.blockSignals(False)

    def _on_seccion_changed(self, index: int) -> None:
        if index < 0:
            return
        filtro = self.combo_secciones.currentData()
        datos = self.evaluacion_detalle
        if filtro:
            datos = [item for item in self.evaluacion_detalle if item.get("seccion") == filtro]
        self._llenar_tabla_evaluacion(datos)

    # ------------------------------------------------------------- helpers UI
    def _llenar_tabla_alumnos(self) -> None:
        self.table_students.setRowCount(0)
        for idx, alumno in enumerate(self.resultados, start=1):
            row = self.table_students.rowCount()
            self.table_students.insertRow(row)
            self.table_students.setItem(row, 0, QTableWidgetItem(str(idx)))
            self.table_students.setItem(row, 1, QTableWidgetItem(str(alumno.pagina)))
            self.table_students.setItem(row, 2, QTableWidgetItem(alumno.dni))
            self.table_students.setItem(row, 3, QTableWidgetItem(""))
            self.table_students.setItem(row, 4, QTableWidgetItem(""))
            self.table_students.setItem(row, 5, QTableWidgetItem(""))
        if self.resultados:
            self.table_students.selectRow(0)

    def _llenar_tabla_evaluacion(self, alumnos: List[dict]) -> None:
        self.table_students.setRowCount(0)
        for idx, alumno in enumerate(alumnos, start=1):
            row = self.table_students.rowCount()
            self.table_students.insertRow(row)
            self.table_students.setItem(row, 0, QTableWidgetItem(str(idx)))
            self.table_students.setItem(row, 1, QTableWidgetItem(str(alumno.get("pagina", ""))))
            self.table_students.setItem(row, 2, QTableWidgetItem(alumno.get("dni", "")))
            self.table_students.setItem(row, 3, QTableWidgetItem(alumno.get("alumno", "")))
            self.table_students.setItem(row, 4, QTableWidgetItem(alumno.get("ciclo", "")))
            self.table_students.setItem(row, 5, QTableWidgetItem(alumno.get("seccion", "")))
        if alumnos:
            self.table_students.selectRow(0)
        # Evitar confusión con el detalle de OMR cuando la tabla muestra datos remotos.
        self.resultados = []

    def _mostrar_detalle(self, index: int) -> None:
        if not (0 <= index < len(self.resultados)):
            return
        alumno = self.resultados[index]
        self._mostrar_imagen(alumno.imagen_path)
        self._llenar_tabla_respuestas(alumno.respuestas)

    def _mostrar_imagen(self, path: Path | None) -> None:
        if not path or not path.exists():
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

