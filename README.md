# HRPython

Procesamiento de hojas de respuestas mediante OMR. El flujo completo está
documentado en `omr_processor.py`, pero los puntos clave del algoritmo son:

1. **Sincronización por anclas inferiores**
   - `_detectar_rectangulos_sync` convierte la página a escala de grises,
     conserva únicamente la franja inferior (`sync_band_height_ratio`) y allí
     busca rectángulos negros válidos. Mientras los marcadores sigan dentro de
     esa banda no hace falta capturar la página completa.
2. **Lectura del DNI**
   - La imagen se lleva a grises y se calcula un rango de respaldo usando
     `OMRConfig.dni_vertical_band` (por defecto del 12 % al 64 % del alto).
   - La banda real se extrae con `_banda_vertical_desde_referencias`, que recorta
     horizontalmente entre la primera y la última ancla, aplica
     `x_band_padding_ratio` y proyecta la tinta verticalmente. El segmento con
     mayor energía sobre `profile_threshold_dni`, expandido con
     `profile_margin_ratio`, define `(y0, y1)` para todas las columnas.
3. **Preguntas**
   - Se aplica la misma lógica de banda vertical pero usando
     `profile_threshold_respuestas` para conservar únicamente la zona donde hay
     burbujas.

Este pipeline permite trabajar con escaneos parciales siempre que los
marcadores sigan visibles en la banda inferior configurada.
