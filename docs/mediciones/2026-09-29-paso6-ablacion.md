# Ablación de la tolerancia (Paso 6)

Cada sesión de diagnóstico reproducida por la máquina de estados con cada variante, a la tasa medida al arrancar la sesión y con su reloj real. Por intento guiado: **entero** = un solo trazo entregado y ningún corte; **tras corte** = un solo trazo, pero después de que un hueco sin rellenar cortara otro (el trazo entregado es un pedazo); **partido** = más de uno; **perdido** = ninguno; **cortes** = trazos interrumpidos por un hueco; **frames** = mediana del largo de los trazos entregados. Sin clasificador: se mide la segmentación. La X de estas sesiones es la de la definición anterior.


- **antes**: {'plausibility.enabled': False, 'smoothing.enabled': False, 'segmentation.dynamic_max_gap_ms': 150.0, 'segmentation.tracking_max_gap_ms': 0.0, 'segmentation.stable_max_gap_ms': 0.0}
- **+plausibilidad**: {'plausibility.enabled': True, 'smoothing.enabled': False, 'segmentation.dynamic_max_gap_ms': 150.0, 'segmentation.tracking_max_gap_ms': 0.0, 'segmentation.stable_max_gap_ms': 0.0}
- **+one_euro**: {'plausibility.enabled': False, 'smoothing.enabled': True, 'segmentation.dynamic_max_gap_ms': 150.0, 'segmentation.tracking_max_gap_ms': 0.0, 'segmentation.stable_max_gap_ms': 0.0, 'smoothing.min_cutoff': 0.5, 'smoothing.beta': 1.0, 'smoothing.d_cutoff': 2.0}
- **+relleno**: {'plausibility.enabled': False, 'smoothing.enabled': False, 'segmentation.dynamic_max_gap_ms': 200.0, 'segmentation.tracking_max_gap_ms': 200.0, 'segmentation.stable_max_gap_ms': 100.0}
- **plaus+relleno**: {'plausibility.enabled': True, 'smoothing.enabled': False, 'segmentation.dynamic_max_gap_ms': 200.0, 'segmentation.tracking_max_gap_ms': 200.0, 'segmentation.stable_max_gap_ms': 100.0}
- **todo**: {'plausibility.enabled': True, 'smoothing.enabled': True, 'segmentation.dynamic_max_gap_ms': 200.0, 'segmentation.tracking_max_gap_ms': 200.0, 'segmentation.stable_max_gap_ms': 100.0, 'smoothing.min_cutoff': 0.5, 'smoothing.beta': 1.0, 'smoothing.d_cutoff': 2.0}

| letra | variante | intentos | enteros | tras corte | partidos | perdidos | cortes | frames |
|---|---|---|---|---|---|---|---|---|
| X | antes | 93 | 12 (13%) | 7 | 3 | 71 | 48 | 49 |
| X | +plausibilidad | 93 | 12 (13%) | 7 | 3 | 71 | 53 | 46 |
| X | +one_euro | 93 | 19 (20%) | 5 | 4 | 65 | 38 | 46 |
| X | +relleno | 93 | 12 (13%) | 7 | 3 | 71 | 52 | 50 |
| X | plaus+relleno | 93 | 12 (13%) | 7 | 3 | 71 | 55 | 49 |
| X | todo | 93 | 19 (20%) | 5 | 4 | 65 | 47 | 50 |
| ENIE | antes | 101 | 39 (39%) | 9 | 10 | 43 | 45 | 45 |
| ENIE | +plausibilidad | 101 | 39 (39%) | 9 | 10 | 43 | 46 | 45 |
| ENIE | +one_euro | 101 | 37 (37%) | 9 | 11 | 44 | 46 | 43 |
| ENIE | +relleno | 101 | 41 (41%) | 7 | 10 | 43 | 42 | 45 |
| ENIE | plaus+relleno | 101 | 41 (41%) | 7 | 10 | 43 | 42 | 45 |
| ENIE | todo | 101 | 39 (39%) | 7 | 11 | 44 | 43 | 43 |
| Q | antes | 111 | 5 (5%) | 3 | 9 | 94 | 73 | 46 |
| Q | +plausibilidad | 111 | 5 (5%) | 3 | 9 | 94 | 82 | 46 |
| Q | +one_euro | 111 | 9 (8%) | 2 | 10 | 90 | 69 | 45 |
| Q | +relleno | 111 | 5 (5%) | 3 | 9 | 94 | 75 | 45 |
| Q | plaus+relleno | 111 | 5 (5%) | 3 | 9 | 94 | 80 | 46 |
| Q | todo | 111 | 8 (7%) | 4 | 10 | 89 | 74 | 47 |
| K | antes | 119 | 26 (22%) | 13 | 27 | 53 | 57 | 53 |
| K | +plausibilidad | 119 | 26 (22%) | 14 | 26 | 53 | 58 | 53 |
| K | +one_euro | 119 | 26 (22%) | 16 | 29 | 48 | 57 | 52 |
| K | +relleno | 119 | 26 (22%) | 13 | 27 | 53 | 60 | 55 |
| K | plaus+relleno | 119 | 26 (22%) | 13 | 27 | 53 | 59 | 55 |
| K | todo | 119 | 26 (22%) | 16 | 29 | 48 | 57 | 53 |
| J | antes | 147 | 29 (20%) | 17 | 19 | 82 | 233 | 46 |
| J | +plausibilidad | 147 | 29 (20%) | 17 | 19 | 82 | 243 | 46 |
| J | +one_euro | 147 | 32 (22%) | 18 | 20 | 77 | 219 | 45 |
| J | +relleno | 147 | 31 (21%) | 15 | 19 | 82 | 233 | 49 |
| J | plaus+relleno | 147 | 31 (21%) | 15 | 19 | 82 | 252 | 49 |
| J | todo | 147 | 33 (22%) | 16 | 20 | 78 | 236 | 50 |
| Z | antes | 96 | 29 (30%) | 6 | 8 | 53 | 51 | 50 |
| Z | +plausibilidad | 96 | 30 (31%) | 5 | 8 | 53 | 48 | 50 |
| Z | +one_euro | 96 | 31 (32%) | 5 | 8 | 52 | 40 | 49 |
| Z | +relleno | 96 | 29 (30%) | 6 | 8 | 53 | 55 | 51 |
| Z | plaus+relleno | 96 | 29 (30%) | 6 | 8 | 53 | 54 | 51 |
| Z | todo | 96 | 31 (32%) | 5 | 8 | 52 | 42 | 50 |
