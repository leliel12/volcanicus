# Migración a los accessors de vistas

Las cinco vistas de los datos (scenes, descriptors, rays, anomalies,
episodes) ahora son **accessors** del `Volcano`, como `v.plot` y `v.stats`.
Los métodos viejos ya no existen y los resultados son idénticos.

## Tabla de equivalencias

| Antes | Ahora | Atajo |
|---|---|---|
| `v.to_scenes()` | `v.scenes.grid()` | `v.scenes()` |
| `v.to_scenes(as_array=True)` | `v.scenes.grid(as_array=True)` | `v.scenes(True)` |
| `v.reduce_directions("mean")` | `v.scenes.reduce("mean")` | |
| `v.scenes_median` | `v.scenes.median()` | |
| `v.scenes_max` | `v.scenes.max()` | |
| `v.scenes_descriptors()` | `v.descriptors.table()` | `v.descriptors()` |
| `v.rays_descriptors()` | `v.rays.descriptors()` | `v.rays()` |
| `v.flag_rays(limit)` | `v.rays.flag(limit)` | |
| `v.mask_rays(limit)` | `v.rays.mask(limit)` | |
| `v.scenes_baseline(min_scenes)` | `v.anomalies.baseline(min_scenes)` | |
| `v.scenes_anomaly(min_scenes)` | `v.anomalies.zscores(min_scenes)` | `v.anomalies()` |
| `v.scenes_anomaly_score(how)` | `v.episodes.score(how)` | |
| `v.episodes(threshold, ...)` | `v.episodes.find(threshold, ...)` | `v.episodes()` |

Los argumentos y lo que devuelven no cambian. `scenes_median` y `scenes_max`
eran propiedades: ahora son métodos, así que hay que agregarles `()`.

## Llamar a una vista

Llamar a una vista ejecuta su método principal. Si el primer argumento es un
string, elige otro método; cualquier otra cosa va al principal:

```python
v.episodes(3.0, 90)           # v.episodes.find(3.0, 90)
v.scenes("reduce", "mean")    # v.scenes.reduce("mean")
```

## Antes y después

```python
v = datasets.load("kadovar").rays.mask()     # era .mask_rays()
desc = v.descriptors()                       # era v.scenes_descriptors()
z = v.anomalies()                            # era v.scenes_anomaly()
events = v.episodes().query("n_scenes >= 2") # sin cambios
```

## Imports y gráficos

Las clases de los accessors se movieron a `volcanicus.accessors` y ya no se
importan desde la raíz del paquete (`from volcanicus.accessors import
AccessorABC, ScenesAccessor`); `volcanicus.utils.AccessorABC` ahora es
`volcanicus.accessors.AccessorABC`. Los gráficos no cambian:
`v.plot.scene()`, `v.plot.baseline()`, `v.plot.episodes()`,
`v.plot.rays_rose()`, ...
