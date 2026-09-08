# ⚾ MLB Betting ML System

Sistema de Machine Learning para proyección de mercados de apuestas de la MLB
usando la **MLB Stats API gratuita**.

## 🎯 Qué predice cada día

| Mercado | Proyección |
|---|---|
| Moneyline | Probabilidad de victoria del local |
| Total del juego | Runs totales |
| Runs por equipo | Runs estimados de cada equipo |
| Hits | Hits por equipo |
| Bases totales | Bases por equipo |
| Ponches | K de bateadores y lanzadores |

## ⚙️ Cómo funciona

1. Descarga calendario y game logs de cada equipo.
2. Calcula características: medias móviles 3/5/10, promedio de temporada, tendencias, descanso.
3. Entrena un Random Forest por mercado.
4. Genera predicciones en `output/`.

## 🚀 Uso

El workflow corre automáticamente a las 12:00 UTC. También puedes ejecutarlo manualmente:
**Actions → MLB Daily Predictions → Run workflow**.

## ⚠️ Descargo de responsabilidad

Herramienta educativa. No es consejo de apuestas. Apuesta responsablemente.
