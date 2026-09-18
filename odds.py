"""
Capa de cuotas de mercado
=========================
The Odds API: para cada juego se toma el MEJOR precio h2h entre casas
(best price) y la línea de totales por consenso (mediana) con el mejor
precio Over/Under disponible en esa línea.
"""

from __future__ import annotations

from typing import Any

import requests

from .config import config


def american_to_decimal(american_odds: float) -> float:
    if american_odds > 0:
        return (american_odds / 100.0) + 1.0
    if american_odds < 0:
        return (100.0 / abs(american_odds)) + 1.0
    return 1.0


def _mejor_precio(outcomes: list, nombre: str) -> float:
    """Mejor (máximo) precio americano para un outcome entre todas las casas."""
    mejores = [o.get("price", 0.0) for o in outcomes if o.get("name") == nombre]
    return float(max(mejores)) if mejores else 0.0


def _consenso_totales(bookmakers: list) -> tuple[float, float, float]:
    """
    Línea de totales por consenso (mediana de puntos) y mejor Over/Under
    en esa línea. Devuelve (linea, over_odds, under_odds) en formato decimal.
    """
    puntos: list[float] = []
    precios: dict[float, dict[str, float]] = {}
    for bm in bookmakers:
        for m in bm.get("markets", []):
            if m.get("key") != "totals":
                continue
            for o in m.get("outcomes", []):
                if "point" not in o:
                    continue
                pt = float(o["point"])
                puntos.append(pt)
                slot = precios.setdefault(pt, {"Over": 0.0, "Under": 0.0})
                lado = o.get("name", "")
                if lado in slot:
                    slot[lado] = max(slot[lado], float(o.get("price", 0.0)))
    if not puntos:
        return 0.0, 0.0, 0.0
    puntos.sort()
    linea = puntos[len(puntos) // 2]
    over_am = precios.get(linea, {}).get("Over", 0.0)
    under_am = precios.get(linea, {}).get("Under", 0.0)
    return (linea, american_to_decimal(over_am), american_to_decimal(under_am))


def obtener_cuotas(fecha: str) -> dict[str, dict[str, float]]:
    """
    Devuelve {matchup: {hom_ml_odds, away_ml_odds, total_line,
    over_cuota, under_cuota}} con cuotas en formato decimal.
    """
    if not config.api.odds_api_key:
        print("  ⚠️ ODDS_API_KEY no configurada. Se omitirá el cálculo de EV.")
        return {}

    params = {
        "apiKey": config.api.odds_api_key,
        "regions": "us",
        "markets": "h2h,totals",
        "oddsFormat": "american",
        "date": fecha,
    }
    try:
        r = requests.get(config.api.odds_api_url, params=params, timeout=15)
        if r.status_code != 200:
            print(f"  ⚠️ The Odds API respondió HTTP {r.status_code}")
            return {}
        data = r.json()
        if not isinstance(data, list):
            return {}

        cuotas: dict[str, dict[str, float]] = {}
        for game in data:
            home = game.get("home_team")
            away = game.get("away_team")
            bookmakers = game.get("bookmakers", [])
            if not home or not away or not bookmakers:
                continue

            h2h_outcomes = [
                o for bm in bookmakers for m in bm.get("markets", [])
                if m.get("key") == "h2h" for o in m.get("outcomes", [])
            ]
            h_am = _mejor_precio(h2h_outcomes, home)
            a_am = _mejor_precio(h2h_outcomes, away)
            linea, over_dec, under_dec = _consenso_totales(bookmakers)

            cuotas[f"{away} @ {home}"] = {
                "hom_ml_odds": american_to_decimal(h_am),
                "away_ml_odds": american_to_decimal(a_am),
                "total_line": linea,
                "over_cuota": over_dec,
                "under_cuota": under_dec,
            }
        print(f"  ✅ Cuotas obtenidas para {len(cuotas)} partidos (The Odds API)")
        return cuotas
    except Exception as e:
        print(f"  ⚠️ Error consultando The Odds API: {e}")
        return {}
