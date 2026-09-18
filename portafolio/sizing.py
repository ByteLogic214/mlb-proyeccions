"""
Sizing y control de riesgo
==========================
Kelly fraccional con topes por apuesta, exposición diaria y número de
picks; stop de drawdown de temporada. Todo en unidades del bankroll.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import config


@dataclass
class Propuesta:
    fecha: str
    matchup: str
    market: str            # 'h2h' | 'totals'
    selection: str         # nombre equipo o 'Over'/'Under'
    odds: float
    prob: float
    ev: float
    stake_u: float
    linea: float | None = None   # línea de totals (solo market='totals')
    motivo: str = ""


def kelly_stake(prob: float, odds: float, bankroll: float,
                kelly_fraction: float | None = None,
                max_stake_pct: float | None = None) -> float:
    """Stake en unidades: Kelly fraccional acotado por max_stake_pct."""
    p = config.portfolio
    kf = p.kelly_fraction if kelly_fraction is None else kelly_fraction
    cap = p.max_stake_pct if max_stake_pct is None else max_stake_pct
    b = odds - 1.0
    if b <= 0 or not (0 < prob < 1):
        return 0.0
    f = ((b * prob) - (1 - prob)) / b
    return max(0.0, min(f * kf, cap) * bankroll)


def generar_propuestas(df: pd.DataFrame, bankroll: float,
                       drawdown_actual: float = 0.0) -> list[Propuesta]:
    """
    Convierte el DataFrame de predicciones en propuestas de apuesta que
    superan los filtros de valor y riesgo, ordenadas por EV descendente.

    Filtros: EV mínimo, rango de cuotas, tope por stake, exposición diaria,
    máximo de picks por día y stop de drawdown de temporada.
    """
    p = config.portfolio
    if df is None or df.empty:
        return []
    if drawdown_actual >= p.max_drawdown_pct:
        print(f"  🛑 Stop de drawdown ({drawdown_actual:.1%} >= {p.max_drawdown_pct:.1%}). "
              "No se generan nuevas apuestas.")
        return []

    candidatos: list[Propuesta] = []
    for _, r in df.iterrows():
        fecha = str(r.get("fecha", ""))
        matchup = str(r.get("matchup", ""))

        # --- Moneyline ---
        for lado, prob_col, cuota_col, ev_col in [
            ("local", "ml_local_%", "cuota_local", "ev_local"),
            ("visitante", "ml_visitante_%", "cuota_visitante", "ev_visitante"),
        ]:
            ev = r.get(ev_col)
            cuota = r.get(cuota_col)
            if pd.isna(ev) or isinstance(cuota, str) or pd.isna(cuota):
                continue
            if ev < p.min_ev or not (p.min_odds <= cuota <= p.max_odds):
                continue
            prob = float(r.get(prob_col, 50.0)) / 100.0
            seleccion = str(r.get("hom_team" if lado == "local" else "away_team"))
            candidatos.append(Propuesta(
                fecha=fecha, matchup=matchup, market="h2h", selection=seleccion,
                odds=float(cuota), prob=prob, ev=float(ev),
                stake_u=kelly_stake(prob, float(cuota), bankroll),
                motivo=f"EV {ev:.1%}"))

        # --- Totals ---
        linea = r.get("linea_total_mercado", 0)
        if linea and not pd.isna(linea) and "total_runs" in r:
            p_over = float(r.get("prob_over_%", 50.0)) / 100.0
            for sel, prob, cuota, ev_col in [
                ("Over", p_over, r.get("over_cuota"), "ev_over"),
                ("Under", 1 - p_over, r.get("under_cuota"), "ev_under"),
            ]:
                if isinstance(cuota, str) or cuota is None or pd.isna(cuota):
                    continue
                ev = r.get(ev_col)
                if ev is None or pd.isna(ev) or ev < p.min_ev:
                    continue
                if not (p.min_odds <= float(cuota) <= p.max_odds):
                    continue
                candidatos.append(Propuesta(
                    fecha=fecha, matchup=matchup, market="totals", selection=sel,
                    odds=float(cuota), prob=float(prob), ev=float(ev),
                    stake_u=kelly_stake(prob, float(cuota), bankroll),
                    linea=float(linea),
                    motivo=f"Total {linea} {sel}, EV {ev:.1%}"))

    # Ordenar por EV y aplicar topes de cartera
    candidatos.sort(key=lambda c: c.ev, reverse=True)
    aceptadas: list[Propuesta] = []
    exposicion = 0.0
    for c in candidatos:
        if len(aceptadas) >= p.max_bets_per_day:
            break
        if c.stake_u <= 0:
            continue
        if exposicion + c.stake_u > p.max_daily_exposure_pct * bankroll:
            c.motivo += " | recortada por tope de exposición diaria"
            disponible = p.max_daily_exposure_pct * bankroll - exposicion
            if disponible <= 0:
                continue
            c.stake_u = round(min(c.stake_u, disponible), 2)
        exposicion += c.stake_u
        aceptadas.append(c)
    return aceptadas


def propuestas_a_dataframe(propuestas: list[Propuesta]) -> pd.DataFrame:
    if not propuestas:
        return pd.DataFrame(columns=[
            "bet_id", "fecha", "matchup", "market", "selection", "linea", "odds",
            "prob", "ev", "stake_u", "status", "result", "pnl_u",
            "closing_odds", "clv"])
    filas = []
    for c in propuestas:
        bet_id = f"{c.fecha}|{c.matchup}|{c.market}|{c.selection}"
        filas.append({
            "bet_id": bet_id, "fecha": c.fecha, "matchup": c.matchup,
            "market": c.market, "selection": c.selection,
            "linea": c.linea if c.linea is not None else "",
            "odds": round(c.odds, 3),
            "prob": round(c.prob, 4), "ev": round(c.ev, 4),
            "stake_u": round(c.stake_u, 2),
            "status": "open", "result": "", "pnl_u": 0.0,
            "closing_odds": "", "clv": "",
        })
    return pd.DataFrame(filas)
