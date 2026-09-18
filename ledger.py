"""
Libro de apuestas (ledger)
==========================
Registro idempotente de apuestas propuestas, liquidación contra resultados
reales, CLV (closing line value) cuando hay cuota de cierre, curva de
equity y drawdown. Persiste en output/bets.csv (commiteado por Actions).
"""

from __future__ import annotations

import os

import pandas as pd

from ..config import config

COLS = ["bet_id", "fecha", "matchup", "market", "selection", "linea", "odds",
        "prob", "ev", "stake_u", "status", "result", "pnl_u",
        "closing_odds", "clv"]


class Ledger:
    def __init__(self, path: str | None = None):
        self.path = path or os.path.join(config.data.output_dir, "bets.csv")
        self.df = self._cargar()

    def _cargar(self) -> pd.DataFrame:
        if os.path.exists(self.path):
            try:
                df = pd.read_csv(self.path, dtype={"closing_odds": str, "clv": str})
                for c in COLS:
                    if c not in df.columns:
                        df[c] = ""
                df = df[COLS]
                for c in ["bet_id", "fecha", "matchup", "market", "selection",
                          "linea", "status", "result", "closing_odds", "clv"]:
                    df[c] = df[c].astype(str)
                df["pnl_u"] = pd.to_numeric(df["pnl_u"], errors="coerce").fillna(0.0)
                return df
            except Exception:
                pass
        return pd.DataFrame(columns=COLS)

    def _guardar(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.df.to_csv(self.path, index=False)

    def registrar(self, nuevas: pd.DataFrame) -> int:
        """Añade propuestas de forma idempotente por bet_id. Devuelve nº nuevas."""
        if nuevas is None or nuevas.empty:
            return 0
        existentes = set(self.df["bet_id"]) if not self.df.empty else set()
        frescas = nuevas[~nuevas["bet_id"].isin(existentes)]
        if frescas.empty:
            return 0
        self.df = frescas.copy() if self.df.empty else pd.concat([self.df, frescas], ignore_index=True)
        self._guardar()
        return len(frescas)

    def liquidar(self, resultados: dict[str, dict]) -> int:
        """
        Liquida apuestas abiertas contra resultados reales.
        resultados: {matchup: {winner: 'home'|'away', total_runs: float}}
        Devuelve cuántas se liquidaron.
        """
        if self.df.empty:
            return 0
        mask = self.df["status"] == "open"
        liquidadas = 0
        for idx in self.df[mask].index:
            row = self.df.loc[idx]
            res = resultados.get(row["matchup"])
            if not res:
                continue
            stake = float(row["stake_u"])
            odds = float(row["odds"])
            if row["market"] == "h2h":
                gano = (res["winner"] == "home" and row["selection"] == res.get("hom_team")) or \
                       (res["winner"] == "away" and row["selection"] == res.get("away_team"))
            else:  # totals — requiere la línea registrada en la propuesta
                try:
                    linea = float(row["linea"])
                except (TypeError, ValueError):
                    continue  # sin línea no se liquida (queda open)
                total = float(res["total_runs"])
                if total == linea:
                    self.df.loc[idx, ["status", "result", "pnl_u"]] = ["push", "push", 0.0]
                    liquidadas += 1
                    continue
                gano = (row["selection"] == "Over" and total > linea) or \
                       (row["selection"] == "Under" and total < linea)
            if gano:
                self.df.loc[idx, ["status", "result", "pnl_u"]] = ["settled", "win", round(stake * (odds - 1), 2)]
            else:
                self.df.loc[idx, ["status", "result", "pnl_u"]] = ["settled", "loss", -stake]
            liquidadas += 1
        if liquidadas:
            self._guardar()
        return liquidadas

    def equity(self) -> pd.DataFrame:
        """Curva de equity cronológica desde las apuestas liquidadas."""
        settled = self.df[self.df["status"] == "settled"].copy()
        if settled.empty:
            return pd.DataFrame(columns=["fecha", "pnl_dia", "bankroll", "drawdown_pct"])
        settled["pnl_u"] = settled["pnl_u"].astype(float)
        por_dia = settled.groupby("fecha", as_index=False)["pnl_u"].sum().rename(
            columns={"pnl_u": "pnl_dia"}).sort_values("fecha")
        por_dia["bankroll"] = config.portfolio.bankroll_units + por_dia["pnl_dia"].cumsum()
        pico = por_dia["bankroll"].cummax()
        por_dia["drawdown_pct"] = ((pico - por_dia["bankroll"]) / pico).round(4)
        return por_dia

    def drawdown_actual(self) -> float:
        eq = self.equity()
        return float(eq["drawdown_pct"].max()) if not eq.empty else 0.0

    def metricas(self) -> dict:
        settled = self.df[self.df["status"] == "settled"].copy()
        if settled.empty:
            return {"apuestas": 0, "staked_u": 0.0, "pnl_u": 0.0,
                    "roi_pct": 0.0, "hit_rate_pct": 0.0}
        settled["pnl_u"] = settled["pnl_u"].astype(float)
        staked = float(settled["stake_u"].astype(float).sum())
        pnl = float(settled["pnl_u"].sum())
        wins = int((settled["result"] == "win").sum())
        n = len(settled)
        clv_vals = pd.to_numeric(settled["clv"], errors="coerce").dropna()
        return {
            "apuestas": n,
            "staked_u": round(staked, 2),
            "pnl_u": round(pnl, 2),
            "roi_pct": round(pnl / staked * 100, 2) if staked else 0.0,
            "hit_rate_pct": round(wins / n * 100, 2),
            "wins": wins,
            "losses": n - wins,
            "avg_odds": round(float(settled["odds"].astype(float).mean()), 3),
            "avg_clv_pct": round(float(clv_vals.mean() * 100), 2) if len(clv_vals) else None,
            "drawdown_max_pct": round(self.drawdown_actual() * 100, 2),
            "bankroll_actual_u": round(config.portfolio.bankroll_units + pnl, 2),
        }
