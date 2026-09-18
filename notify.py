"""
Notificaciones Telegram
=======================
Envío de mensajes vía Bot API con chunking automático (límite 4096).
Opcional por secrets TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID; falla de
forma silenciosa y nunca interrumpe el pipeline.
"""

from __future__ import annotations

import time

import pandas as pd
import requests

from ..config import config


def _fmt(x, dec=2) -> str:
    try:
        if isinstance(x, str):
            return x
        return f"{float(x):.{dec}f}"
    except (TypeError, ValueError):
        return "—"


def enviar(mensaje: str) -> bool:
    tg = config.telegram
    if not tg.bot_token or not tg.chat_id:
        print("  ⚠️ Telegram no configurado (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID). Se omite.")
        return False
    try:
        url = f"https://api.telegram.org/bot{tg.bot_token}/sendMessage"
        r = requests.post(url, json={
            "chat_id": tg.chat_id, "text": mensaje,
            "disable_web_page_preview": True}, timeout=15)
        if r.status_code != 200:
            print(f"  ⚠️ Telegram respondió HTTP {r.status_code}: {r.text[:200]}")
            return False
        return True
    except Exception as e:
        print(f"  ⚠️ Error enviando a Telegram: {e}")
        return False


def enviar_lote(mensajes: list[str]) -> bool:
    ok = True
    for i, m in enumerate(mensajes):
        ok = enviar(m) and ok
        if i < len(mensajes) - 1:
            time.sleep(0.5)
    if ok and mensajes:
        print(f"  ✅ Enviado a Telegram: {len(mensajes)} mensaje(s).")
    return ok


def _chunk(lineas: list[str]) -> list[str]:
    texto = "\n".join(lineas)
    if len(texto) <= config.telegram.max_len:
        return [texto]
    mensajes, actual = [], [lineas[0]]
    for linea in lineas[1:]:
        if len("\n".join(actual + [linea])) > config.telegram.max_len:
            mensajes.append("\n".join(actual))
            actual = [lineas[0], "(continuación)", linea]
        else:
            actual.append(linea)
    mensajes.append("\n".join(actual))
    return mensajes


def mensaje_predicciones(fecha: str, df: pd.DataFrame,
                         propuestas: pd.DataFrame | None = None) -> list[str]:
    lineas = [f"⚾ PROYECCIONES MLB — {fecha}",
              f"Juegos analizados: {len(df)}", "=" * 34]
    for _, r in df.iterrows():
        lineas += ["", f"🏟 {r.get('matchup', '—')}",
                   f"   Abridores: {r.get('away_pitcher', '—')} @ {r.get('hom_pitcher', '—')}",
                   f"   ML: {r.get('hom_team')} {r.get('ml_local_%', '—')}% | "
                   f"{r.get('away_team')} {r.get('ml_visitante_%', '—')}%",
                   f"   Carreras: {_fmt(r.get('proj_carreras_local'))} - "
                   f"{_fmt(r.get('proj_carreras_visitante'))} "
                   f"(Total {_fmt(r.get('proj_total_carreras'))})"]
        linea = r.get("linea_total_mercado", 0)
        if linea and not pd.isna(linea):
            lineas.append(f"   Línea total: {_fmt(linea, 1)} "
                          f"(Over {r.get('prob_over_%', '—')}%)")
    if propuestas is not None and not propuestas.empty:
        lineas += ["", "💰 APUESTAS DEL DÍA:"]
        for _, b in propuestas.iterrows():
            lineas.append(f"   • {b['selection']} ({b['market']}) @ {b['odds']} "
                          f"— EV {float(b['ev']):.1%} — Stake {b['stake_u']}u")
    else:
        lineas += ["", "💰 Sin apuestas que superen los filtros de valor/riesgo."]
    lineas += ["", "=" * 34,
               "⚠️ Herramienta educativa. No es consejo de apuestas."]
    return _chunk(lineas)


def mensaje_reporte(metricas: dict, equity_dia: pd.DataFrame | None = None) -> str:
    m = metricas
    lineas = ["📊 REPORTE DE INVERSIÓN MLB", "=" * 30,
              f"Bankroll actual:   {m.get('bankroll_actual_u', '—')}u",
              f"Apuestas:          {m.get('apuestas', 0)} "
              f"({m.get('wins', 0)}W / {m.get('losses', 0)}L)",
              f"Staked acumulado:  {m.get('staked_u', 0)}u",
              f"PnL:               {m.get('pnl_u', 0):+}u",
              f"ROI / Yield:       {m.get('roi_pct', 0):+.2f}%",
              f"Hit rate:          {m.get('hit_rate_pct', 0)}%",
              f"Cuota media:       {m.get('avg_odds', '—')}",
              f"Drawdown máx:      {m.get('drawdown_max_pct', 0)}%"]
    if m.get("avg_clv_pct") is not None:
        lineas.append(f"CLV medio:         {m['avg_clv_pct']:+.2f}%")
    if equity_dia is not None and not equity_dia.empty:
        ult = equity_dia.iloc[-1]
        lineas += ["-" * 30,
                   f"Último día: PnL {ult['pnl_dia']:+.2f}u | "
                   f"Bankroll {ult['bankroll']:.2f}u"]
    lineas.append("=" * 30)
    return "\n".join(lineas)
