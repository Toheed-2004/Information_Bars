import os
import imageio
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from pathlib import Path
import plotly.graph_objects as go


def add_FVG(fig, df, fvg_data):
    """Add Fair Value Gaps to the chart"""
    for i in range(len(fvg_data)):
        if not np.isnan(fvg_data["fvg"].iloc[i]) and i < len(df):
            x1 = int(
                fvg_data["fvg_mitigated_index"].iloc[i]
                if fvg_data["fvg_mitigated_index"].iloc[i] != 0
                else len(df) - 1
            )
            if x1 >= len(df):
                x1 = len(df) - 1

            fig.add_shape(
                type="rect",
                x0=df.index[i],
                y0=fvg_data["fvg_top"].iloc[i],
                x1=df.index[x1],
                y1=fvg_data["fvg_bottom"].iloc[i],
                line=dict(width=0),
                fillcolor="yellow",
                opacity=0.2,
            )
            mid_x = round((i + x1) / 2)
            mid_y = (fvg_data["fvg_top"].iloc[i] + fvg_data["fvg_bottom"].iloc[i]) / 2
            
            # Add label for FVG
            fig.add_annotation(
                x=df.index[mid_x],
                y=mid_y,
                text=f"FVG {'Bull' if fvg_data['fvg'].iloc[i] == 1 else 'Bear'}",
                font=dict(color="rgba(255, 255, 255, 0.6)", size=9),
                showarrow=False,
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="yellow",
                borderwidth=1,
            )
    return fig


def add_swing_highs_lows(fig, df, swing_data):
    """Add Swing Highs and Lows to the chart - with horizontal lines"""
    for i in range(len(swing_data)):
        if not np.isnan(swing_data["swing_high_low"].iloc[i]) and i < len(df):
            level = swing_data["swing_high_low_level"].iloc[i]
            swing_type = swing_data["swing_high_low"].iloc[i]
            
            # Find the next swing point for horizontal line length
            next_swing_idx = None
            for j in range(i + 1, len(swing_data)):
                if not np.isnan(swing_data["swing_high_low"].iloc[j]):
                    next_swing_idx = j
                    break
            
            if next_swing_idx is None:
                next_swing_idx = len(df) - 1
            
            # Add horizontal line from current swing to next swing
            fig.add_trace(
                go.Scatter(
                    x=[df.index[i], df.index[next_swing_idx]],
                    y=[level, level],
                    mode="lines",
                    line=dict(
                        color="rgba(0, 255, 0, 0.3)" if swing_type == -1 else "rgba(255, 0, 0, 0.3)",
                        width=1,
                        dash="dash"
                    ),
                    showlegend=False,
                )
            )
            
            # Add marker at swing point
            fig.add_trace(
                go.Scatter(
                    x=[df.index[i]],
                    y=[level],
                    mode="markers+text",
                    marker=dict(
                        size=10,
                        color="green" if swing_type == -1 else "red",
                        symbol="triangle-down" if swing_type == -1 else "triangle-up"
                    ),
                    text=f"{'Swing Low' if swing_type == -1 else 'Swing High'}: {level:.5f}",
                    textposition="top center",
                    textfont=dict(color="rgba(255, 255, 255, 0.6)", size=8),
                    showlegend=False,
                )
            )
    return fig


def add_bos_choch(fig, df, bos_choch_data):
    """Add Break of Structure and Change of Character to the chart"""
    for i in range(len(bos_choch_data)):
        if not np.isnan(bos_choch_data["bos"].iloc[i]) and i < len(df):
            broken_idx = int(bos_choch_data["bos_broken"].iloc[i])
            if broken_idx < len(df):
                level = bos_choch_data["bos_level"].iloc[i]
                
                # Add line for BOS
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[i], df.index[broken_idx]],
                        y=[level, level],
                        mode="lines",
                        line=dict(color="rgba(255, 165, 0, 0.5)", width=2),
                        showlegend=False,
                    )
                )
                
                # Add label for BOS
                mid_x = (i + broken_idx) // 2
                fig.add_annotation(
                    x=df.index[mid_x],
                    y=level,
                    text=f"BOS {'Bull' if bos_choch_data['bos'].iloc[i] == 1 else 'Bear'}: {level:.5f}",
                    font=dict(color="rgba(255, 165, 0, 0.8)", size=9),
                    showarrow=False,
                    bgcolor="rgba(0,0,0,0.5)",
                    bordercolor="orange",
                    borderwidth=1,
                )

        if not np.isnan(bos_choch_data["bos_choch"].iloc[i]) and i < len(df):
            broken_idx = int(bos_choch_data["bos_broken"].iloc[i])
            if broken_idx < len(df):
                level = bos_choch_data["bos_level"].iloc[i]
                
                # Add line for CHoCH
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[i], df.index[broken_idx]],
                        y=[level, level],
                        mode="lines",
                        line=dict(color="rgba(0, 0, 255, 0.5)", width=2, dash="dot"),
                        showlegend=False,
                    )
                )
                
                # Add label for CHoCH
                mid_x = (i + broken_idx) // 2
                fig.add_annotation(
                    x=df.index[mid_x],
                    y=level,
                    text=f"CHoCH {'Bull' if bos_choch_data['bos_choch'].iloc[i] == 1 else 'Bear'}: {level:.5f}",
                    font=dict(color="rgba(0, 0, 255, 0.8)", size=9),
                    showarrow=False,
                    bgcolor="rgba(0,0,0,0.5)",
                    bordercolor="blue",
                    borderwidth=1,
                )
    return fig


def add_OB(fig, df, ob_data):
    """Add Order Blocks to the chart"""

    def format_volume(volume):
        if volume >= 1e12:
            return f"{volume / 1e12:.3f}T"
        elif volume >= 1e9:
            return f"{volume / 1e9:.3f}B"
        elif volume >= 1e6:
            return f"{volume / 1e6:.3f}M"
        elif volume >= 1e3:
            return f"{volume / 1e3:.3f}k"
        else:
            return f"{volume:.2f}"

    for i in range(len(ob_data)):
        if ob_data["ob"].iloc[i] in [1, -1]:
            x1 = int(
                ob_data["ob_mitigated_index"].iloc[i]
                if ob_data["ob_mitigated_index"].iloc[i] != 0
                else len(df) - 1
            )

            fig.add_shape(
                type="rect",
                x0=df.index[i],
                y0=ob_data["ob_bottom"].iloc[i],
                x1=df.index[x1],
                y1=ob_data["ob_top"].iloc[i],
                line=dict(color="Purple", width=1),
                fillcolor="Purple",
                opacity=0.2,
            )

            x_center = df.index[int(i + (x1 - i) / 2)]
            y_center = (ob_data["ob_bottom"].iloc[i] + ob_data["ob_top"].iloc[i]) / 2
            volume_text = format_volume(ob_data["ob_volume"].iloc[i])
            
            # Add detailed label for OB
            ob_type = "Bull" if ob_data["ob"].iloc[i] == 1 else "Bear"
            fig.add_annotation(
                x=x_center,
                y=y_center,
                text=f"OB {ob_type}\nVol: {volume_text}\nStr: {ob_data['ob_percentage'].iloc[i]:.1f}%",
                font=dict(color="white", size=8),
                showarrow=False,
                bgcolor="rgba(128, 0, 128, 0.7)",
                bordercolor="purple",
                borderwidth=1,
                align="center",
            )
    return fig


def add_previous_high_low(fig, df, previous_hl_data):
    """Add Previous High/Low levels to the chart - with labels"""
    # Get unique levels
    prev_highs = previous_hl_data["phl_previous_high"].dropna().unique()
    prev_lows = previous_hl_data["phl_previous_low"].dropna().unique()
    
    # Add horizontal lines for previous highs
    for high_level in prev_highs:
        if not np.isnan(high_level):
            # Find first and last occurrence of this level
            high_indices = previous_hl_data[previous_hl_data["phl_previous_high"] == high_level].index
            if len(high_indices) > 0:
                start_idx = high_indices[0]
                end_idx = high_indices[-1]
                
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[start_idx], df.index[end_idx]],
                        y=[high_level, high_level],
                        mode="lines",
                        line=dict(color="rgba(255, 255, 255, 0.4)", width=1, dash="dash"),
                        showlegend=False,
                    )
                )
                
                # Add label
                mid_idx = start_idx + (end_idx - start_idx) // 2
                fig.add_annotation(
                    x=df.index[mid_idx],
                    y=high_level,
                    text=f"Prev High: {high_level:.5f}",
                    font=dict(color="rgba(255, 255, 255, 0.6)", size=8),
                    showarrow=False,
                    bgcolor="rgba(0,0,0,0.5)",
                    yshift=10,
                )
    
    # Add horizontal lines for previous lows
    for low_level in prev_lows:
        if not np.isnan(low_level):
            # Find first and last occurrence of this level
            low_indices = previous_hl_data[previous_hl_data["phl_previous_low"] == low_level].index
            if len(low_indices) > 0:
                start_idx = low_indices[0]
                end_idx = low_indices[-1]
                
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[start_idx], df.index[end_idx]],
                        y=[low_level, low_level],
                        mode="lines",
                        line=dict(color="rgba(255, 255, 255, 0.4)", width=1, dash="dash"),
                        showlegend=False,
                    )
                )
                
                # Add label
                mid_idx = start_idx + (end_idx - start_idx) // 2
                fig.add_annotation(
                    x=df.index[mid_idx],
                    y=low_level,
                    text=f"Prev Low: {low_level:.5f}",
                    font=dict(color="rgba(255, 255, 255, 0.6)", size=8),
                    showarrow=False,
                    bgcolor="rgba(0,0,0,0.5)",
                    yshift=-10,
                )
    
    return fig


def add_sessions(fig, df, sessions_data):
    """Add Trading sessions to the chart"""
    # Find session start and end points
    session_starts = []
    session_ends = []
    
    for i in range(len(sessions_data)):
        if sessions_data["session_active"].iloc[i] == 1:
            if i == 0 or sessions_data["session_active"].iloc[i-1] == 0:
                session_starts.append(i)
            if i == len(sessions_data) - 1 or sessions_data["session_active"].iloc[i+1] == 0:
                session_ends.append(i)
    
    # Draw session rectangles
    for start_idx, end_idx in zip(session_starts, session_ends):
        if start_idx < len(df) and end_idx < len(df):
            session_high = sessions_data["session_high"].iloc[end_idx]
            session_low = sessions_data["session_low"].iloc[end_idx]
            
            fig.add_shape(
                type="rect",
                x0=df.index[start_idx],
                y0=session_low,
                x1=df.index[end_idx],
                y1=session_high,
                line=dict(width=0),
                fillcolor="#16866E",
                opacity=0.1,
            )
            
            # Add session label
            mid_idx = start_idx + (end_idx - start_idx) // 2
            fig.add_annotation(
                x=df.index[mid_idx],
                y=(session_high + session_low) / 2,
                text="Session",
                font=dict(color="rgba(22, 134, 110, 0.8)", size=9),
                showarrow=False,
                bgcolor="rgba(0,0,0,0.5)",
            )
    
    return fig


def add_retracements(fig, df, retracements_data):
    """Add Retracement levels to the chart with labels"""
    for i in range(1, len(retracements_data)):
        if retracements_data["retracement_direction"].iloc[i] != 0:
            # Add retracement level annotation
            if i % 20 == 0:  # Show every 20 candles to avoid clutter
                current_retracement = retracements_data["retracement_current_retracement%"].iloc[i]
                deepest_retracement = retracements_data["retracement_deepest_retracement%"].iloc[i]
                direction = "Bull" if retracements_data["retracement_direction"].iloc[i] == 1 else "Bear"
                
                fig.add_annotation(
                    x=df.index[i],
                    y=df["close"].iloc[i],
                    text=f"{direction} Retrace\nC: {current_retracement}%\nD: {deepest_retracement}%",
                    font=dict(color="rgba(255, 255, 255, 0.6)", size=8),
                    showarrow=False,
                    bgcolor="rgba(0,0,0,0.5)",
                    bordercolor="cyan",
                    borderwidth=1,
                    align="center",
                )
    return fig


def add_algorithmic_order_block(fig, df, aob_data):
    """Add Algorithmic Order Blocks to chart"""
    for i in range(len(aob_data)):
        if not np.isnan(aob_data["aob_type"].iloc[i]):
            color = (
                "rgba(0, 255, 0, 0.3)"
                if aob_data["aob_type"].iloc[i] == 1
                else "rgba(255, 0, 0, 0.3)"
            )
            border_color = "green" if aob_data["aob_type"].iloc[i] == 1 else "red"
            label = "AOB-Bull" if aob_data["aob_type"].iloc[i] == 1 else "AOB-Bear"

            fig.add_shape(
                type="rect",
                x0=df.index[i],
                y0=aob_data["aob_bottom"].iloc[i],
                x1=df.index[i + 1] if i < len(df) - 1 else df.index[i],
                y1=aob_data["aob_top"].iloc[i],
                line=dict(color=border_color, width=2),
                fillcolor=color,
                opacity=0.2,
            )

            fig.add_annotation(
                x=df.index[i],
                y=(aob_data["aob_top"].iloc[i] + aob_data["aob_bottom"].iloc[i]) / 2,
                text=f"{label}\nStr: {aob_data['aob_strength'].iloc[i]:.0f}%",
                font=dict(size=8, color=border_color),
                showarrow=False,
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor=border_color,
                borderwidth=1,
                yshift=10 if aob_data["aob_type"].iloc[i] == 1 else -10,
            )
    return fig


def add_breaker_block(fig, df, bb_data):
    """Add Breaker Blocks to chart"""
    for i in range(len(bb_data)):
        if not np.isnan(bb_data["bb_type"].iloc[i]):
            color = (
                "rgba(0, 200, 255, 0.7)"
                if bb_data["bb_type"].iloc[i] == 1
                else "rgba(255, 100, 0, 0.7)"
            )
            border_color = "cyan" if bb_data["bb_type"].iloc[i] == 1 else "orange"
            label = "BB-Bull" if bb_data["bb_type"].iloc[i] == 1 else "BB-Bear"

            # Add marker
            fig.add_trace(
                go.Scatter(
                    x=[df.index[i]],
                    y=[bb_data["bb_level"].iloc[i]],
                    mode="markers",
                    marker=dict(
                        size=12,
                        color=color,
                        symbol="diamond" if bb_data["bb_type"].iloc[i] == 1 else "square",
                        line=dict(width=2, color=border_color),
                    ),
                    showlegend=False,
                )
            )
            
            # Add label
            fig.add_annotation(
                x=df.index[i],
                y=bb_data["bb_level"].iloc[i],
                text=f"{label}\n{bb_data['bb_level'].iloc[i]:.5f}\nStr: {bb_data['bb_strength'].iloc[i]:.0f}%",
                font=dict(size=8, color=border_color),
                showarrow=False,
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor=border_color,
                borderwidth=1,
                yshift=15 if bb_data["bb_type"].iloc[i] == 1 else -15,
            )

            # Add horizontal line
            fig.add_shape(
                type="line",
                x0=df.index[i],
                y0=bb_data["bb_level"].iloc[i],
                x1=df.index[min(i + 20, len(df) - 1)],
                y1=bb_data["bb_level"].iloc[i],
                line=dict(color=color, width=1, dash="dash"),
            )
    return fig


def add_mitigation_block(fig, df, mb_data):
    """Add Mitigation Blocks to chart"""
    for i in range(len(mb_data)):
        if not np.isnan(mb_data["mb_type"].iloc[i]):
            color = (
                "rgba(255, 215, 0, 0.3)"
                if mb_data["mb_type"].iloc[i] == 1
                else "rgba(138, 43, 226, 0.3)"
            )
            border_color = "gold" if mb_data["mb_type"].iloc[i] == 1 else "purple"
            label = "MB-Bull" if mb_data["mb_type"].iloc[i] == 1 else "MB-Bear"

            fig.add_shape(
                type="rect",
                x0=df.index[i],
                y0=mb_data["mb_bottom"].iloc[i],
                x1=df.index[i + 1] if i < len(df) - 1 else df.index[i],
                y1=mb_data["mb_top"].iloc[i],
                line=dict(color=border_color, width=1),
                fillcolor=color,
                opacity=0.15,
            )

            fig.add_annotation(
                x=df.index[i],
                y=(mb_data["mb_top"].iloc[i] + mb_data["mb_bottom"].iloc[i]) / 2,
                text=f"{label}\nTop: {mb_data['mb_top'].iloc[i]:.5f}\nBottom: {mb_data['mb_bottom'].iloc[i]:.5f}",
                font=dict(size=7, color=border_color),
                showarrow=False,
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor=border_color,
                borderwidth=1,
                yshift=5 if mb_data["mb_type"].iloc[i] == 1 else -5,
            )
    return fig


def add_bpr(fig, df, bpr_data):
    """Add Balanced Price Ranges to chart"""
    for i in range(len(bpr_data)):
        if not np.isnan(bpr_data["bpr_top"].iloc[i]) and not np.isnan(bpr_data["bpr_bottom"].iloc[i]):
            # Draw BPR rectangle
            fig.add_shape(
                type="rect",
                x0=df.index[max(0, i - 5)],
                y0=bpr_data["bpr_bottom"].iloc[i],
                x1=df.index[min(i + 5, len(df) - 1)],
                y1=bpr_data["bpr_top"].iloc[i],
                line=dict(color="rgba(100, 100, 255, 0.5)", width=1),
                fillcolor="rgba(100, 100, 255, 0.1)",
            )
            
            # Add label every 20 candles
            if i % 20 == 0:
                fig.add_annotation(
                    x=df.index[i],
                    y=(bpr_data["bpr_top"].iloc[i] + bpr_data["bpr_bottom"].iloc[i]) / 2,
                    text=f"BPR Range\nTop: {bpr_data['bpr_top'].iloc[i]:.5f}\nBottom: {bpr_data['bpr_bottom'].iloc[i]:.5f}\nStr: {bpr_data['bpr_strength'].iloc[i]:.0f}%",
                    font=dict(size=7, color="rgba(100, 100, 255, 0.8)"),
                    showarrow=False,
                    bgcolor="rgba(0,0,0,0.5)",
                    bordercolor="blue",
                    borderwidth=1,
                )
    return fig


def add_liquidity_swing_hl(fig, df, liq_data):
    """Add Swing High/Low liquidity to chart"""
    swept_indices = np.where(~np.isnan(liq_data["lshl_swept"].values))[0]

    for idx in swept_indices:
        swing_idx = int(liq_data["lshl_swing_index"].iloc[idx])
        liq_level = liq_data["lshl_level"].iloc[idx]
        liq_type = liq_data["lshl_type"].iloc[idx]

        color = "rgba(255, 100, 0, 0.6)" if liq_type == 1 else "rgba(0, 200, 100, 0.6)"
        border_color = "orange" if liq_type == 1 else "lightgreen"
        label = "Swing High Liq" if liq_type == 1 else "Swing Low Liq"

        # Draw line from swing to sweep
        fig.add_trace(
            go.Scatter(
                x=[df.index[swing_idx], df.index[idx]],
                y=[liq_level, liq_level],
                mode="lines",
                line=dict(color=color, width=2),
                showlegend=False,
            )
        )

        # Add label at sweep point
        fig.add_annotation(
            x=df.index[idx],
            y=liq_level,
            text=f"{label}\nLevel: {liq_level:.5f}",
            font=dict(size=8, color=border_color),
            showarrow=False,
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor=border_color,
            borderwidth=1,
            xshift=25,
            yshift=10 if liq_type == 1 else -10,
        )

        # Add marker at sweep point
        fig.add_trace(
            go.Scatter(
                x=[df.index[idx]],
                y=[liq_level],
                mode="markers",
                marker=dict(size=10, color=color, symbol="star", line=dict(width=1, color=border_color)),
                showlegend=False,
            )
        )
        
        # Add marker at swing point
        fig.add_trace(
            go.Scatter(
                x=[df.index[swing_idx]],
                y=[liq_level],
                mode="markers",
                marker=dict(size=8, color=color, symbol="circle", line=dict(width=1, color=border_color)),
                showlegend=False,
            )
        )

    return fig


def detect_indicator_type(data):
    """Detect indicator type based on lowercase column names"""
    if isinstance(data, pd.DataFrame):
        cols = set(col.lower() for col in data.columns)
        
        # FVG indicators
        if {"fvg", "fvg_top", "fvg_bottom", "fvg_mitigated_index"}.issubset(cols):
            return "fvg"
        # Swing highs/lows
        elif {"swing_high_low", "swing_high_low_level"}.issubset(cols):
            return "swing_highs_lows"
        # BOS/CHoCH
        elif {"bos", "bos_choch", "bos_level", "bos_broken"}.issubset(cols):
            return "bos_choch"
        # Order Blocks
        elif {"ob", "ob_top", "ob_bottom", "ob_volume", "ob_mitigated_index", "ob_percentage"}.issubset(cols):
            return "ob"
        # Previous High/Low
        elif {"phl_previous_high", "phl_previous_low", "phl_broken_high", "phl_broken_low"}.issubset(cols):
            return "previous_high_low"
        # Sessions
        elif {"session_active", "session_high", "session_low"}.issubset(cols):
            return "sessions"
        # Retracements
        elif {"retracement_direction", "retracement_current_retracement%", "retracement_deepest_retracement%"}.issubset(cols):
            return "retracements"
        # Algorithmic Order Block
        elif {"aob_type", "aob_top", "aob_bottom", "aob_strength"}.issubset(cols):
            return "algorithmic_order_block"
        # Breaker Block
        elif {"bb_type", "bb_level", "bb_strength"}.issubset(cols):
            return "breaker_block"
        # Mitigation Block
        elif {"mb_type", "mb_top", "mb_bottom", "mb_fvg_index"}.issubset(cols):
            return "mitigation_block"
        # Balanced Price Range
        elif {"bpr_top", "bpr_bottom", "bpr_strength"}.issubset(cols):
            return "bpr"
        # Liquidity Swing HL
        elif {"lshl_type", "lshl_level", "lshl_swing_index", "lshl_swept"}.issubset(cols):
            return "liquidity_swing_hl"
        
        # Check for columns with smc_ prefix
        smc_cols = set(col.replace('smc_', '').lower() for col in data.columns)
        
        # Check all patterns again with smc_ prefix removed
        if {"fvg", "top", "bottom", "mitigated_index"}.issubset(smc_cols):
            return "fvg"
        elif {"swing_high_low", "level"}.issubset(smc_cols) and len(cols) == 2:
            return "swing_highs_lows"
        elif {"bos", "choch", "level", "broken"}.issubset(smc_cols):
            return "bos_choch"
        elif {"ob", "top", "bottom", "volume", "mitigated_index", "percentage"}.issubset(smc_cols):
            return "ob"
        elif {"previous_high", "previous_low", "broken_high", "broken_low"}.issubset(smc_cols):
            return "previous_high_low"
        elif {"active", "high", "low"}.issubset(smc_cols):
            return "sessions"
        elif {"direction", "current_retracement%", "deepest_retracement%"}.issubset(smc_cols):
            return "retracements"
        elif {"aob", "type", "top", "bottom", "strength"}.issubset(smc_cols):
            return "algorithmic_order_block"
        elif {"bb", "type", "level", "strength"}.issubset(smc_cols):
            return "breaker_block"
        elif {"mb", "type", "top", "bottom", "fvg_index"}.issubset(smc_cols):
            return "mitigation_block"
        elif {"bpr", "top", "bottom", "strength"}.issubset(smc_cols):
            return "bpr"
        elif {"lshl", "type", "level", "swing_index", "swept"}.issubset(smc_cols):
            return "liquidity_swing_hl"
    
    return None


def get_price_range(df):
    """Calculate a reasonable price range for the y-axis"""
    # Get the price range from candlesticks
    price_min = df["low"].min()
    price_max = df["high"].max()
    price_range = price_max - price_min
    
    # Add 10% padding on top and bottom
    padding = price_range * 0.10
    
    # For very small ranges (like 1 pip in Forex), add minimum padding
    if padding < price_min * 0.001:  # Less than 0.1%
        padding = price_min * 0.005  # Add 0.5% padding
    
    return price_min - padding, price_max + padding


def create_figure_with_smc(df, smc_data_list, width=1920, height=1080):
    """Create a complete SMC figure with selected indicators
    
    Parameters:
    df: DataFrame - OHLCV data
    smc_data_list: list or DataFrame - list of indicator DataFrames or single DataFrame
    """
    # Calculate price range for y-axis BEFORE creating the figure
    y_min, y_max = get_price_range(df)
    
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                increasing_line_color="#77dd76",
                decreasing_line_color="#ff6962",
                name="Price",
            )
        ]
    )
    
    # Set y-axis range BEFORE adding indicators to prevent auto-scaling
    fig.update_yaxes(range=[y_min, y_max])
    
    # Convert single DataFrame to list
    if isinstance(smc_data_list, pd.DataFrame):
        smc_data_list = [smc_data_list]
    elif isinstance(smc_data_list, dict):
        smc_data_list = list(smc_data_list.values())
    elif not isinstance(smc_data_list, list):
        smc_data_list = [smc_data_list]
    
    indicator_functions = {
        "fvg": add_FVG,
        "swing_highs_lows": add_swing_highs_lows,
        "bos_choch": add_bos_choch,
        "ob": add_OB,
        "previous_high_low": add_previous_high_low,
        "sessions": add_sessions,
        "retracements": add_retracements,
        "algorithmic_order_block": add_algorithmic_order_block,
        "breaker_block": add_breaker_block,
        "mitigation_block": add_mitigation_block,
        "bpr": add_bpr,
        "liquidity_swing_hl": add_liquidity_swing_hl,
    }
    
    # Process each indicator
    for indicator_data in smc_data_list:
        indicator_type = detect_indicator_type(indicator_data)
        
        if indicator_type and indicator_type in indicator_functions:
            try:
                print(f"Adding {indicator_type} indicator...")
                fig = indicator_functions[indicator_type](fig, df, indicator_data)
            except Exception as e:
                print(f"Warning: Failed to add {indicator_type}: {str(e)}")
    
    # Update layout with proper y-axis scaling
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        showlegend=False,
        margin=dict(l=0, r=0, b=0, t=0),
        plot_bgcolor="rgba(0,0,0,1)",
        paper_bgcolor="rgba(0,0,0,1)",
        font=dict(color="white"),
        width=width,
        height=height,
        # Force y-axis to use our calculated range
        yaxis=dict(
            range=[y_min, y_max],
            showgrid=True,
            gridcolor="rgba(50, 50, 50, 0.3)",
            showline=True,
            linecolor='rgba(255,255,255,0.3)',
            side='right',
            fixedrange=False,  # Allow user to zoom but start with good range
            tickformat=".5f",  # Show 5 decimal places for Forex
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(50, 50, 50, 0.3)",
            showline=True,
            linecolor='rgba(255,255,255,0.3)',
            rangeslider=dict(visible=False)
        ),
    )
    
    return fig


def extract_indicator_data(df):
    """Extract individual indicator DataFrames from combined DataFrame"""
    # Get all columns
    all_cols = df.columns.tolist()
    
    # Define indicator patterns
    indicator_patterns = {
        "fvg": ["fvg", "fvg_top", "fvg_bottom", "fvg_mitigated_index"],
        "swing_highs_lows": ["swing_high_low", "swing_high_low_level"],
        "bos_choch": ["bos", "bos_choch", "bos_level", "bos_broken"],
        "ob": ["ob", "ob_top", "ob_bottom", "ob_volume", "ob_mitigated_index", "ob_percentage"],
        "previous_high_low": ["phl_previous_high", "phl_previous_low", "phl_broken_high", "phl_broken_low"],
        "sessions": ["session_active", "session_high", "session_low"],
        "retracements": ["retracement_direction", "retracement_current_retracement%", "retracement_deepest_retracement%"],
        "algorithmic_order_block": ["aob_type", "aob_top", "aob_bottom", "aob_strength"],
        "breaker_block": ["bb_type", "bb_level", "bb_strength"],
        "mitigation_block": ["mb_type", "mb_top", "mb_bottom", "mb_fvg_index"],
        "bpr": ["bpr_top", "bpr_bottom", "bpr_strength"],
        "liquidity_swing_hl": ["lshl_type", "lshl_level", "lshl_swing_index", "lshl_swept"],
    }
    
    # Also check for smc_ prefixed versions
    smc_indicator_patterns = {}
    for indicator, patterns in indicator_patterns.items():
        smc_patterns = [f"smc_{pattern}" for pattern in patterns]
        smc_indicator_patterns[indicator] = patterns + smc_patterns
    
    # Extract indicator data
    indicator_data = {}
    
    for indicator, patterns in smc_indicator_patterns.items():
        # Find columns that match any pattern
        matching_cols = []
        for col in all_cols:
            col_lower = col.lower()
            for pattern in patterns:
                if pattern.lower() in col_lower:
                    matching_cols.append(col)
                    break
        
        if matching_cols:
            # Remove duplicates while preserving order
            matching_cols = list(dict.fromkeys(matching_cols))
            indicator_data[indicator] = df[matching_cols].copy()
    
    return indicator_data


def validate_path(path):
    """Validate if the path is valid and writable"""
    try:
        path_obj = Path(path)
        parent_dir = path_obj.parent
        if not parent_dir.exists():
            return False, f"Directory does not exist: {parent_dir}"
        if not os.access(parent_dir, os.W_OK):
            return False, f"Directory is not writable: {parent_dir}"
        extension = path_obj.suffix.lower()
        if extension not in ['.png', '.gif']:
            return False, f"Invalid file extension: {extension}. Must be .png or .gif"
        return True, None
    except Exception as e:
        return False, f"Invalid path: {str(e)}"


def plot(df, smc_data=None, return_type='png', save_path=None, width=1920, height=1080, 
         window_size=100, duration=0.1):
    """
    Main plotting function for SMC charts - works for both PNG and GIF
    
    Parameters:
    df: DataFrame - OHLCV data with datetime index
    smc_data: DataFrame, list of DataFrames, or dict - SMC indicator output(s)
              Can pass single or multiple indicator DataFrames
              If None, will extract indicators from df
    return_type: str - 'png' or 'gif'
    save_path: str or None - Path to save the file
    width: int - chart width
    height: int - chart height
    window_size: int - number of candles per frame (for GIF only)
    duration: float - duration of each frame in seconds (for GIF only)
    
    Returns:
    - For PNG without save_path: returns plotly Figure object
    - For PNG with save_path: returns True on success, False on failure
    - For GIF without save_path: returns list of frame arrays
    - For GIF with save_path: returns True on success, False on failure
    """
    if return_type not in ['png', 'gif']:
        print(f"Error: Invalid return_type '{return_type}'. Must be 'png' or 'gif'")
        return False
    
    # Ensure we have OHLC columns
    required_cols = ["open", "high", "low", "close"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Extract OHLC data (ensure it has index)
    if hasattr(df, 'index'):
        ohlc_df = df[required_cols]
    else:
        ohlc_df = pd.DataFrame({
            'open': df['open'],
            'high': df['high'],
            'low': df['low'],
            'close': df['close']
        })
        if 'datetime' in df.columns:
            ohlc_df.index = pd.to_datetime(df['datetime'])
        elif 'time' in df.columns:
            ohlc_df.index = pd.to_datetime(df['time'])
        else:
            # Create a simple index if no datetime column
            ohlc_df.index = pd.RangeIndex(len(ohlc_df))
    
    # Prepare smc_data_list
    if smc_data is None:
        # Extract indicator data from df
        indicator_data = extract_indicator_data(df)
        smc_data_list = list(indicator_data.values())
        print(f"Extracted {len(smc_data_list)} indicator groups from DataFrame")
    elif isinstance(smc_data, pd.DataFrame):
        smc_data_list = [smc_data]
    elif isinstance(smc_data, list):
        smc_data_list = smc_data
    elif isinstance(smc_data, dict):
        smc_data_list = list(smc_data.values())
    else:
        smc_data_list = [smc_data]
    
    if len(smc_data_list) == 0:
        print("Warning: smc_data is empty. Only candlestick chart will be displayed.")
    
    if save_path is not None:
        is_valid, error_msg = validate_path(save_path)
        if not is_valid:
            print(f"Error: {error_msg}")
            return False
    
    # Generate PNG
    if return_type == 'png':
        fig = create_figure_with_smc(ohlc_df, smc_data_list, width=width, height=height)
        
        if save_path:
            try:
                fig.write_image(save_path)
                print(f"PNG saved to {save_path}")
                return True
            except Exception as e:
                print(f"Error saving PNG: {str(e)}")
                return False
        else:
            return fig
    
    # Generate GIF
    elif return_type == 'gif':
        gif_frames = []
        total_frames = len(df) - window_size
        
        if total_frames <= 0:
            print(f"Error: DataFrame has {len(df)} rows, need at least {window_size + 1} for GIF")
            return False
        
        print(f"Creating GIF with {total_frames} frames...")
        
        for pos in range(window_size, len(df)):
            if (pos - window_size) % 10 == 0:
                print(f"Processing frame {pos - window_size + 1}/{total_frames}")
            
            # Create windowed data
            window_df = ohlc_df.iloc[pos - window_size:pos].copy()
            window_start = pos - window_size
            
            # Slice the indicator data for this window
            window_smc_data_list = []
            for indicator_data in smc_data_list:
                try:
                    if isinstance(indicator_data, pd.DataFrame):
                        sliced = indicator_data.iloc[window_start:pos].reset_index(drop=True)
                        window_smc_data_list.append(sliced)
                    else:
                        window_smc_data_list.append(indicator_data)
                except Exception as e:
                    print(f"Warning: Could not slice indicator: {str(e)}")
                    window_smc_data_list.append(indicator_data)
            
            # Reset index for window_df
            window_df = window_df.reset_index(drop=True)
            window_df.index = range(len(window_df))
            
            fig = create_figure_with_smc(window_df, window_smc_data_list, width=width, height=height)
            
            # Convert to image array
            try:
                fig_bytes = fig.to_image(format="png")
                fig_buffer = BytesIO(fig_bytes)
                fig_image = Image.open(fig_buffer)
                gif_frames.append(np.array(fig_image))
            except Exception as e:
                print(f"Warning: Failed to convert frame to image: {str(e)}")
                continue
        
        if save_path:
            try:
                print("Saving GIF...")
                imageio.mimsave(save_path, gif_frames, duration=duration)
                print(f"GIF saved to {save_path}")
                return True
            except Exception as e:
                print(f"Error saving GIF: {str(e)}")
                return False
        else:
            return gif_frames


# Alternative simpler interface for when you have the combined DataFrame
def plot_from_combined(df, return_type='png', save_path=None, width=1920, height=1080):
    """
    Simplified plotting function when you have the combined DataFrame with all indicators
    """
    # Extract indicators from combined DataFrame
    indicator_data = extract_indicator_data(df)
    
    print(f"Found {len(indicator_data)} indicator groups:")
    for ind in indicator_data.keys():
        print(f"  - {ind}")
    
    # Create OHLC DataFrame
    ohlc_df = df[['open', 'high', 'low', 'close']].copy()
    if hasattr(df, 'index'):
        ohlc_df.index = df.index
    elif 'datetime' in df.columns:
        ohlc_df.index = pd.to_datetime(df['datetime'])
    elif 'time' in df.columns:
        ohlc_df.index = pd.to_datetime(df['time'])
    else:
        # Create a simple index if no datetime column
        ohlc_df.index = pd.RangeIndex(len(ohlc_df))
    
    # Call the main plot function
    return plot(
        df=ohlc_df,
        smc_data=list(indicator_data.values()),
        return_type=return_type,
        save_path=save_path,
        width=width,
        height=height
    )