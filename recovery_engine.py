import logging
import config
from utils import calculate_atr

logger = logging.getLogger()

def enforce_anti_gap_sl(entry_price, sl_price, direction, analysis_df, lot_size,
                        risk_amount, contract_size, atr_multiplier=None, point=0.00001, digits=None):
    """Anti-Gap SL Protection: Widen SL to at least ATR × multiplier from entry.
    When SL is widened, lot size is reduced proportionally to keep dollar risk constant.
    
    This prevents flash-crash slippage where price gaps through a tight SL.
    If the SL is already wider than the ATR floor, nothing changes.
    
    Returns: (adjusted_sl, adjusted_lot, was_widened, original_sl_dist, new_sl_dist)
    """
    if not config.ANTI_GAP_SL_ENABLED:
        return sl_price, lot_size, False, 0, 0
    
    if atr_multiplier is None:
        atr_multiplier = config.ANTI_GAP_ATR_MULTIPLIER
    
    # Calculate ATR on the analysis window
    atr_value = calculate_atr(analysis_df, period=14)
    if atr_value <= 0:
        return sl_price, lot_size, False, 0, 0
    
    min_sl_distance = atr_value * atr_multiplier
    
    # Current SL distance
    if direction == "Buy":
        current_sl_dist = entry_price - sl_price
    else:
        current_sl_dist = sl_price - entry_price
    
    if current_sl_dist <= 0:
        return sl_price, lot_size, False, 0, 0
    
    # If SL is already wide enough, no change needed
    if current_sl_dist >= min_sl_distance:
        return sl_price, lot_size, False, current_sl_dist, current_sl_dist
    
    # Widen the SL to the ATR floor
    if direction == "Buy":
        new_sl = entry_price - min_sl_distance
    else:
        new_sl = entry_price + min_sl_distance
    
    if digits is not None:
        new_sl = round(new_sl, digits)
    
    # Reduce lot size proportionally: wider SL = smaller lot = same dollar risk
    if min_sl_distance > 0 and contract_size > 0:
        new_lot = risk_amount / min_sl_distance / contract_size
    else:
        new_lot = lot_size
    
    # Don't increase lot size (safety)
    new_lot = min(new_lot, lot_size)
    
    safe_point = point if (point and point > 0) else 0.00001
    logger.info("[ANTI-GAP] SL widened: %.5f → %.5f (dist: %.1f → %.1f pts, ATR=%.5f×%.1f). "
                "Lot reduced: %.4f → %.4f. Dollar risk unchanged.",
                sl_price, new_sl, current_sl_dist / safe_point, min_sl_distance / safe_point,
                atr_value, atr_multiplier, lot_size, new_lot)
    
    return new_sl, new_lot, True, current_sl_dist, min_sl_distance

def update_slippage_recovery(excess_loss, symbol_point=0.0001):
    """Called when a live trade closes with more loss than intended risk.
    excess_loss: positive number = how much MORE was lost beyond intended risk."""
    if not config.slippage_recovery_tracker['enabled'] or excess_loss <= 0:
        return
    tracker = config.slippage_recovery_tracker
    tracker['excess_loss_cumulative'] += excess_loss
    tracker['recovery_trades_remaining'] = tracker['recovery_spread_trades']
    tracker['lot_reduction_active'] = True
    
    # Update running average slippage
    tracker['slippage_count'] += 1
    n = tracker['slippage_count']
    tracker['avg_slippage_observed'] = (
        tracker['avg_slippage_observed'] * (n - 1) + excess_loss
    ) / n
    # Set SL buffer based on observed slippage
    tracker['sl_buffer_pips'] = tracker['avg_slippage_observed'] * tracker['sl_buffer_multiplier']
    # Temporarily elevate RRR
    tracker['elevated_rrr'] = 0.5  # Add 0.5 to min RRR during recovery
    logger.info("[RECOVERY] Slippage excess $%.2f tracked. Cumulative: $%.2f. "
                "Recovery over %d trades. SL buffer: %.5f",
                excess_loss, tracker['excess_loss_cumulative'],
                tracker['recovery_trades_remaining'], tracker['sl_buffer_pips'])

def get_recovery_adjustments(lot_size, min_rrr, sl_price, entry_price, direction, is_fixed_lot=False):
    """Apply recovery adjustments to lot size, RRR, and SL.
    Returns (adjusted_lot, adjusted_rrr, adjusted_sl)."""
    tracker = config.slippage_recovery_tracker
    adjusted_lot = lot_size
    adjusted_rrr = min_rrr
    adjusted_sl = sl_price

    if not tracker['enabled'] or not tracker['lot_reduction_active']:
        return adjusted_lot, adjusted_rrr, adjusted_sl

    if tracker['recovery_trades_remaining'] <= 0:
        # Recovery complete — reset
        tracker['lot_reduction_active'] = False
        tracker['excess_loss_cumulative'] = 0.0
        tracker['elevated_rrr'] = 0.0
        logger.info("[RECOVERY] Recovery complete. Resuming normal trading.")
        return adjusted_lot, adjusted_rrr, adjusted_sl

    # 1. Lot reduction: Apply a safe, flat 15% reduction instead of a magic number formula
    # that breaks on non-forex pairs or varying contract sizes.
    if not is_fixed_lot:
        reduction_factor = 0.85
        adjusted_lot = lot_size * reduction_factor

    # 2. Elevated RRR
    adjusted_rrr = min_rrr + tracker['elevated_rrr']

    # 3. SL buffer — widen SL to absorb slippage
    orig_sl_dist = abs(entry_price - sl_price)
    if tracker['sl_buffer_pips'] > 0:
        if direction == "Buy":
            adjusted_sl = sl_price - tracker['sl_buffer_pips']
        else:
            adjusted_sl = sl_price + tracker['sl_buffer_pips']
        
        # Scale lot size proportionally to keep risk constant with wider SL distance
        if not is_fixed_lot:
            new_sl_dist = abs(entry_price - adjusted_sl)
            if new_sl_dist > orig_sl_dist and orig_sl_dist > 0:
                scaling_factor = orig_sl_dist / new_sl_dist
                adjusted_lot = adjusted_lot * scaling_factor
                logger.info("[RECOVERY] SL widened: SL distance increased from %.5f to %.5f. Scaling lot size by factor %.4f.",
                            orig_sl_dist, new_sl_dist, scaling_factor)

    tracker['recovery_trades_remaining'] -= 1
    logger.info("[RECOVERY] Applied: lot %.4f→%.4f, RRR %.2f→%.2f, SL buffer %.5f. %d trades left.",
                lot_size, adjusted_lot, min_rrr, adjusted_rrr, tracker['sl_buffer_pips'],
                tracker['recovery_trades_remaining'])

    return adjusted_lot, adjusted_rrr, adjusted_sl


# ---------------------------------------------------------------------------
# INSTITUTIONAL RISK & PROP FIRM PROTECTION (GoodBot V3 Enhancements)
# ---------------------------------------------------------------------------



