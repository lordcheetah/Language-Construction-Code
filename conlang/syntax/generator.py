"""Roll a typologically plausible set of syntax parameters.

The basic constituent order is sampled from roughly its cross-linguistic proportions
(SOV and SVO together cover the great majority of languages; verb-initial orders are
uncommon and object-initial orders are rare). The remaining parameters are then derived
as harmonic correlates of that order with realistic noise (see
:func:`conlang.syntax.parameters.derive_correlates`).
"""

from __future__ import annotations

import random

from conlang.syntax.parameters import WordOrder, SyntaxParameters, derive_correlates

# Approximate cross-linguistic proportions of basic word order (Dryer / WALS-flavoured).
_ORDER_WEIGHTS: dict[WordOrder, float] = {
    WordOrder.SOV: 0.45,
    WordOrder.SVO: 0.42,
    WordOrder.VSO: 0.09,
    WordOrder.VOS: 0.03,
    WordOrder.OVS: 0.008,
    WordOrder.OSV: 0.002,
}


def random_syntax(rng: random.Random | None = None) -> SyntaxParameters:
    rng = rng or random.Random()
    orders = list(_ORDER_WEIGHTS)
    weights = [_ORDER_WEIGHTS[o] for o in orders]
    basic = rng.choices(orders, weights=weights, k=1)[0]
    return derive_correlates(basic, rng)
