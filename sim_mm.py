import dataclasses
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd

@dataclass
class MMConfig:
    base_spread: float = 0.04          #baseline quoted spread in price space
    min_spread: float = 0.01           #never quote tighter than this
    max_spread: float = 0.20           #and never wider than this
    inventory_limit: float = 500.0     #max absolute YES shares we'll hold
    trade_size: float = 10.0           #size of each quote on each side
    inventory_risk_aversion: float = 0.001  #how aggressively we skew for inventory
    vol_spread_coeff: float = 0.005    #how much volatility widens the spread
    step_dt_hours: float = 0.03        #~2 minutes per step
    sim_hours: float = 3.0             #total horizon per market
    fill_prob_base: float = 0.08       #background chance a quote gets lifted
    fill_prob_informed: float = 0.25   #extra chance when next move is in that quote's disadvantage
    tick_size: float = 0.01            #price grid
    seed: int = 0

@dataclass
class MMState:
    t: int
    time_hours: float
    mid: float
    inventory: float
    cash: float
    pnl: float

@dataclass
class Quote:
    t: int
    time_hours: float
    bid: float
    ask: float
    size_bid: float
    size_ask: float
    spread: float


@dataclass
class Fill:
    t: int
    time_hours: float
    side: str   #'buy' means we buy YES at our bid, 'sell' means we sell YES at our ask
    price: float
    size: float
    mid_before: float
    mid_after: float


class SimplePolymarketMM:
    def __init__(self, config: Optional[MMConfig] = None):
        self.cfg = config or MMConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

    #quoting logic
    def compute_spread(self, sigma: float) -> float:
        spread = self.cfg.base_spread + self.cfg.vol_spread_coeff * sigma
        spread = max(self.cfg.min_spread, min(self.cfg.max_spread, spread))
        return spread

    def quote_levels(self, mid: float, inventory: float, sigma: float) -> Tuple[float, float]:
        """Return (bid, ask) given mid, inventory and volatility.
        Inventory skew: if we are long YES (inventory>0) we want to
        *buy less / sell more*, so we move the center of our quotes
        slightly down; if short we move it up.
        """
        spread = self.compute_spread(sigma)
        half = spread / 2.0

        #simple linear inventory skew in price space
        inv_skew = self.cfg.inventory_risk_aversion * inventory
        center = mid - inv_skew

        bid = center - half
        ask = center + half

        #clip to [tick, 1 - tick]
        bid = max(self.cfg.tick_size, min(1.0 - self.cfg.tick_size, bid))
        ask = max(self.cfg.tick_size, min(1.0 - self.cfg.tick_size, ask))

        #ensure bid <= ask
        if bid > ask:
            bid = ask = max(self.cfg.tick_size, min(1.0 - self.cfg.tick_size, center))

        return bid, ask

    def quote_sizes(self, inventory: float) -> Tuple[float, float]:
        #Return (size_bid, size_ask) based on how close we are to limits.
        w = max(0.0, 1.0 - abs(inventory) / self.cfg.inventory_limit)
        base = self.cfg.trade_size * w

        size_bid = size_ask = base

        #if we are very long, stop bidding (only sell to reduce risk)
        if inventory > self.cfg.inventory_limit:
            size_bid = 0.0
        #if we are very short, stop asking
        if inventory < -self.cfg.inventory_limit:
            size_ask = 0.0

        return size_bid, size_ask

    #price process
    def simulate_price_path(self, start_mid: float, sigma: float) -> np.ndarray:
        """Very simple bounded random walk in price space.
        This is NOT a realistic Polymarket model but is enough to stress-test
        inventory + spread logic.
        """
        steps = int(self.cfg.sim_hours / self.cfg.step_dt_hours)
        mids = np.empty(steps + 1)
        mids[0] = start_mid

        # convert a scale like the 3h volatility metric to a step size
        # we just normalise to something small so the process stays in [0,1]
        step_scale = 0.005 + 0.0005 * sigma

        for t in range(steps):
            d = self.rng.normal(scale=step_scale)
            mids[t + 1] = np.clip(mids[t] + d, self.cfg.tick_size, 1.0 - self.cfg.tick_size)

        return mids

    #fill model
    def simulate_fills_for_step(
        self,
        t: int,
        time_hours: float,
        mid_before: float,
        mid_after: float,
        bid: float,
        ask: float,
        size_bid: float,
        size_ask: float,
    ) -> List[Fill]:
        """Simulate which of our quotes get hit in this step.
        Very stylised model:
        * background noise flow hits each side with probability fill_prob_base
        * if the next mid price is higher than current, takers are more likely
          to lift our ask (adverse selection); similarly for the bid.
        """
        fills: List[Fill] = []

        direction = np.sign(mid_after - mid_before)

        p_ask = self.cfg.fill_prob_base
        p_bid = self.cfg.fill_prob_base

        if direction > 0:   #price going up -> more buys at our ask
            p_ask += self.cfg.fill_prob_informed
        elif direction < 0: #price going down -> more sells at our bid
            p_bid += self.cfg.fill_prob_informed

        #cap at 1
        p_ask = min(1.0, max(0.0, p_ask))
        p_bid = min(1.0, max(0.0, p_bid))

        u1 = self.rng.random()
        if u1 < p_ask and size_ask > 0:
            fills.append(Fill(t, time_hours, "sell", ask, size_ask, mid_before, mid_after))

        u2 = self.rng.random()
        if u2 < p_bid and size_bid > 0:
            fills.append(Fill(t, time_hours, "buy", bid, size_bid, mid_before, mid_after))

        return fills

    #main simulation
    def simulate_single_market(
        self,
        market_row: pd.Series,
        sigma_3h: float,
    ) -> Dict[str, pd.DataFrame]:
        """Run a backtest for one market.
        Returns a dict of dataframes:
        * states: inventory, cash, pnl over time
        * quotes: all quote updates
        * fills: all simulated fills
        """
        start_mid = float((market_row["best_bid"] + market_row["best_ask"]) / 2.0)

        mids = self.simulate_price_path(start_mid=start_mid, sigma=sigma_3h)
        steps = len(mids) - 1

        inventory = 0.0
        cash = 0.0

        states: List[MMState] = []
        quotes: List[Quote] = []
        fills_all: List[Fill] = []

        for t in range(steps):
            time_hours = t * self.cfg.step_dt_hours

            mid_now = mids[t]
            mid_next = mids[t + 1]

            bid, ask = self.quote_levels(mid_now, inventory, sigma_3h)
            size_bid, size_ask = self.quote_sizes(inventory)

            #record quote
            quotes.append(
                Quote(
                    t=t,
                    time_hours=time_hours,
                    bid=bid,
                    ask=ask,
                    size_bid=size_bid,
                    size_ask=size_ask,
                    spread=ask - bid,
                )
            )

            #simulate which quotes are hit
            step_fills = self.simulate_fills_for_step(
                t=t,
                time_hours=time_hours,
                mid_before=mid_now,
                mid_after=mid_next,
                bid=bid,
                ask=ask,
                size_bid=size_bid,
                size_ask=size_ask,
            )

            for fill in step_fills:
                if fill.side == "sell":
                    #we sell YES at our ask -> inventory decreases, cash increases
                    inventory -= fill.size
                    cash += fill.size * fill.price
                else:  #buy
                    inventory += fill.size
                    cash -= fill.size * fill.price

            fills_all.extend(step_fills)

            pnl = cash + inventory * mid_next
            states.append(
                MMState(
                    t=t,
                    time_hours=time_hours,
                    mid=mid_next,
                    inventory=inventory,
                    cash=cash,
                    pnl=pnl,
                )
            )

        states_df = pd.DataFrame([dataclasses.asdict(s) for s in states])
        quotes_df = pd.DataFrame([dataclasses.asdict(q) for q in quotes])
        fills_df = pd.DataFrame([dataclasses.asdict(f) for f in fills_all])

        return {"states": states_df, "quotes": quotes_df, "fills": fills_df}

    def simulate_multiple_markets(
        self,
        markets: pd.DataFrame,
        n_markets: int = 5,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Pick n_markets markets and run the backtest on each.
        markets is expected to have columns:
        * best_bid, best_ask
        * 3_hour (volatility proxy)
        * market_slug or question to identify
        """
        results: Dict[str, Dict[str, pd.DataFrame]] = {}

        chosen = markets.sample(n=n_markets, random_state=self.cfg.seed)
        for idx, row in chosen.iterrows():
            ident = row.get("market_slug", None) or row.get("question", None) or str(idx)
            sigma_3h = float(row.get("3_hour", 1.0))
            res = self.simulate_single_market(row, sigma_3h=sigma_3h)
            results[ident] = res

        return results


def summarize_results(results: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    #Create one summary row per market from the simulation outputs.
    summaries = []
    for ident, res in results.items():
        states = res["states"]
        quotes = res["quotes"]

        if states.empty:
            continue

        final_pnl = float(states["pnl"].iloc[-1])
        max_inv = float(states["inventory"].abs().max())
        avg_spread = float(quotes["spread"].mean())
        traded_spread = float(res["fills"]["price"].mean()) if not res["fills"].empty else float("nan")
        n_trades = int(len(res["fills"]))

        summaries.append(
            {
                "market": ident,
                "final_pnl": final_pnl,
                "max_inventory": max_inv,
                "avg_quoted_spread": avg_spread,
                "avg_fill_price": traded_spread,
                "num_trades": n_trades,
            }
        )

    return pd.DataFrame(summaries)
