import pandas as pd
import numpy as np
from tastytrade.instruments import *
from tastytrade.market_data import *
from itertools import chain
from scipy.stats import norm, gaussian_kde
from scipy.optimize import brentq
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline, UnivariateSpline
import requests
import yfinance as yf
from curl_cffi import requests as yfrequests
import matplotlib.pyplot as plt


class OptionMethods:

    @staticmethod
    def batch_findoption(session, items, func, batch_size=90):
        # generator of batches
        batches = (
            items[i: i + batch_size]
            for i in range(0, len(items), batch_size)
        )
        return list(chain.from_iterable(func(session, [option.symbol for option in batch]) for batch in batches))

    @staticmethod
    def fetch_options(session, option_batch):
        optiondata = get_market_data_by_type(session, options=option_batch)
        return optiondata

    @staticmethod
    def convertchain(session, chain):
        all_results = OptionMethods.batch_findoption(session, chain, OptionMethods.fetch_options, batch_size=90)

        res_map = {res.symbol: res for res in all_results}

        combined = [
            {
                "symbol": opt.symbol,
                "strike_price": float(opt.strike_price),
                "option_type": opt.option_type,
                "bid": float(getattr(res_map.get(opt.symbol, None), "bid", np.nan)),
                "ask": float(getattr(res_map.get(opt.symbol, None), "ask", np.nan)),
            }
            for opt in chain
        ]

        df = pd.DataFrame(combined)
        vc = df["strike_price"].value_counts()
        # find any strikes that don’t have exactly 2 entries
        wrong = vc[vc != 2]

        # assert, printing out the offending strike_price values
        if not wrong.empty:
            print(f"Strikes without exactly two entries: {wrong.index.tolist()}")
            df = df[~df["strike_price"].isin(wrong.index)]

        df["type"] = df["option_type"].apply(
            lambda ot: "C" if ot == OptionType.CALL
            else "P" if ot == OptionType.PUT
            else (_ for _ in ()).throw(ValueError(f"Unexpected OptionType: {ot!r}"))
        )

        df_pivot = (
            df
            .pivot(index="strike_price", columns="type", values=["bid", "ask"])
            .reset_index()
        )

        df_pivot.columns = [
            f"{col_type.lower()}{val}" if isinstance(col_type, str)
            else col_type  # this picks up the strike_price index as-is
            for val, col_type in df_pivot.columns
        ]

        # 7. Rename strike_price → strike, reorder
        dfchain = (
            df_pivot
            .rename(columns={"strike_price": "strike"})
            [["cbid", "cask", "strike", "pbid", "pask"]]
        )
        return dfchain

    @staticmethod
    def find_atmf_strike(chain):
        c = chain.copy()
        c['cmid'] = (c['cbid'] + c['cask']) / 2
        c['pmid'] = (c['pbid'] + c['pask']) / 2
        atmfs = c.loc[abs(c['cmid'] - c['pmid']).idxmin(), "strike"]
        return atmfs

    @staticmethod
    def get_current_chain(session, symbol: str, exp: pd.Timestamp = None, dte: int = None):
        if dte is None and exp is None:
            raise ValueError("Either dte or exp must be provided")
        elif dte is not None and exp is not None:
            raise ValueError("Only one of dte or exp should be provided")
        elif dte is not None and exp is None:
            exp = datetime.now() + pd.Timedelta(days=dte)
        options = get_option_chain(session, symbol)
        expiries = list(options.keys())
        if exp.date() not in expiries:
            raise ValueError(f"No options for {symbol} expiring on {exp.date()}, available expirations: {expiries}")
        chain = options[exp.date()]
        chain = OptionMethods.convertchain(session, chain)
        return chain

    @staticmethod
    def find_ivs(chain, dte, rfr, div_yield=0, gaussian_sigma=2, time_method='calendar'):
        '''
        chain: pd DF with columns: cbid, cask, pbid, pask, strike
        dte in days, currently only supports "dirty IV" ie /365
        rfr in percentage points, ie DO NOT normalize before passing
        div yield in percentage points
        gaussian_sigma is the sigma for the gaussian filter applied to the IVs
        '''
        if 'iv' in chain.columns:
            print('IVs already present in chain')
            return chain

        T = TimeMethods.calculate_tte(dte, time_method=time_method)
        rfr /= 100
        div_yield /= 100

        df = chain.copy()
        if 'cmid' not in df.columns or 'pmid' not in df.columns:
            df['cmid'] = (df['cbid'] + df['cask']) / 2
            df['pmid'] = (df['pbid'] + df['pask']) / 2

        K_atm = OptionMethods.find_atmf_strike(df)
        atm_row = df.loc[df['strike'] == K_atm].iloc[0]
        cmid_atm, pmid_atm = atm_row['cmid'], atm_row['pmid']
        F = K_atm + np.exp((rfr-div_yield) * T) * (cmid_atm - pmid_atm)

        civ_list, piv_list = [], []
        for _, row in df.iterrows():
            K = row['strike']
            civ = OptionMethods.bs_iv(row['cmid'], F, K, T, rfr, is_call=True)
            piv = OptionMethods.bs_iv(row['pmid'], F, K, T, rfr, is_call=False)
            civ_list.append(civ)
            piv_list.append(piv)

        df['civ'] = civ_list
        df['piv'] = piv_list
        if gaussian_sigma > 0:
            df['civ'] = gaussian_filter1d(df['civ'], sigma=gaussian_sigma, mode='nearest')
            df['piv'] = gaussian_filter1d(df['piv'], sigma=gaussian_sigma, mode='nearest')
        return df

    @staticmethod
    def bs_call(F, K, T, sigma, rfr):
        if sigma <= 0 or T <= 0:
            return max(F - K, 0.0) * np.exp(-rfr * T)
        d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return np.exp(-rfr * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))

    @staticmethod
    def bs_put(F, K, T, sigma, rfr):
        if sigma <= 0 or T <= 0:
            return max(K - F, 0.0) * np.exp(-rfr * T)
        d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return np.exp(-rfr * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

    @staticmethod
    def bs_iv(price, F, K, T, rfr, is_call=True):
        if price < 1e-8 or T <= 0:
            return 0.0

        def objective(sigma):
            if is_call:
                return OptionMethods.bs_call(F, K, T, sigma, rfr) - price
            else:
                return OptionMethods.bs_put(F, K, T, sigma, rfr) - price

        try:
            iv = brentq(objective, 1e-6, 5.0, maxiter=500)
        except ValueError:
            iv = 0.0
        return iv * 100

    @staticmethod
    def plot_iv(session, symbol: str, exp: pd.Timestamp, bounds=None, moneyness=False, time_method='calendar', gaussian_sigma=2, fig=None):
        options = get_option_chain(session, symbol)
        chain = options[exp.date()]
        chain = OptionMethods.convertchain(session, chain)
        dte, rfr = TimeMethods.calendar_dte(datetime.now(), exp), Auxiliary.get_rfr()
        chain = OptionMethods.find_ivs(chain, dte, rfr, gaussian_sigma=gaussian_sigma, time_method=time_method)
        atmf = OptionMethods.find_atmf_strike(chain)
        chain['iv'] = np.where(chain['strike'] >= atmf, chain['civ'], chain['piv'])
        if fig is None:
            fig = plt.figure(figsize=(12, 8))
            fig.moneyness = moneyness
            if moneyness == True:
                plt.xlabel('Moneyness')
            else:
                plt.xlabel('Strike')
            plt.title(f'{datetime.now().date()} IV')
            plt.ylabel('IV')
        else:
            plt.figure(fig.number)
            if not hasattr(fig, 'moneyness'):
                fig.moneyness = moneyness
        if fig.moneyness == True:
            chain['strike'] /= atmf
        else:
            atmfiv = chain.loc[chain['strike'] == atmf, 'iv'].iloc[0]
            plt.scatter(atmf, atmfiv, marker='o', color='red', s=30, zorder=5)
        if bounds is None:
            plt.plot(chain['strike'], chain['iv'], label=f'{symbol} {exp.date()} | ATMF: {round(atmf)}')
        else:
            lbound, ubound = bounds
            mask = chain['strike'].between(lbound, ubound)
            plt.plot(chain.loc[mask, 'strike'], chain.loc[mask, 'iv'], label=f'{symbol} {exp.date()} | ATMF: {round(atmf)}')
        plt.legend()
        return fig

    @staticmethod
    def plot_implied_dist(chain, dte: int, rfr, symbol:str, mode="butterfly", gaussian_sigma=2, time_method='calendar', fig=None):
        '''
        chain needs to come in as a df with ba and strike, dte in days, rfr pre normalizing
        '''
        rfr /= 100
        if fig is None:
            fig = plt.figure(figsize=(12, 8))
            fig.moneyness = True
            plt.title(f'{symbol} implied distribution')
            plt.xlabel('Return')
            plt.ylabel('Probability')
        else:
            plt.figure(fig.number)
            if not hasattr(fig, 'moneyness'):
                fig.moneyness = True

        if mode == "butterfly":
            atmf = OptionMethods.find_atmf_strike(chain)
            chain = chain.sort_values(by='strike').reset_index(drop=True)
            if 'cmid' not in chain.columns or 'pmid' not in chain.columns:
                chain['cmid'] = (chain['cbid'] + chain['cask']) / 2
                chain['pmid'] = (chain['pbid'] + chain['pask']) / 2
            T = TimeMethods.calculate_tte(dte, time_method=time_method)
            K = chain['strike'].values
            Cc = chain['cmid'].values
            Cp = chain['pmid'].values
            N = len(K)
            d2C = np.zeros(N)
            for i in range(1, N - 1):
                dk_fwd = K[i + 1] - K[i]
                dk_bwd = K[i] - K[i - 1]
                if K[i] >= atmf:
                    delta_fwd = (Cc[i + 1] - Cc[i]) / dk_fwd
                    delta_bwd = (Cc[i] - Cc[i - 1]) / dk_bwd
                else:
                    delta_fwd = (Cp[i + 1] - Cp[i]) / dk_fwd
                    delta_bwd = (Cp[i] - Cp[i - 1]) / dk_bwd
                d2C[i] = 2 * (delta_fwd - delta_bwd) / (dk_fwd + dk_bwd)
            if gaussian_sigma >0:
                d2C = gaussian_filter1d(d2C, sigma=gaussian_sigma, mode='nearest')

            chain['density'] = np.exp(rfr * T) * d2C
            chain['strike'] /= atmf
            chain['density'] /= np.trapezoid(chain['density'].values, chain['strike'].values)
            plt.plot(chain['strike'], chain['density'], label=f'{symbol} {dte}d implied distribution')
            plt.grid(True)
            plt.legend()
            return fig, chain
        elif mode == "breedenlitzenberger":
            return None
        else:
            raise ValueError(f"Unsupported mode: {mode}, supported modes are 'butterfly' and 'breedenlitzenberger'.")

class Historical:
    @staticmethod
    def plot_return_distribution(ticker: str, period: int, mode: str = 'OC', time_method: str = 'calendar', start_date: str = None, end_date: str = None, fig=None):
        mode = mode.upper()
        if mode not in ['OC', 'CC']:
            raise ValueError("mode must be OC for open-close or CC for close-close")
        if (mode == 'CC' and period < 1) or (mode == 'OC' and period < 0):
            raise ValueError(f"period {period} not supported for {mode} mode")
        yfsession = yfrequests.Session(impersonate="safari")

        df = yf.download(ticker, start=start_date, end=end_date, interval='1d', auto_adjust=False, session=yfsession, progress=False)
        if df.empty:
            raise RuntimeError(f"No data fetched for {ticker}")
        df.index = pd.to_datetime(df.index)
        dates_index = df.index

        records = []
        for i in range(len(df)):
            t = dates_index[i]
            row_start = df.iloc[i]
            start_price = row_start['Open'].item() if mode == 'OC' else row_start['Close'].item()
            if period == 0 and mode == 'OC':
                end_price = row_start['Close'].item()
            else:
                target_ts = t + pd.Timedelta(days=period)
                pos = dates_index.searchsorted(target_ts)
                if pos < len(df) and dates_index[pos].normalize() == target_ts.normalize():
                    end_price = df.iloc[pos]['Close'].item()
                else:
                    prev_pos = pos - 1
                    if prev_pos >= 0 and pos < len(df):
                        prev_close = df.iloc[prev_pos]['Close'].item()
                        next_open = df.iloc[pos]['Open'].item()
                        end_price = (prev_close + next_open) / 2
                    else:
                        continue
            pct = (end_price / start_price)
            records.append({'Date': t.date(), 'Return': pct})

        returns_df = pd.DataFrame(records)
        if returns_df.empty:
            raise RuntimeError("No valid return records to compute distribution.")
        returns_df.set_index('Date', inplace=True)

        data = returns_df['Return'].values
        kde = gaussian_kde(data, bw_method='scott')  # or adjust bw_method

        xs = np.linspace(data.min(), data.max(), 100)
        ys = kde(xs)

        if fig is None:
            fig = plt.figure(figsize=(12, 8))
            plt.title("Return Distribution")
            plt.xlabel("Return")
            plt.ylabel("Density")
            plt.grid(True)
        else:
            plt.figure(fig.number)
        calc_mode = 'Open-Close' if mode == 'OC' else 'Close-Close'
        plt.plot(xs, ys, label = f"{ticker} {period}-day ({time_method}) {calc_mode}")
        plt.legend()

        return fig #spline, returns_df, hist_df if want to see inner data

class TimeMethods:

    @staticmethod
    def calendar_dte(start, end):
        if not isinstance(start, pd.Timestamp):
            start = pd.to_datetime(start)
        if not isinstance(end, pd.Timestamp):
            end = pd.to_datetime(end)
        if start > end:
            return None
        else:
            return (end - start).days

    @staticmethod
    def unix_to_time(t: int, granularity: str = 'd', scaled: bool = True):
        if scaled:
            dt = datetime.fromtimestamp(t / 1000)
        else:
            dt = datetime.fromtimestamp(t)

        if granularity == 'd':
            t = date(dt.year, dt.month, dt.day)
            return t
        elif granularity == 's':
            return dt
        else:
            raise ValueError(f'invalid granularity: {granularity}')

    @staticmethod
    def calculate_tte(dte, time_method='calendar'):
        if time_method == 'calendar':
            return dte / 365
        else:
            raise ValueError(f"Unsupported time_method: {time_method}")

class Auxiliary:
    @staticmethod
    def get_rfr():
        yfsession = yfrequests.Session(impersonate="safari")
        try:
            treasury_bill = yf.Ticker("^IRX", session=yfsession)
            hist = treasury_bill.history(period="1d")
            rfr = float(hist['Close'].iloc[-1])
        except Exception as e:
            rfr = 4.2
        return rfr