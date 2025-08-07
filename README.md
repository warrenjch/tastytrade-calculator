# tastytrade-calculator
A calculator and visualizer for option metrics using tastytrade's API. Requires a tastytrade account to use.

(Latest update: 2025-08-07, estimated next update: mid Sept 2025)

## Changelog

1. Added a calculator for the more important greeks (delta, gamma, vega, theta).
2. Added the ATM term structure visualizer.
3. A full yield curve can now be extracted using the Treasury website's data. However, using time-variable rates for rfr params is not yet supported.
4. Minor changes to time methods.

## Setup

1. Clone and install dependencies
2. Edit the 'config.json' file to replace the values for "login", "password", and "account-numbers" with the username, password, and account number of your tastytrade account. "cert" is used for a sandbox account, which can be created using an account on tastytrade's dev portal. "prod" is used for your live account.

## Usage

To start a streamer session, run the following code:
```python
session=None
account=None

async def setup():
    global session, account, config
    # alternatively, return session from this function
    config = Config(test=False)
    session = Session(config.username, config.password, is_test=config.test)
    account = await Account.a_get(session, config.account_number)

asyncio.run(setup())
print("Session and account setup complete.")

streamer = DXLinkStreamer(session)
```
Note that if this is run on a jupyter notebook, nest_asyncio needs to be imported and applied before running this.

For example, to get the options chain as a dataframe for a specific symbol and expiration date / DTE:
```python
chain = OptionMethods.get_current_chain(session, 'SPY', dte=14)
```

This type of dataframe is used as an input in the various calculators. You can use it to find the IV values, plot the IV surface, or find the implied distribution for a given expiry.

You can also use the "HistoricalMethods" class to pull historical daily data from yfinance in order to compare it to the option implied distribution. For example:
```python
ticker = 'IBIT'
dte = 7
chain = OptionMethods.get_current_chain(session, ticker, dte=dte)
rfr = Auxiliary.get_rfr()
fig, chain = OptionMethods.plot_implied_dist(chain, dte, rfr, symbol=ticker, gaussian_sigma=2)
fig = Historical.plot_return_distribution(ticker=ticker, period=dte, mode='CC', time_method='calendar', start_date='2025-01-01', end_date='2025-07-10', fig=fig)

plt.show()
```

All plotting methods return a matplotlib figure object, which can be used to further customize the plot or pass through to other plotting functions in the "fig" variable.

## Planned features and known bugs

1. (Prio: high) Currently only a calendar time method is supported, which makes near-dated pricing inaccurate when non trading hours are considered. Future updates will extend this to trading time methods and possibly other vol-time estimators to allow for more robust calculations and support for 0dte. UPDATE: this could either be an adjustable vol calendar or a method for "cleaning" calendar vols, or I might include both.
2. NEW: (Prio: medium) In the future, the YieldCurve object will be a valid input for rfr params (which would allow for flexible rfr calculation across time).
3. (Prio: medium) I'm planning to add Gatheral's SVI to at some point, after which a new mode will be added to the implied distribution plotter (currently 'breedenlitzenberger' is accepted as an input but returns nothing).
4. (Prio: medium) The IV calculator will fail for highly illiquid options. See example.ipynb for such a case. Future updates will try to address this in one of two ways:
   - Cutting off calculations where BS returns invalid values
   - Interpolating illiquid IV values
5. (Prio: low) The ATM term structure visualizer will be extended with earnings vol visualizations to help with calculating vol attribution around earnings.
6. NEW: (Prio: low) Historical Candle objects from tastytrade data may allow us to see the historical price evolution for existing options at a low frequency (daily) if the options have not yet expired. If this works, it would allow us to compare IV surfaces across time. Planned features under this: IV surface history, ATM vol history, vol move decomposition.