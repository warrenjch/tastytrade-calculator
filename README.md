# tastytrade-calculator
A calculator and visualizer for option metrics using tastytrade's API. Requires a tastytrade account to use.

This is the public version of a proprietary tool and will only be updated if new major features are added to the original.

(Latest update: 2025-07-13, estimated next update: end Jul 2025)

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

1. (Prio: high) Next update will include a calculator for greeks.
2. (Prio: high) Next update will include a term structure visualizer, which already exists in the private version but needs to be adapted.
3. (Prio: high) Currently only a calendar time method is supported, which makes near-dated pricing inaccurate when non trading hours are considered. Future updates will extend this to trading time methods and possibly other vol-time estimators to allow for more robust calculations and support for 0dte.
4. (Prio: medium) I'm planning to add Gatheral's SVI to at some point, after which a new mode will be added to the implied distribution plotter (currently 'breedenlitzenberger' is accepted as an input but returns nothing).
5. (Prio: medium) The IV calculator will fail for highly illiquid options. See example.ipynb for such a case. Future updates will try to address this in one of two ways:
   - Cutting off calculations where BS returns invalid values
   - Interpolating illiquid IV values
6. (Prio: low) The get_rfr method only supports getting the 13-week T-bill rate. In the far future this will be extended to a full yield curve, but for most cases this is really a big hassle for very little difference.
7. (Prio: low) The ATM term structure visualizer will be extended with earnings vol visualizations to help with calculating vol attribution around earnings.