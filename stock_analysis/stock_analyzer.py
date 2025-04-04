import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, Output, Input, callback
import dash_bootstrap_components as dbc
from datetime import date, timedelta
import logging
import os
import requests
from bs4 import BeautifulSoup
from nsepy import get_history

# Set up logging
logging.basicConfig(filename='stock_analyzer.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_index_template():
    """
    Returns the HTML content of the index.html template as a string.
    """
    return """
    <!DOCTYPE html>
    <html lang="en">
        <head>
            <meta charset="UTF-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0" />
            <title>Stock Analysis Dashboard</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <link
                href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap"
                rel="stylesheet" />
            <style>
                body {
                    font-family: 'Inter', sans-serif;
                }
            </style>
        </head>
        <body class="bg-gray-100">
            <div class="container mx-auto p-4 md:p-6 lg:p-8">
                <h1 class="text-2xl md:text-3xl lg:text-4xl font-semibold text-blue-600 text-center mb-6 md:mb-8 lg:mb-10">
                    Stock Analysis Dashboard
                </h1>
                <div class="bg-white rounded-lg shadow-md p-4 md:p-6 mb-6 space-y-4">
                    <h2 class="text-xl font-semibold text-gray-800">Select Stock</h2>
                    <div class="flex flex-col md:flex-row gap-4">
                        <input
                            type="text"
                            id="stock-input"
                            placeholder="Enter stock symbol (e.g., AAPL, GOOG, RELI.NS)"
                            class="border border-gray-300 rounded-md py-2 px-3 focus:outline-none focus:ring-2 focus:ring-blue-500" />
                        <button
                            id="fetch-data-btn"
                            class="bg-blue-500 hover:bg-blue-700 text-white font-semibold rounded-md py-2 px-4 focus:outline-none focus:shadow-outline"
                            onclick="fetchStockData()"
                        >
                            Fetch Data
                        </button>
                        <div id="error-message" class="text-red-500"></div>
                    </div>
                </div>
                <div class="bg-white rounded-lg shadow-md p-4 md:p-6 mb-6 space-y-4">
                    <h2 class="text-xl font-semibold text-gray-800">Stock Overview</h2>
                    <div id="stock-overview" class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div class="bg-gray-50 rounded-md p-4">
                            <p class="text-gray-600 font-medium">Company Name:</p>
                            <p id="company-name" class="text-gray-800 font-semibold"></p>
                        </div>
                        <div class="bg-gray-50 rounded-md p-4">
                            <p class="text-gray-600 font-medium">Stock Symbol:</p>
                            <p id="stock-symbol" class="text-gray-800 font-semibold"></p>
                        </div>
                        <div class="bg-gray-50 rounded-md p-4">
                            <p class="text-gray-600 font-medium">Country:</p>
                            <p id="country" class="text-gray-800 font-semibold"></p>
                        </div>
                        <div class="bg-gray-50 rounded-md p-4">
                            <p class="text-gray-600 font-medium">Market Cap:</p>
                            <p id="market-cap" class="text-gray-800 font-semibold"></p>
                        </div>
                    </div>
                </div>
                <div class="bg-white rounded-lg shadow-md p-4 md:p-6 mb-6">
                    <h2 class="text-xl font-semibold text-gray-800 mb-4">Price and Volume Chart</h2>
                    <div id="price-volume-chart" class="w-full">
                    </div>
                </div>
                <div class="bg-white rounded-lg shadow-md p-4 md:p-6 mb-6">
                    <h2 class="text-xl font-semibold text-gray-800 mb-4">Technical Indicators</h2>
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div class="bg-gray-50 rounded-md p-4">
                            <p class="text-gray-600 font-medium">RSI:</p>
                            <p id="rsi" class="text-gray-800 font-semibold"></p>
                        </div>
                        <div class="bg-gray-50 rounded-md p-4">
                            <p class="text-gray-600 font-medium">MACD:</p>
                            <p id="macd" class="text-gray-800 font-semibold"></p>
                        </div>
                        <div class="bg-gray-50 rounded-md p-4">
                            <p class="text-gray-600 font-medium">Signal:</p>
                            <p id="signal" class="text-gray-800 font-semibold"></p>
                        </div>
                        <div class="bg-gray-50 rounded-md p-4">
                            <p class="text-gray-600 font-medium">20-Day MA:</p>
                            <p id="ma20" class="text-gray-800 font-semibold"></p>
                        </div>
                        <div class="bg-gray-50 rounded-md p-4">
                            <p class="text-gray-600 font-medium">50-Day MA:</p>
                            <p id="ma50" class="text-gray-800 font-semibold"></p>
                        </div>
                        <div class="bg-gray-50 rounded-md p-4">
                            <p class="text-gray-600 font-medium">100-Day MA:</p>
                            <p id="ma100" class="text-gray-800 font-semibold"></p>
                        </div>
                    </div>
                </div>
                <div class="bg-white rounded-lg shadow-md p-4 md:p-6 mb-6">
                    <h2 class="text-xl font-semibold text-gray-800 mb-4">Stock Recommendation</h2>
                    <div id="recommendation" class="bg-gray-50 rounded-md p-4 text-center">
                        <p class="text-gray-800 font-semibold">Loading...</p>
                    </div>
                </div>
                <div class="bg-white rounded-lg shadow-md p-4 md:p-6 mb-6">
                    <h2 class="text-xl font-semibold text-gray-800 mb-4">News Sentiment</h2>
                    <div id="news-sentiment" class="bg-gray-50 rounded-md p-4">
                        <p class="text-gray-800 font-semibold">Loading...</p>
                    </div>
                </div>
            </div>
            <script>
                function fetchStockData() {
                    const stockSymbol = document.getElementById("stock-input").value;
                    const errorMessage = document.getElementById("error-message");
                    if (!stockSymbol) {
                        errorMessage.textContent = "Please enter a stock symbol.";
                        return;
                    }
                    errorMessage.textContent = "";
                    const url = `/stock_data?symbol=${encodeURIComponent(stockSymbol)}`;
                    fetch(url)
                        .then(response => {
                            if (!response.ok) {
                                throw new Error(`HTTP error! status: ${response.status}`);
                            }
                            return response.json();
                        })
                        .then(data => {
                            updateUI(data);
                        })
                        .catch(error => {
                            console.error("Error fetching data:", error);
                            errorMessage.textContent = "Failed to fetch data. Please check the stock symbol and try again.";
                            document.getElementById("company-name").textContent = "";
                            document.getElementById("stock-symbol").textContent = "";
                            document.getElementById("country").textContent = "";
                            document.getElementById("market-cap").textContent = "";
                            document.getElementById("rsi").textContent = "";
                            document.getElementById("macd").textContent = "";
                            document.getElementById("signal").textContent = "";
                            document.getElementById("ma20").textContent = "";
                            document.getElementById("ma50").textContent = "";
                            document.getElementById("ma100").textContent = "";
                            document.getElementById("recommendation").textContent = "Loading...";
                            document.getElementById("news-sentiment").textContent = "Loading...";
                            Plotly.newPlot("price-volume-chart", [], {});
                        });
                }
                function updateUI(data) {
                    document.getElementById("company-name").textContent = data.overview.company_name;
                    document.getElementById("stock-symbol").textContent = data.overview.symbol;
                    document.getElementById("country").textContent = data.overview.country;
                    document.getElementById("market-cap").textContent = data.overview.market_cap;
                    document.getElementById("rsi").textContent = data.technical_indicators.rsi;
                    document.getElementById("macd").textContent = data.technical_indicators.macd;
                    document.getElementById("signal").textContent = data.technical_indicators.signal;
                    document.getElementById("ma20").textContent = data.technical_indicators.ma20;
                    document.getElementById("ma50").textContent = data.technical_indicators.ma50;
                    document.getElementById("ma100").textContent = data.technical_indicators.ma100;
                    document.getElementById("recommendation").textContent = data.recommendation;
                    document.getElementById("news-sentiment").textContent = data.news_sentiment;
                    const fig = make_subplots(rows=2, cols=1, shared_xaxes=true, vertical_spacing=0.06, row_heights=[0.7, 0.3]);
                    fig.add_trace(go.Candlestick(x=data.prices.date, open=data.prices.open, high=data.prices.high, low=data.prices.low, close=data.prices.close, name="Price"), row=1, col=1);
                    fig.add_trace(go.Bar(x=data.volumes.date, y=data.volumes.volume, name="Volume", marker_color='rgba(158,202,225,0.6)'), row=2, col=1);
                    fig.update_layout({
                        title: { text: `${data.overview.company_name} - Price and Volume`, yanchor: 'top', xanchor: 'center', y: 0.95, x: 0.5 },
                        margin: { l: 20, r: 20, t: 60, b: 20 },
                        showlegend: false,
                        plot_bgcolor: 'white',
                        xaxis_rangeslider_visible: false,
                    });
                    fig.update_xaxes({ gridcolor: 'lightgray', tickformat: '%Y-%m-%d', title_text: "Date", row: 1 });
                    fig.update_yaxes({ gridcolor: 'lightgray', title_text: "Price (USD)", row: 1 });
                    fig.update_xaxes({ gridcolor: 'lightgray', tickformat: '%Y-%m-%d', title_text: "Date", row: 2 });
                    fig.update_yaxes({ gridcolor: 'lightgray', title_text: "Volume", row: 2 });
                    Plotly.newPlot("price-volume-chart", fig, { responsive: true });
                }
            </script>
        </body>
    </html>
    """

def get_ticker_symbol(stock_name):
    """
    Fetches the ticker symbol for a given stock name using yfinance.

    Args:
        stock_name (str): The name or symbol of the stock.

    Returns:
        str: The ticker symbol if found, otherwise None.
    """
    try:
        search_results = yf.Ticker(stock_name)
        if search_results:
            return search_results.ticker
        else:
            return None
    except Exception as e:
        logger.error(f"Error fetching ticker symbol for {stock_name}: {e}")
        return None

class StockDataFetcher:
    """
    Fetches stock data from various sources, including yfinance and NSE.
    """

    def __init__(self):
        self.start_date = date(2023, 1, 1)
        self.end_date = date.today()

    def get_stock_data(self, symbol):
        """
        Fetches historical stock prices and volumes from yfinance.

        Args:
            symbol (str): The stock symbol.

        Returns:
            tuple: A tuple containing two pandas DataFrames:
                   - prices: DataFrame with date, open, high, low, close prices.
                   - volumes: DataFrame with date and volume.
                   Returns (None, None) if data fetching fails.
        """
        try:
            stock_data = yf.download(symbol, start=self.start_date, end=self.end_date)
            if stock_data.empty:
                logger.warning(f"No data found for symbol: {symbol} from yfinance")
                return pd.DataFrame(), pd.DataFrame()

            prices_df = stock_data[['Open', 'High', 'Low', 'Close']].reset_index()
            prices_df.columns = ['date', 'open', 'high', 'low', 'close']
            volumes_df = stock_data[['Volume']].reset_index()
            volumes_df.columns = ['date', 'volume']
            return prices_df, volumes_df
        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {e}")
            return None, None

    def get_company_overview(self, symbol):
        """
        Fetches company overview information from yfinance.

        Args:
            symbol (str): The stock symbol.

        Returns:
            dict: A dictionary containing company name, symbol, country,
                  market capitalization. Returns None if fetching fails.
        """
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            if not info:
                logger.warning(f"No overview information found for symbol: {symbol}")
                return {}
            overview = {
                "company_name": info.get('longName', 'N/A'),
                "symbol": info.get('symbol', 'N/A'),
                "country": info.get('country', 'N/A'),
                "market_cap": info.get('marketCap', 'N/A'),
            }
            return overview
        except Exception as e:
            logger.error(f"Error fetching company overview for {symbol}: {e}")
            return None

    def get_index_data(self, index_symbol):
        """
        Fetches historical data for a specific index from NSE.

        Args:
            index_symbol (str): The symbol of the index (e.g., "NIFTY 50").

        Returns:
            pandas.DataFrame: DataFrame with historical index data,
                            or None if fetching fails.
        """
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=365)
            index_data = get_history(symbol=index_symbol, start=start_date, end=end_date)
            return index_data
        except Exception as e:
            logger.error(f"Error fetching index data for {index_symbol}: {e}")
            return None

    def get_nse_stock_data(self, symbol):
        """
        Fetches historical stock data from NSE.

        Args:
            symbol (str): The stock symbol.

        Returns:
            pandas.DataFrame: DataFrame with historical stock data,
                            or None if fetching fails.
        """
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=365)
            stock_data = get_history(symbol=symbol, start=start_date, end=end_date)
            return stock_data
        except Exception as e:
            logger.error(f"Error fetching NSE stock data for {symbol}: {e}")
            return None
            
    def get_news_sentiment(self, company_name):
        """
        Fetches news headlines and calculates a basic sentiment score based on keywords.

        Args:
            company_name (str): The name of the company to fetch news for.

        Returns:
            str: A sentiment description (Positive, Negative, Neutral),
                 or "No news found" if no relevant news is found,
                 or None if an error occurs.
        """
        try:
            query = f"{company_name} stock news"
            url = f"https://www.google.com/search?q={query}&tbm=nws"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            news_items = soup.find_all("div", class_="Gx5Zad")

            if not news_items:
                return "No news found"

            positive_keywords = ["good", "positive", "strong", "growth", "increase", "rise", "beat", "exceed", "record", "high", "profitable"]
            negative_keywords = ["bad", "negative", "weak", "decline", "decrease", "fall", "miss", "below", "low", "loss", "unprofitable"]
            
            total_positive_mentions = 0
            total_negative_mentions = 0

            for item in news_items:
                title_element = item.find("div", class_="mCBhHd")
                if title_element:
                    title_text = title_element.text.lower()
                    for positive_keyword in positive_keywords:
                        if positive_keyword in title_text:
                            total_positive_mentions += 1
                    for negative_keyword in negative_keywords:
                        if negative_keyword in title_text:
                            total_negative_mentions += 1

            if total_positive_mentions > total_negative_mentions:
                return "Positive"
            elif total_negative_mentions > total_positive_mentions:
                return "Negative"
            else:
                return "Neutral"

        except Exception as e:
            logger.error(f"Error fetching news sentiment for {company_name}: {e}")
            return None

class TechnicalAnalyzer:
    """
    Calculates technical indicators for stock data.
    """

    def calculate_rsi(self, prices, window=14):
        """
        Calculates the Relative Strength Index (RSI).

        Args:
            prices (pandas.Series): A pandas Series of closing prices.
            window (int): The window period for the RSI calculation.

        Returns:
            float: The RSI value, or None if calculation fails.
        """
        try:
            delta = prices.diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)

            ema_up = up.rolling(window=window).mean()
            ema_down = down.rolling(window=window).mean()

            rs = ema_up / ema_down
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1]
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return None

    def calculate_macd(self, prices, fast_period=12, slow_period=26, signal_period=9):
        """
        Calculates the Moving Average Convergence Divergence (MACD).

        Args:
            prices (pandas.Series): A pandas Series of closing prices.
            fast_period (int): The period for the fast EMA.
            slow_period (int): The period for the slow EMA.
            signal_period (int): The period for the signal line EMA.

        Returns:
            tuple: A tuple containing the last MACD value and the last signal line value,
                   or (None, None) if calculation fails.
        """
        try:
            ema_fast = prices.ewm(span=fast_period).mean()
            ema_slow = prices.ewm(span=slow_period).mean()
            macd = ema_fast - ema_slow
            signal = macd.ewm(span=signal_period).mean()
            return macd.iloc[-1], signal.iloc[-1]
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return None, None

    def calculate_moving_averages(self, prices, windows=[20, 50, 100]):
        """
        Calculates moving averages for the given prices.

        Args:
            prices (pandas.Series): A pandas Series of closing prices.
            windows (list): A list of window periods for the moving averages.

        Returns:
            dict: A dictionary where keys are window sizes (e.g., 'ma20')
                  and values are the last moving average values.
                  Returns an empty dict if calculation fails.
        """
        try:
            moving_averages = {}
            for window in windows:
                ma = prices.rolling(window=window).mean()
                moving_averages[f"ma{window}"] = ma.iloc[-1]
            return moving_averages
        except Exception as e:
            logger.error(f"Error calculating moving averages: {e}")
            return {}

class StockRecommender:
    """
    Provides stock recommendations based on technical indicators.
    """

    def __init__(self):
        self.recommendation_rules = {
            "RSI": {"buy": 30, "sell": 70},
            "MACD": {"buy": 0, "sell": 0},
            "MA": {
                "buy_fast_above_slow": True,
                "sell_fast_below_slow": True,
            },
        }
        self.previous_recommendations = {}

    def get_recommendation(self, indicators, prices, symbol):
        """
        Generates a stock recommendation based on technical indicators
        and price trends, considering previous recommendations to avoid
        frequent changes.

        Args:
            indicators(dict): A dictionary of technical indicators
                               (e.g., {'rsi': 60, 'macd': 0.1, 'signal': 0.2, 'ma20': 150, 'ma50': 145, 'ma100': 140}).
            prices (pandas.DataFrame): DataFrame with price data (for MA comparison).
            symbol (str): The stock symbol.

        Returns:
            str: A recommendation ("Buy", "Sell", "Hold"),
                 or None if a recommendation cannot be made.
        """
        recommendation = "Hold"

        if not indicators or not prices.any().any():
            return "Hold"

        rsi = indicators.get("rsi")
        macd = indicators.get("macd")
        signal = indicators.get("signal")
        ma20 = indicators.get("ma20")
        ma50 = indicators.get("ma50")
        ma100 = indicators.get("ma100")
        last_close = prices['close'].iloc[-1]

        if rsi is not None and macd is not None and signal is not None and ma20 is not None and ma50 is not None:
            if rsi < self.recommendation_rules["RSI"]["buy"]:
                recommendation = "Buy"
            elif rsi > self.recommendation_rules["RSI"]["sell"]:
                recommendation = "Sell"

            if macd > signal and recommendation != "Sell":
                recommendation = "Buy"
            elif macd < signal:
                recommendation = "Sell"

            if ma20 > ma50 and ma50 > ma100:
                recommendation = "Buy"
            elif ma20 < ma50 and ma50 < ma100:
                recommendation = "Sell"

            if recommendation == "Buy" and not (last_close > ma20 and last_close > ma50 and last_close > ma100):
                recommendation = "Hold"
            elif recommendation == "Sell" and not (last_close < ma20 and last_close < ma50 and last_close < ma100):
                recommendation = "Hold"

        if symbol in self.previous_recommendations:
            prev_recommendation = self.previous_recommendations[symbol]
            if recommendation != prev_recommendation:
                if (rsi is not None and (rsi < 20 or rsi > 80)) or (macd is not None and signal is not None and abs(macd - signal) > 0.5):
                    self.previous_recommendations[symbol] = recommendation
                else:
                    recommendation = prev_recommendation
            else:
                recommendation = prev_recommendation
        else:
            self.previous_recommendations[symbol] = recommendation

        return recommendation

# Initialize components
data_fetcher = StockDataFetcher()
analyzer = TechnicalAnalyzer()
recommender = StockRecommender()

# Dash app with Bootstrap theme
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.H1("Stock Analysis Dashboard", className="text-center my-4 text-4xl font-semibold text-blue-600"),
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Label("Enter Stock Symbol (e.g., AAPL, GOOG, RELI.NS):", className="font-medium"),
                dcc.Input(id="stock-input", type="text", placeholder="Stock Symbol", className="w-full p-2 border rounded"),
                html.Button("Fetch Data", id="fetch-button", n_clicks=0, className="mt-2 bg-blue-500 text-white p-2 rounded hover:bg-blue-700"),
                html.Div(id="error-message", className="text-red-500 mt-2")
            ], width=12)
        ]),
        dbc.Row([
            dbc.Col([
                html.H2("Stock Overview", className="text-xl font-semibold mt-4"),
                html.Div(id="stock-overview")
            ], width=12)
        ]),
        dbc.Row([
            dbc.Col([
                html.H2("Price and Volume Chart", className="text-xl font-semibold mt-4"),
                dcc.Graph(id="price-volume-chart")
            ], width=12)
        ]),
        dbc.Row([
            dbc.Col([
                html.H2("Technical Indicators", className="text-xl font-semibold mt-4"),
                html.Div(id="technical-indicators")
            ], width=12)
        ]),
        dbc.Row([
            dbc.Col([
                html.H2("Recommendation", className="text-xl font-semibold mt-4"),
                html.Div(id="recommendation", className="text-center p-4 bg-gray-100 rounded")
            ], width=12)
        ]),
        dbc.Row([
            dbc.Col([
                html.H2("News Sentiment", className="text-xl font-semibold mt-4"),
                html.Div(id="news-sentiment", className="p-4 bg-gray-100 rounded")
            ], width=12)
        ])
    ])
])

@callback(
    [Output("stock-overview", "children"),
     Output("price-volume-chart", "figure"),
     Output("technical-indicators", "children"),
     Output("recommendation", "children"),
     Output("news-sentiment", "children"),
     Output("error-message", "children")],
    [Input("fetch-button", "n_clicks")],
    [Input("stock-input", "value")]
)
def update_dashboard(n_clicks, symbol):
    if not n_clicks or not symbol:
        return [], {}, [], "Enter a stock symbol and click Fetch Data", "Loading...", ""

    # Fetch data
    prices_df, volumes_df = data_fetcher.get_stock_data(symbol)
    overview = data_fetcher.get_company_overview(symbol)

    if prices_df is None or prices_df.empty or overview is None:
        return [], {}, [], "Failed to retrieve data", "No news found", "Invalid symbol or data unavailable"

    close_prices = prices_df['close']
    rsi = analyzer.calculate_rsi(close_prices)
    macd, signal = analyzer.calculate_macd(close_prices)
    moving_averages = analyzer.calculate_moving_averages(close_prices)
    indicators = {"rsi": rsi, "macd": macd, "signal": signal, **moving_averages}
    recommendation = recommender.get_recommendation(indicators, prices_df, symbol)
    news_sentiment = data_fetcher.get_news_sentiment(overview['company_name']) or "No news found"

    # Stock Overview
    overview_content = [
        html.P(f"Company Name: {overview['company_name']}"),
        html.P(f"Symbol: {overview['symbol']}"),
        html.P(f"Country: {overview['country']}"),
        html.P(f"Market Cap: {overview['market_cap']}")
    ]

    # Price and Volume Chart
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=prices_df['date'], open=prices_df['open'], high=prices_df['high'],
                                 low=prices_df['low'], close=prices_df['close'], name="Price"), row=1, col=1)
    fig.add_trace(go.Bar(x=volumes_df['date'], y=volumes_df['volume'], name="Volume", marker_color='rgba(158,202,225,0.6)'), row=2, col=1)
    fig.update_layout(title=f"{overview['company_name']} - Price and Volume", showlegend=False, xaxis_rangeslider_visible=False)

    # Technical Indicators
    indicators_content = [
        html.P(f"RSI: {rsi:.2f}" if rsi else "RSI: N/A"),
        html.P(f"MACD: {macd:.2f}" if macd else "MACD: N/A"),
        html.P(f"Signal: {signal:.2f}" if signal else "Signal: N/A"),
        html.P(f"20-Day MA: {moving_averages.get('ma20', 'N/A'):.2f}" if moving_averages.get('ma20') else "20-Day MA: N/A"),
        html.P(f"50-Day MA: {moving_averages.get('ma50', 'N/A'):.2f}" if moving_averages.get('ma50') else "50-Day MA: N/A"),
        html.P(f"100-Day MA: {moving_averages.get('ma100', 'N/A'):.2f}" if moving_averages.get('ma100') else "100-Day MA: N/A")
    ]

    # Recommendation (Map to Bullish/Bearish)
    rec_text = f"{recommendation} ({'Bullish' if recommendation == 'Buy' else 'Bearish' if recommendation == 'Sell' else 'Neutral'})"

    return overview_content, fig, indicators_content, rec_text, news_sentiment, ""

if __name__ == '__main__':
    if not os.path.exists("recommendations.csv"):
        df = pd.DataFrame(columns=["Symbol", "Recommendation"])
        initial_stocks = ["AAPL", "GOOG", "MSFT", "AMZN", "TSLA", "RELI.NS", "TCS.NS", "INFY.NS"]
        for stock in initial_stocks:
            df = pd.concat([df, pd.DataFrame([{"Symbol": stock, "Recommendation": "Hold"}])], ignore_index=True)
        df.to_csv("recommendations.csv", index=False)

    app.run(debug=True)