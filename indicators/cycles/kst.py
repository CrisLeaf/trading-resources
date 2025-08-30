import pandas as pd


def know_sure_thing(
        df: pd.DataFrame,
        roclen1: int = 10,
        roclen2: int = 15,
        roclen3: int = 20,
        roclen4: int = 30,
        smalen1: int = 10,
        smalen2: int = 10,
        smalen3: int = 10,
        smalen4: int = 15,
        signal_period: int = 9,
        column: str = 'Close'
    ) -> pd.DataFrame:
    """
    Calculates the Know Sure Thing (KST) indicator for a given DataFrame.
    The KST is a momentum oscillator that combines multiple rate-of-change (ROC) calculations smoothed over different
    periods. It helps identify major price cycle turns and provides buy or sell signals using its signal line.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        roclen1 (int, optional): Period for the first ROC calculation. Default is 10.
        roclen2 (int, optional): Period for the second ROC calculation. Default is 15.
        roclen3 (int, optional): Period for the third ROC calculation. Default is 20.
        roclen4 (int, optional): Period for the fourth ROC calculation. Default is 30.
        smalen1 (int, optional): Smoothing period for the first ROC. Default is 10.
        smalen2 (int, optional): Smoothing period for the second ROC. Default is 10.
        smalen3 (int, optional): Smoothing period for the third ROC. Default is 10.
        smalen4 (int, optional): Smoothing period for the fourth ROC. Default is 15.
        signal_period (int, optional): Period for calculating the signal line. Default is 9.
        column (str, optional): Name of the column containing price data. Default is 'Close'.

    Returns:
        pd.DataFrame: DataFrame with columns for KST and KST_signal.
    """
    close = df[column]
    
    # Calculate ROC
    roc1 = close.pct_change(periods=roclen1).rolling(window=smalen1, min_periods=smalen1).mean() * 100
    roc2 = close.pct_change(periods=roclen2).rolling(window=smalen2, min_periods=smalen2).mean() * 100
    roc3 = close.pct_change(periods=roclen3).rolling(window=smalen3, min_periods=smalen3).mean() * 100
    roc4 = close.pct_change(periods=roclen4).rolling(window=smalen4, min_periods=smalen4).mean() * 100
    
    # KST
    kst = roc1 + 2*roc2 + 3*roc3 + 4*roc4
    
    # Signal Line
    kst_signal = kst.rolling(window=signal_period, min_periods=signal_period).mean()
    
    return pd.DataFrame({
        'KST': kst,
        'KST_signal': kst_signal
    })


if __name__ == '__main__':
    import yfinance as yf
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = yf.download('USDCLP=X', start='2024-01-01')
    df.columns = df.columns.droplevel(1)

    kst = know_sure_thing(df)

    # Plots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.7, 0.3]
    )

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        line=dict(color='skyblue', width=1),
        name='Close'
    ), row=1, col=1)

    # Signals
    fig.add_trace(
        go.Scatter(
            y=kst['KST'],
            x=kst.index,
            mode='lines',
            line=dict(color='skyblue'),
            name='KST'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            y=kst['KST_signal'],
            x=kst.index,
            mode='lines',
            line=dict(color='coral'),
            name='Signal'
        ),
        row=2, col=1
    )

    fig.update_layout(
        template='plotly_dark',
        title='Signals Plot',
        xaxis2_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        plot_bgcolor='rgb(20, 20, 20)',
        paper_bgcolor='rgb(20, 20, 20)',
        font=dict(color='white'),
        height=900,
        width=1000
    )

    fig.show()
