#!/usr/bin/env python
# coding: utf-8

# In[1]:


from binance.client import Client


# In[2]:


binance_api="**********************"
binance_secret="*********************************"
new_client=Client(api_key=binance_api,api_secret=binance_secret,tld="com", testnet=True)
new_client.get_account()
from binance import ThreadedWebsocketManager
from datetime import datetime, timedelta


# In[3]:


import pandas as ps
import yfinance as yf


# In[4]:


binance_api="***************************************"
binance_secret="*******************************************"
new_client=Client(api_key=binance_api,api_secret=binance_secret,tld="com", testnet=True)


# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.display.float_format= '{: .4f}' .format
plt.style.use("seaborn")


# In[71]:


start="2021-02-27"
end="2021-04-27"
symbol="BTC-USD"
df=yf.download(symbol,start,end,interval="1h")


# In[72]:


df


# In[73]:


btc_close=df.Close.copy()


# In[74]:


0.9920	1.0240	0.9980


# In[75]:


btc_close=btc_close.to_frame()


# In[76]:


btc_close.Close.plot(figsize=(15,8),fontsize=13)

plt.legend(fontsize=13)
plt.show()


# In[77]:


btc_close["Lag_price"]=btc_close.Close.shift(periods=3)


# In[131]:


btc_close


# In[79]:


btc_close.head(10)


# In[80]:


btc_close["Returns"]=btc_close.Close.pct_change(periods=1)


# In[81]:


btc_close


# In[82]:


btc_close.drop('Returns', inplace=True,axis=1)


# In[83]:


btc_close


# In[84]:


btc_close["position"]=0


# In[ ]:





# In[85]:


btc_close


# In[86]:


btc_close.position.value_counts()


# In[87]:


btc_close["Lag_position"]=0


# In[88]:


btc_close.Lag_position=btc_close.position.shift(periods=1)


# In[89]:


btc_close


# In[90]:


btc_close["growth"]=btc_close.Close.pct_change(periods=1)


# In[91]:


btc_close


# In[92]:


btc_close.iloc[2,0]


# In[93]:


#	0.9300	1.0200	0.9800
enter=0
for i in range(len(btc_close)-1):
    if btc_close.iloc[i,0]<=btc_close.iloc[i,1]*0.94:
        btc_close.iloc[i+1,2]=1
        enter=btc_close.iloc[i,0]
    elif btc_close.iloc[i,2]==1 and btc_close.iloc[i,0]>enter*0.98 and btc_close.iloc[i,0]<enter*1.02:
        btc_close.iloc[i+1,2]=1
    elif btc_close.iloc[i,0]<=enter*0.98:
        btc_close.iloc[i+1,2]=0
    elif btc_close.iloc[i,0]>=enter*1.02:
        btc_close.iloc[i+1,2]=0
    else:
        btc_close.iloc[i+1,2]=0
        


# In[94]:


btc_close.loc[btc_close.position== 1,'position'].count()


# In[95]:



x=1   
for i in range(len(btc_close)):
    if btc_close.iloc[i,2]==1:
        x+=x*btc_close.iloc[i,4]
        print(x)

        
  


# In[ ]:


class Trader():
    
    def __init__(self):
        
        
        self.available_intervals=["1m","3m","5m","15m","30m","1h","2h","4h","6h","12h","1d","3d","1w","1M"]
        
    def start_trading(self,historical_days):
        self.twm=ThreadedWebsocketManager()
        self.twm.start()
        
        
        self.get_most_recent()
        self.twm.start_kline_socket(callback=self.stream_candles, symbol="BTCUSDT",interval="1m")
    def get_most_recent(self):
        now=datetime.utcnow()
        past=str(now-timedelta(days=1/24))
        
        bars=new_client.get_historical_klines(symbol="BTCUSDT",interval="1m",start_str=past,end_str=None,limit=1000)
    
        df=pd.DataFrame(bars)
        df["Date"]=pd.to_datetime(df.iloc[:,0],unit="ms")
        df.columns=["Open time","Open","High","Low","Close","Volume","Clos Time","Quote Asset Volume",
                "Number of Trades","Taker Buy Base Asset Volume","Taker Buy Quote Asset Volume","Ignore","Date"]
        df=df[["Date","Open","High","Low","Close","Volume"]].copy()
        df.set_index("Date",inplace=True)
        for column in df.columns:
            df[column]=pd.to_numeric(df[column],errors="coerce")
        df["Complete"]=[True for row in range(len(df)-1)]+[False]   
    
        self.data=df
    
            
    def stream_candles(self,msg):
        event_time=pd.to_datetime(msg["E"],unit="ms")
        start_time=pd.to_datetime(msg["k"]["t"],unit="ms")
        first=    float(msg["k"]["o"])
        high=     float(msg["k"]["h"])
        low=      float(msg["k"]["l"])
        close=    float(msg["k"]["c"])
        volume=   float(msg["k"]["v"])
        complete= float(msg["k"]["x"])
        
        print(".", end="",flush=True)
        
        self.data.loc[start_time]=[first,high,low,close,volume,complete]
        
        if complete==True:
            self.define_strategy()
            self.executetrades()
    def executetrades(self):
        order= new_client.create_order(symbol="BTCUSDT",side="BUY",type="MARKET",quantity=0.1)
            
    def define_strategy(self):
        df=self.data.copy()
 


# In[ ]:





# In[ ]:





# In[195]:


btc_close.position


# In[ ]:


btc_close.iloc[i,1]


# In[32]:


def backtest(parameters):
    enter=0
    x=1  
    for i in range(len(btc_close)-1):
        if btc_close.iloc[i,0]<=btc_close.iloc[i,1]*parameters[0]:
            btc_close.iloc[i+1,2]=1
            enter=btc_close.iloc[i,0]
        elif btc_close.iloc[i,2]==1 and btc_close.iloc[i,0]>enter*parameters[2] and btc_close.iloc[i,0]<enter*parameters[1]:
            btc_close.iloc[i+1,2]=1
        elif btc_close.iloc[i,0]<=enter*parameters[2]:
            btc_close.iloc[i+1,2]=0
        elif btc_close.iloc[i,0]>=enter*parameters[1]:
            btc_close.iloc[i+1,2]=0
        else:
            btc_close.iloc[i+1,2]=0 
        
          
    for i in range(len(btc_close)):
        if btc_close.iloc[i,2]==1:
            x+=x*btc_close.iloc[i,4]
    return x
     

    
    


# In[ ]:


^#def backtest(parameters):
    
    pos=[0]*(len(btc_close)+1)
    prices=btc_close.Close.values.tolist()
    lprice=btc_close.Lag_price.values.tolist()
    gr=btc_close.growth.values.tolist()
    gro=[]
    enter=0
    a=1

    for i in range(len(prices)):
        if prices[i]<=lprice[i]*parameters[0]:
            pos[i+1]=1
            enter=prices[i]
       
        elif pos[i]==1 and prices[i]> enter*parameters[2] and prices[i]<enter*parameters[1]:
            pos[i+1]=1
        
        elif prices[i]<=enter*parameters[2]:
            pos[i+1]=0
        
        elif prices[i]>=enter*parameters[1]:
            pos[i+1]=0
       
        else:
            pos[i+1]=0
                for i in range(len(prices)):
        
        
    for i in range(len(prices)):
        if pos[i]==1:
            gro.append(gr[i])
            a+=a*(gr[i])
        
    
            
    return a  
            


# In[ ]:


grow=[]
enter=0


for i in range(len(prices)):
    if prices[i]<=lprice[i]*0.95:
        pos[i+1]=1
        enter=prices[i]
       
    elif pos[i]==1 and prices[i]> enter*0.995 and prices[i]<enter*1.03:
        pos[i+1]=1
        
    elif prices[i]<=enter*0.995:
        pos[i+1]=0
        
    elif prices[i]>=enter*1.03:
        pos[i+1]=0
       
    else:
        pos[i+1]=0
        
        
    
    


# In[241]:


#def backtest(parameters):
    
    pos=[0]*(len(btc_close)+1)
    prices=btc_close.Close.values.tolist()
    lprice=btc_close.Lag_price.values.tolist()
    gr=btc_close.growth.values.tolist()
    gro=[]
    enter=0
    a=1

    for i in range(len(prices)):
        if prices[i]<=lprice[i]*parameters[0]:
            pos[i+1]=1
            enter=prices[i]
       
        elif pos[i]==1 and prices[i]> enter*parameters[2] and prices[i]<enter*parameters[1]:
            pos[i+1]=1
        
        elif prices[i]<=enter*parameters[2]:
            pos[i+1]=0
        
        elif prices[i]>=enter*parameters[1]:
            pos[i+1]=0
       
        else:
            pos[i+1]=0
     

    for i in range(len(prices)):
        if pos[i]==1:
            gro.append(gr[i])
            a+=a*(gr[i])
        
    
            
    return a      
    
    


# In[242]:


pos=[0]*(len(btc_close)+1)
prices=btc_close.Close.values.tolist()
lprice=btc_close.Lag_price.values.tolist()
gr=btc_close.growth.values.tolist()


# In[243]:


pos


# In[244]:


lprice


# In[198]:


grow=[]
enter=0


for i in range(len(prices)):
    if prices[i]<=lprice[i]*0.95:
        pos[i+1]=1
        enter=prices[i]
       
    elif pos[i]==1 and prices[i]> enter*0.995 and prices[i]<enter*1.03:
        pos[i+1]=1
        
    elif prices[i]<=enter*0.995:
        pos[i+1]=0
        
    elif prices[i]>=enter*1.03:
        pos[i+1]=0
       
    else:
        pos[i+1]=0
        
        
    
    


# In[199]:


pos


# In[200]:


len(grow)


# In[251]:


pos.count(1)


# In[35]:


btc_close["new"]=pos


# In[ ]:


btc_close


# In[ ]:


pos


# In[36]:


len(pos)


# In[37]:


btc_close.new.plot(figsize=(15,8),fontsize=13)

plt.legend(fontsize=13)
plt.show()


# In[ ]:


pos


# In[ ]:


[i for i,val in enumerate(pos) if val==1]


# In[38]:


btc_close.Close[1148]


# In[39]:


btc_close.Close.take([8])


# In[252]:


gro=[]
x=1
for i in range(len(prices)):
    if pos[i]==1:
        gro.append(gr[i])
        x+=x*(gr[i])
        print(x)
        


# In[253]:


gro


# In[246]:


len(gro)


# In[247]:


x


# In[44]:


#def backtest(parameters):
    
    pos=[0]*(len(btc_close)+1)
    prices=btc_close.Close.values.tolist()
    lprice=btc_close.Lag_price.values.tolist()
    gr=btc_close.growth.values.tolist()
    gro=[]
    enter=0
    a=1

    for i in range(len(prices)):
        if prices[i]<=lprice[i]*parameters[0]:
            pos[i+1]=1
            enter=prices[i]
       
        elif pos[i]==1 and prices[i]> enter*parameters[2] and prices[i]<enter*parameters[1]:
            pos[i+1]=1
        
        elif prices[i]<=enter*parameters[2]:
            pos[i+1]=0
        
        elif prices[i]>=enter*parameters[1]:
            pos[i+1]=0
       
        else:
            pos[i+1]=0
     

    for i in range(len(prices)):
        if pos[i]==1:
            gro.append(gr[i])
            a+=a*(gr[i])
        
    
            
    return a      
    
    
    
    
    
    
    
    
    


# In[248]:


#def backtest(parameters):
    
    pos=[0]*(len(btc_close)+1)
    prices=btc_close.Close.values.tolist()
    lprice=btc_close.Lag_price.values.tolist()
    gr=btc_close.growth.values.tolist()
    gro=[]
    enter=0
    a=1

    for i in range(len(prices)):
        if prices[i]<=lprice[i]*parameters[0]:
            pos[i+1]=1
            enter=prices[i]
       
        elif pos[i]==1 and prices[i]> enter*parameters[2] and prices[i]<enter*parameters[1]:
            pos[i+1]=1
        
        elif prices[i]<=enter*parameters[2]:
            pos[i+1]=0
        
        elif prices[i]>=enter*parameters[1]:
            pos[i+1]=0
       
        else:
            pos[i+1]=0
     

    for i in range(len(prices)):
        if pos[i]==1:
            gro.append(gr[i])
            a+=a*(gr[i])
        
    
            
    return a      
    
    
    


# In[97]:


import warnings
warnings.filterwarnings('ignore')


# In[96]:


backtest(parameters=(0.94,1.1,0.986)) #0.9800	1.1000	0.9860


# In[35]:


xrange=np.arange(0.93,0.99,0.01)
yrange=np.arange(1.02,1.1,0.01)
zrange=np.arange(0.98,0.995,0.003)


# In[36]:


from itertools import product


# In[37]:


combinations=list(product(xrange,yrange,zrange))
combinations


# In[98]:


results=[]
for comb in combinations:
    results.append(backtest(parameters=comb))


# In[99]:


many_results=pd.DataFrame(data=combinations,columns=["x","y","z"])
many_results["performance"]=results


# In[100]:


len(combinations)


# In[101]:


many_results


# In[103]:


many_results.nsmallest(20,"performance")


# In[51]:


many_results.loc(many_results.performance<1)


# In[104]:


many_results.nlargest(20,"performance")


# In[44]:


len(combinations)


# In[105]:


many_results.groupby("x").performance.mean().plot()


# In[106]:


many_results.groupby("y").performance.mean().plot()


# In[107]:


many_results.groupby("z").performance.mean().plot()


# In[45]:


many_results.groupby("x").performance.mean().plot()


# In[46]:


many_results.groupby("y").performance.mean().plot()


# In[47]:


many_results.groupby("z").performance.mean().plot()


# In[ ]:


backtest((0.95,1.06,0.995))


# In[ ]:


class DeadDrop:
    
    def __init__(self,symbol,bar_length):
        self.symbol=symbol
        self.bar_length=bar_length
        self.data=pd.DataFrame(columns=["Open","High","Low","Close","Volume","Complete"])
        self.available_intervals=["1m","3m","5m","15m","30m","1h","2h","4h","6h","12h","1d","3d","1w","1M"]
    def start_trading(self):
        self.twm=ThreadedWebsocketManager()
        self.twm.start()
        
        if self.bar_length in self.available_intervals:
            self.twm.start_kline_socket(callback=self.stream_candles, symbol=self.symbol,interval=self.bar_length)
            
    def stream_candles(self,msg):
        event_time=pd.to_datetime(msg["E"],unit="ms")
        start_time=pd.to_datetime(msg["k"]["t"],unit="ms")
        first=    float(msg["k"]["o"])
        high=     float(msg["k"]["h"])
        low=      float(msg["k"]["l"])
        close=    float(msg["k"]["c"])
        volume=   float(msg["k"]["v"])
        complete= float(msg["k"]["x"])
        
        print("Time: {} | Price: {} ".format(event_time,close))
        
        self.data.loc[start_time]=[first,high,low,close,volume,complete]
    
    


# In[122]:


class Trader():
    
    def __init__(self):
        
        
        self.available_intervals=["1m","3m","5m","15m","30m","1h","2h","4h","6h","12h","1d","3d","1w","1M"]
        
    def start_trading(self,historical_days):
        
            self.twm=ThreadedWebsocketManager()
            self.twm.start()
            self.get_most_recent()
            self.twm.start_kline_socket(callback=self.stream_candles, symbol="BTCUSDT",interval="1h")
    def get_most_recent(self):
        now=datetime.utcnow()
        past=str(now-timedelta(days=30))
        
        bars=new_client.get_historical_klines(symbol="BTCUSDT",interval="1h",start_str=past,end_str=None)
    
        df=pd.DataFrame(bars)
        df["Date"]=pd.to_datetime(df.iloc[:,0],unit="ms")
        df.columns=["Open time","Open","High","Low","Close","Volume","Clos Time","Quote Asset Volume",
                "Number of Trades","Taker Buy Base Asset Volume","Taker Buy Quote Asset Volume","Ignore","Date"]
        df=df[["Date","Open","High","Low","Close","Volume"]].copy()
        df.set_index("Date",inplace=True)
        for column in df.columns:
            df[column]=pd.to_numeric(df[column],errors="coerce")
        df["Complete"]=[True for row in range(len(df)-1)]+[False]   
    
        self.data=df
    
            
    def stream_candles(self,msg):
        event_time=pd.to_datetime(msg["E"],unit="ms")
        start_time=pd.to_datetime(msg["k"]["t"],unit="ms")
        first=    float(msg["k"]["o"])
        high=     float(msg["k"]["h"])
        low=      float(msg["k"]["l"])
        close=    float(msg["k"]["c"])
        volume=   float(msg["k"]["v"])
        complete= float(msg["k"]["x"])
        
        print(".", end="",flush=True)
        
        self.data.loc[start_time]=[first,high,low,close,volume,complete]
        
        if complete==True:
            self.define_strategy()
            self.executetrades()
    def executetrades(self):
        order= new_client.create_order(symbol="BTCUSDT",side="BUY",type="MARKET",quantity=0.1)
            
    def define_strategy(self):
        df=self.data.copy()
 


# In[121]:



tradeer=Trader(data)
tradeer.start_trading(historical_days=30)


# In[169]:


tradeer.start_trading(historical_days=1/24)


# In[170]:


new_client.get_account()


# In[ ]:


#def backtest(parameters):
    
    pos=[0]*(len(btc_close)+1)
    prices=btc_close.Close.values.tolist()
    lprice=btc_close.Lag_price.values.tolist()
    gr=btc_close.growth.values.tolist()
    gro=[]
    enter=0
    a=1

    for i in range(len(prices)):
        if prices[i]<=lprice[i]*parameters[0]:
            pos[i+1]=1
            enter=prices[i]
       
        elif pos[i]==1 and prices[i]> enter*parameters[2] and prices[i]<enter*parameters[1]:
            pos[i+1]=1
        
        elif prices[i]<=enter*parameters[2]:
            pos[i+1]=0
        
        elif prices[i]>=enter*parameters[1]:
            pos[i+1]=0
       
        else:
            pos[i+1]=0
     

    for i in range(len(prices)):
        if pos[i]==1:
            gro.append(gr[i])
            a+=a*(gr[i])
        
    
            
    return a      
    
    
    


# In[65]:


trader=DeadDrop(symbol="BTCUSDT",bar_length="1m")


# In[66]:


trader


# In[67]:


trader.symbol


# In[68]:


trader.data


# In[69]:


trader.available_intervals


# In[73]:


trader.start_trading()


# In[94]:


trader.twm.stop()

trader.data
# In[75]:


trader.data


# In[63]:


now=datetime.utcnow()


# In[64]:


now


# In[78]:


now


# In[79]:


now


# In[80]:


now


# In[81]:


now;


# In[62]:


now


# In[83]:


past=now-timedelta(days=2)


# In[84]:


past


# In[85]:


str(past)


# In[88]:


def get_most_recent(symbol,interval,days):
    now=datetime.utcnow()
    past=str(now-timedelta(days=days))
    
    bars=new_client.get_historical_klines(symbol=symbol,interval=interval,start_str=past,end_str=None,limit=1000)
    
    df=pd.DataFrame(bars)
    df["Date"]=pd.to_datetime(df.iloc[:,0],unit="ms")
    df.columns=["Open time","Open","High","Low","Close","Volume","Clos Time","Quote Asset Volume",
                "Number of Trades","Taker Buy Base Asset Volume","Ignore","Date"]
    df=df[["Date","Open","High","Low","Close","Volume"]].copy()
    df.set_index("Date",inplace="True")
    for column in df.columns:
        df[column]=pd.to_numeric(df[column],errors="coerce")
    df["Complete"]=[True for row in range(len(df)-1)]+[False]   
    
    return df
    


# In[98]:


df=get_most_recent(symbol="BTCUSDT",interval="1m",days=2)


# In[90]:


df


# In[4]:





# In[63]:


3


# In[ ]:





# In[92]:


trader.twm.stop()


# In[ ]:


2


# ## 

# In[ ]:


2


# In[150]:


3


# In[38]:


3


# In[ ]:


class TradeForMe():
    
    def __init__(self):
        
        
        self.available_intervals=["1m","3m","5m","15m","30m","1h","2h","4h","6h","12h","1d","3d","1w","1M"]
        
    def start_trading(self,historical_days):
        
            self.twm=ThreadedWebsocketManager()
            self.twm.start()
            self.get_most_recent()
            self.twm.start_kline_socket(callback=self.stream_candles, symbol="BTCUSDT",interval="1h")
    def get_most_recent(self):
        now=datetime.utcnow()
        past=str(now-timedelta(days=30))
        
        bars=new_client.get_historical_klines(symbol="BTCUSDT",interval="1h",start_str=past,end_str=None)
    
        df=pd.DataFrame(bars)
        df["Date"]=pd.to_datetime(df.iloc[:,0],unit="ms")
        df.columns=["Open time","Open","High","Low","Close","Volume","Clos Time","Quote Asset Volume",
                "Number of Trades","Taker Buy Base Asset Volume","Taker Buy Quote Asset Volume","Ignore","Date"]
        df=df[["Date","Open","High","Low","Close","Volume"]].copy()
        df.set_index("Date",inplace=True)
        for column in df.columns:
            df[column]=pd.to_numeric(df[column],errors="coerce")
        df["Complete"]=[True for row in range(len(df)-1)]+[False]   
    
        self.data=df
    
            
    def stream_candles(self,msg):
        event_time=pd.to_datetime(msg["E"],unit="ms")
        start_time=pd.to_datetime(msg["k"]["t"],unit="ms")
        first=    float(msg["k"]["o"])
        high=     float(msg["k"]["h"])
        low=      float(msg["k"]["l"])
        close=    float(msg["k"]["c"])
        volume=   float(msg["k"]["v"])
        complete= float(msg["k"]["x"])
        
        print(".", end="",flush=True)
        
        self.data.loc[start_time]=[first,high,low,close,volume,complete]
        
        if complete==True:
            self.define_strategy()
            self.executetrades()
    def executetrades(self):
        order= new_client.create_order(symbol="BTCUSDT",side="BUY",type="MARKET",quantity=0.1)
            
    def define_strategy(self):
        df=self.data.copy()
 


# In[132]:


depth = new_client.get_order_book(symbol='BNBBTC')


# In[134]:


depth


# In[135]:


trades = new_client.get_historical_trades(symbol='BNBBTC')


# In[136]:


trades


# In[139]:


for kline in new_client.get_historical_klines_generator("BTCUSDT", Client.KLINE_INTERVAL_1MINUTE, "1 day ago UTC")
    print(kline)


# In[141]:


klines = new_client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1WEEK, "1 Jan, 2019")


# In[142]:


klines


# In[145]:


data=new_client.get_historical_klines('BTCUSDT',interval='1h',start_str='4h')


# In[146]:


df=pd.DataFrame(data,columns=["Date","Open","High","Low","Close","Volume","Clos Time","Quote Asset Volume",
                "Number of Trades","Taker Buy Base Asset Volume","Taker Buy Quote Asset Volume","Ignore"])


# In[149]:


df


# In[150]:


df=df[["Close"]].copy()


# In[151]:


df


# In[152]:


df["Lag_price"]=df.Close.shift(periods=3)


# In[153]:


df


# In[156]:


enter=df.iloc[3,0]


# In[157]:


enter


# In[160]:


new_client().get_


# In[184]:


def trade():
    x=0
    enter=0
    data=new_client.get_historical_klines('BTCUSDT',interval='1h',start_str='4h')
    btc=pd.DataFrame(data,columns=["Date","Open","High","Low","Close","Volume","Clos Time","Quote Asset Volume",
                "Number of Trades","Taker Buy Base Asset Volume","Taker Buy Quote Asset Volume","Ignore"])
    btc=btc[["Close"]].copy()
    btc["Lag_price"]=btc.Close.shift(periods=3)
    for column in btc.columns:
        btc[column]=pd.to_numeric(btc[column],errors="coerce")
    
    if btc.iloc[3,0]<=btc.iloc[3,0]*0.97 and x==0:
        order= new_client.create_order(symbol="BTCUSDT",side="BUY",type="MARKET",quantity=0.01)
        print("BUYING")
        x=1
        enter=btc.iloc[3,0]
    elif x==1 and btc.iloc[3,0]>=enter*1.07:
        order= new_client.create_order(symbol="BTCUSDT",side="SELL",type="MARKET",quantity=0.01)
        print("SELLING")
        x=0
        enter=0
        
    elif x==1 and btc.iloc[3,0]<=enter*0.988:
        order= new_client.create_order(symbol="BTCUSDT",side="SELL",type="MARKET",quantity=0.01)
        print("SELLING")
        x=0
        enter=0
    else:
        x=0
        enter=0
        
        
    
    


# In[185]:


while True:
    trade()


# In[164]:


order= new_client.create_order(symbol="BTCUSDT",side="BUY",type="MARKET",quantity=0.01)


# In[165]:


new_client.get_account()


# In[ ]:



pos=[0]*(len(btc_close)+1)
prices=btc_close.Close.values.tolist()
lprice=btc_close.Lag_price.values.tolist()
gr=btc_close.growth.values.tolist()
gro=[]
enter=0
a=1

for i in range(len(prices)):
    if prices[i]<=lprice[i]*parameters[0]:
        pos[i+1]=1
        enter=prices[i]
   
    elif pos[i]==1 and prices[i]> enter*parameters[2] and prices[i]<enter*parameters[1]:
        pos[i+1]=1
    
    elif prices[i]<=enter*parameters[2]:
        pos[i+1]=0
    
    elif prices[i]>=enter*parameters[1]:
        pos[i+1]=0
   
    else:
        pos[i+1]=0
 

for i in range(len(prices)):
    if pos[i]==1:
        gro.append(gr[i])
        a+=a*(gr[i])
    

        
return a      



