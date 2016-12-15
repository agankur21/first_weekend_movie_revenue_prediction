from bs4 import BeautifulSoup
import urllib
import string
import pandas as pd

rootUrl='http://www.boxofficemojo.com'
dF=pd.read_csv('../data/after2007.csv',encoding='utf-8')
length=len(dF['title'])
dF.loc[:,'weekendRevenue']=pd.Series(['$0']*length,index=dF.index)
dF.loc[:,'weekendScreens']=pd.Series(['0']*length,index=dF.index)
dF.loc[:,'firstRevenue']=pd.Series(['$0']*length,index=dF.index)
dF.loc[:,'firstScreens']=pd.Series(['0']*length,index=dF.index)
dF.loc[:,'multipleTables']=pd.Series([False]*length,index=dF.index)
for i,row in dF.iterrows():
    #scrape the weekend data
    suffix=row['link']
    source=urllib.urlopen(rootUrl+suffix+"&adjust_yr=2016&view=chart&page=weekend").read()
    soup=BeautifulSoup(source,"lxml")
    revenueChart=soup.findAll("table",{"class":"chart-wide"})
    if(len(revenueChart) > 0):
        if(len(revenueChart)>1):
            dF.loc[i,"multipleTables"]=True
        revenueChart=revenueChart[0]
        revenueRows=revenueChart.findAll("tr")
        revenueCol=revenueRows[1].findAll("td");
        revenue=revenueCol[2].text
        nScreens=revenueCol[4].text
        dF.loc[i,"weekendRevenue"]=revenue;
        dF.loc[i,"weekendScreens"]=nScreens;
        #print revenue,nScreens
    else:
        dF.loc[i,"weekendRevenue"]="-1";
        dF.loc[i,"weekendScreens"]="-2";
        #print "weekend not found"
    

    # For the daily data
    source=urllib.urlopen(rootUrl+suffix+"&adjust_yr=2016&view=chart&page=daily").read()
    soup=BeautifulSoup(source,"lxml")
    revenueChart=soup.findAll("table",{"class":"chart-wide"})
    if(len(revenueChart) > 0):
        revenueChart=revenueChart[0]
        revenueRows=revenueChart.findAll("tr")
        revenueCol=revenueRows[1].findAll("td");
        revenueDaily=revenueCol[3].text
        nScreensDaily=revenueCol[6].text
        dF.loc[i,"firstRevenue"]=revenueDaily;
        dF.loc[i,"firstScreens"]=nScreensDaily;
        #print revenueDaily,nScreensDaily
    else:
        dF.loc[i,"firstRevenue"]="-3";
        dF.loc[i,"firstScreens"]="-4";
        #print "daily not found"


dF.to_csv('revenue.csv',encoding='utf-8',index=False);
