# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 19:04:06 2017

@author: teogo
"""

from selenium import webdriver
import time
import os
import zipfile

#w_dir = 'C:/Users/Matteo/OneDrive/Z - Financial Big Data/project1'
w_dir = 'D:/OneDrive/Z - Financial Big Data/project1/Financial-Big-Data-1'


directory = 'D:/BIG DATA'

os.chdir(w_dir)
if not os.path.exists(directory):
    os.makedirs(directory)



driver = webdriver.Chrome()
driver.implicitly_wait(30)

ccys = ['eurusd','eurchf','eurgbp','eurjpy','euraud']

for ccy in ccys:
    for year_int in range(2003,2017): 
        year = str(year_int)
        for month_int in range(1,13): 
        
            month = str(month_int)
            url = 'http://www.histdata.com/download-free-forex-historical-data/?/ascii/tick-data-quotes/'\
                  +ccy+'/'+year+'/'+month
            driver.get(url)
            if month_int ==1:
                time.sleep(20)
            else:
                time.sleep(10)
                
            
            file_name = 'HISTDATA_COM_ASCII_'+ccy.upper()+'_T_'+year+month.zfill(2)+'.zip'
            link = driver.find_element_by_link_text(file_name)
            
            link.click()
        
    
    
    time.sleep(20)
    
    
    
    for year_int in range(2003,2017): 
        year = str(year_int)
        for month_int in range(1,13):
        
            month = str(month_int)
            file_name = 'HISTDATA_COM_ASCII_'+ccy.upper()+'_T'+year+month.zfill(2)+'.zip'
            
#            download_path = 'D:/OneDrive/Z - Financial Big Data/project1/project 1 data/'
            download_path = "C:/Users/teogo/Downloads/"
            
            with zipfile.ZipFile(download_path+file_name,"r") as zip_ref:
                zip_ref.extractall(directory)
                
            time.sleep(10)
            
driver.close()  
        