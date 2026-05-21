@echo off

echo =====================================
echo Running Minute Bars
echo =====================================

E:\miniconda3\envs\bitpredict\python.exe E:\Information_Bars\bars\main.py ^
--source minute ^
--minute-csv E:\bitpredict\1min_ohlcv_raw.csv

echo =====================================
echo Minute Bars Done
echo =====================================

echo =====================================
echo Running Tick Bars
echo =====================================

E:\miniconda3\envs\bitpredict\python.exe E:\Information_Bars\bars\main.py ^
--source tick ^
--tick-csv E:\tick_data\BTCUSDT-aggTrades-ALL.csv

echo =====================================
echo ALL DONE
echo =====================================

pause