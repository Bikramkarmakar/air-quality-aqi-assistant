@echo off
echo Starting Air Quality AQI Assistant...
cd /d "d:\Important SSD\IBM INTERNSHIP\New Internship\Air Quality AQI Assistant"
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
python -m streamlit run app.py --server.port 8501 --browser.gatherUsageStats false
pause
