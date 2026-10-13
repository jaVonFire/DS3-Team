@echo off
title DS3 Team - Priorizacion Sismica Santander
cls
echo ====================================================================
echo      HACKATON SISMOS SANTANDER (NIVEL AVANZADO) - DS3 TEAM
echo      Iniciando Tablero de Control y Simulador Streamlit...
echo ====================================================================
echo.

if not exist .venv\Scripts\python.exe (
    echo [ERROR] No se encontro el entorno .venv.
    echo Asegurate de ejecutar este archivo desde la raiz del proyecto.
    pause
    exit /b 1
)

echo [OK] Entorno virtual detectado (.venv)
echo [OK] Abriendo aplicacion en http://localhost:8501...
echo.
.\.venv\Scripts\python.exe -m streamlit run app\app.py --browser.gatherUsageStats=false
pause
