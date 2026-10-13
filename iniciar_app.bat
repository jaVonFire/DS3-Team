@echo off
title DS3 Team - Priorizacion Sismica Santander
cls

:: Asegurar que el directorio de trabajo siempre sea la raiz del proyecto
cd /d "%~dp0"

echo ====================================================================
echo      HACKATON SISMOS SANTANDER (NIVEL AVANZADO) - DS3 TEAM
echo      Iniciando Tablero de Control y Simulador Streamlit...
echo ====================================================================
echo.

if not exist .venv\Scripts\python.exe (
    echo [ERROR] No se encontro el entorno .venv en: %~dp0
    echo Asegurate de ejecutar este archivo desde la raiz del proyecto.
    pause
    exit /b 1
)

echo [OK] Entorno virtual detectado (.venv)
echo [OK] Abriendo navegador en http://localhost:8501...
echo.

:: Abrir automaticamente el navegador en localhost:8501
start http://localhost:8501

:: Iniciar el servidor Streamlit
.\.venv\Scripts\python.exe -m streamlit run app\app.py --server.headless=false --browser.gatherUsageStats=false
pause

