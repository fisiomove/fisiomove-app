@echo off
echo ========================================
echo FISIOMOVE - GENERAZIONE PDF PRESENTAZIONI
echo ========================================
echo.
echo Generazione versione 1: ATHLETIC PERFORMANCE
echo (Stile dinamico, colori energetici, layout sportivo)
echo.
python generate_pdf_version1_athletic.py
echo.
echo ----------------------------------------
echo.
echo Generazione versione 2: CLINICAL EXCELLENCE  
echo (Stile elegante, professionale, scientifico)
echo.
python generate_pdf_version2_clinical.py
echo.
echo ========================================
echo COMPLETATO!
echo ========================================
echo.
echo PDF generati:
echo 1. Fisiomove_Athletic_Performance.pdf
echo 2. Fisiomove_Clinical_Excellence.pdf
echo.
echo Aprire entrambi i PDF per confrontare...
echo.
start Fisiomove_Athletic_Performance.pdf
start Fisiomove_Clinical_Excellence.pdf
echo.
pause
