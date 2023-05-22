@echo off

echo changing directory to %BRAINART%

CD %BRAINART%

echo Currently in: %cd%
echo ------------------------------------------



call .esp\Scripts\activate.bat

pause

echo.

echo Union Neurotech 2023. (www.unionneurotech.com) 
echo Brain Art Program.
echo For Inquires please contact unionneurotech@gmail.com
echo. 
pause

echo Starting BrainArt. This may take a moment to load.
echo.

streamlit run src/BrainArt.py