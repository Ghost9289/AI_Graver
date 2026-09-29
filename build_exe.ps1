# Run after installing PyInstaller: py -m pip install pyinstaller
py -m PyInstaller --noconfirm --clean AI_Graver.spec
& "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe" installer.iss
