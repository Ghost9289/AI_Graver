#define MyAppName "AI Graver"
#ifndef MyAppVersion
  #define MyAppVersion "0.7.0"
#endif
#define MyAppPublisher "AI Graver"
#define MyAppExeName "AI_Graver.exe"

[Setup]
AppId={{A58EFC1F-AC72-4C23-9D51-B334E8B9FF96}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\AI Graver
DefaultGroupName=AI Graver
DisableProgramGroupPage=yes
OutputDir=installer
OutputBaseFilename=AI_Graver_Setup_v{#MyAppVersion}
Compression=lzma2/normal
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
UninstallDisplayName=AI Graver

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[Tasks]
Name: "desktopicon"; Description: "Создать ярлык на рабочем столе"; GroupDescription: "Дополнительные значки:"; Flags: unchecked

[Files]
Source: "dist\AI_Graver\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\AI Graver"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\AI Graver"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Запустить AI Graver"; Flags: nowait postinstall
