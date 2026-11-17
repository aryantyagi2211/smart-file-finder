[Setup]
AppName=Smart File Finder
AppVersion=1.0
AppPublisher=Aryan Tyagi
DefaultGroupName=Smart File Finder
PrivilegesRequired=lowest
DefaultDirName={localappdata}\Programs\SmartFileFinder
OutputDir=..\dist_installer
OutputBaseFilename=SmartFileFinderSetup
Compression=lzma2/fast
SolidCompression=yes
SetupIconFile=..\app_icon.ico

[Files]
Source: "..\dist\SmartFileFinder\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs; Excludes: "_internal\torch\include\*;_internal\torch\testing\*;_internal\torch\test\*"

[Icons]
; Start Menu shortcut
Name: "{group}\Smart File Finder"; Filename: "{app}\SmartFileFinder.exe"
; Desktop shortcut (optional, user can uncheck during install)
Name: "{autodesktop}\Smart File Finder"; Filename: "{app}\SmartFileFinder.exe"; Tasks: desktopicon
; Optional startup shortcut; Inno Setup removes it during uninstall.
Name: "{userstartup}\Smart File Finder"; Filename: "{app}\SmartFileFinder.exe"; Tasks: startup

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional options:"; Flags: unchecked
Name: "startup"; Description: "Start Smart File Finder when Windows starts"; GroupDescription: "Additional options:"; Flags: unchecked

[Run]
Filename: "{app}\SmartFileFinder.exe"; Description: "Launch Smart File Finder now"; Flags: nowait postinstall skipifsilent