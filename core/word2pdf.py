import platform
import subprocess
from pathlib import Path
from docx2pdf import convert

def grant_mac_permissions(file_path):
    """Grant permissions for macOS by removing the quarantine attribute if it exists."""
    try:
        subprocess.run(['xattr', '-d', 'com.apple.quarantine', str(file_path)], check=True)
    except subprocess.CalledProcessError:
        # Ignore error if the attribute doesn't exist
        pass

def grant_windows_permissions(file_path):
    """Grant permissions for Windows."""
    command = f'icacls "{file_path}" /grant Everyone:F'
    subprocess.run(command, shell=True)

def convert_to_pdf(input_path):
    """Convert Word file to PDF."""
    try:
        input_path = Path(input_path).resolve()
        if platform.system() == "Windows":
            import pythoncom
            grant_windows_permissions(input_path)
            pythoncom.CoInitialize()
            convert(input_path)
        elif platform.system() == "Darwin":
            grant_mac_permissions(input_path)
            pythoncom.CoInitialize()
            convert(input_path)  # No use_comtypes argument needed
        else:
            print("Unsupported operating system")
    except Exception as e:
        print(f"Error during conversion: {e}")