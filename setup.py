import sys
from cx_Freeze import setup, Executable

setup(name="Object Detecion Software",
      version="0.1",
      description="This software detects objects in realtime",
      executables=[Executable("object identifier.py")]
      )