import ctypes
from ctypes import wintypes
from graver_bridge import _top_windows, _children, _window_text, _class_name, _process_id, user32

for hwnd in _top_windows(4788):
    print('TOP', hwnd, repr(_class_name(hwnd)), repr(_window_text(hwnd)))
    for child in _children(hwnd):
        print(' CHILD', child, repr(_class_name(child)), repr(_window_text(child)), user32.GetDlgCtrlID(child), 'visible=',bool(user32.IsWindowVisible(child)))
