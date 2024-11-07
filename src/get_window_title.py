import pygetwindow as gw

# すべてのウィンドウタイトルを取得
window_titles = [window.title for window in gw.getAllWindows()]

# 取得したウィンドウタイトルを表示
for title in window_titles:
    print(title)