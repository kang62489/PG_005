## Modules
class CtrlMain:
    # Wires each popwins[key] to view_main's same-named btn_<key>, if present.
    def __init__(self, view_main, popwins: dict) -> None:
        # to get the buttons
        self.view_main = view_main
        # to get the popout windows
        self.popwins = popwins

        self.connect_shell_buttons()

    def connect_shell_buttons(self) -> None:
        for key in self.popwins:
            btn = getattr(self.view_main, f"btn_{key}", None)
            if btn is None:
                continue
            btn.clicked.connect(lambda _checked=False, k=key: self.show_popwin(k))

    def show_popwin(self, key: str) -> None:
        popwin = self.popwins[key]
        popwin.show()
        popwin.raise_()
        popwin.activateWindow()
